# ------------------------------------------------------------------------------
# DSL-Aux -- "Decoupled Split Learning via Auxiliary Loss"
# (Zihad, Owino, Tang & Huang, arXiv:2601.19261).
#
# PROVENANCE: AI-assisted no-code reimplementation per CLAUDE.md's rule for
# methods without a released implementation. It follows the PAPER directly --
# Algorithm 1 (p. 3) and the training procedure in SIII-B -- cross-checked
# against the `train_dgl` function of the partial third-party reference
# juniorfelix998/sl-fl-dgl (`cifar-100/sl/split-2/sl_dsl.py`), which likewise
# logs only a forward transmission at the cut ("DGL has NO backward comm from
# Server to Client").
#
# WHAT MAKES THIS METHOD DECOUPLED (and what an earlier revision of this file
# got backwards): the server computes dL/dz internally, but SIII-B is explicit
# that "this gradient is NOT transmitted to the client. It is only used for the
# server's weight updates", and SIII-C fixes lambda = 0 "to avoid any BP signals
# from the server to the client". So:
#   * the client's ONLY training signal is its local auxiliary loss;
#   * nothing is charged to `cut.grad_down` -- the cut carries activations
#     (and labels, since the global loss is evaluated server-side) one way only;
#   * the client finishes its backward BEFORE the server runs, so it holds no
#     autograd graph across the cut.
# These are exactly the paper's two headline claims (Obs. 2: ~50% less
# communication than conventional SL, since the backward half disappears;
# Obs. 3: up to 58% lower peak client memory, since no activations are retained
# for a cross-cut backward pass). Both are asserted in test/check_accounting.py.
#
# Note the paper also predicts DSL is *slower* per epoch than conventional SL
# (Obs. 4) -- the auxiliary head's forward/backward is extra client-side
# compute. A latency BELOW vanilla SL's would mean the aux head is not running.
#
# AUXILIARY HEAD: SIV-A specifies "a single fully-connected layer that maps the
# activations z to 10/100 classes, plus a softmax layer". This file therefore
# REPLACES the harness's default auxiliary model (which for `model=resnet18` is
# a whole mirrored ResNet stage, orders of magnitude larger than the paper's
# head and a serious distortion of both the client-memory and latency columns)
# with `LinearGradScalarAuxiliaryModel` -- literally `nn.Linear` + log_softmax.
# Its input width is derived from a dummy forward through the client model, so
# it is correct at every cut (shallow/middle/deep) without hardcoding.
#
# NON-FEDERATED: the paper is a single client/server split with no cross-client
# aggregation, so `aggregate()` averages nothing and `comm_load_weights` stays 0
# for the whole run -- per CLAUDE.md's metric definition ("~=0 for non-federated
# methods"). `num_clients=1` is the paper-faithful configuration.
# ------------------------------------------------------------------------------
import time
import torch

from algos import register_algorithm, FLAlgorithm
from models import config_optimizer
from models.aux_models.simple_conv import LinearGradScalarAuxiliaryModel

# ------------------------------------------------------------------------------
@register_algorithm("dsl_aux")
class DSLAux(FLAlgorithm):

    def __init__(self, *args, **kwargs):
        super(DSLAux, self).__init__(*args, **kwargs)
        # non-federated: no aggregation ever happens, so `aggregated_client`
        # just needs to exist for full_model()/eval_mode() from round 0.
        self.aggregated_client = self.clients[0].model
        self.__install_paper_auxiliary_head()

    # -- paper SIV-A: a single FC layer + softmax at the cut ---------------
    def __install_paper_auxiliary_head(self):
        '''Swap the harness-default auxiliary model for the paper's head.

        Width is measured, not assumed: a one-sample forward through the client
        model gives the cut activation, and the server model applied to it gives
        the class count. This keeps the head correct across cuts and datasets.
        '''
        sample = next(iter(self.test_loader))[0][:1].to(self.device)
        # eval mode for the probe: BatchNorm rejects a 1-sample batch while
        # training, and the probe must not disturb running stats either.
        was_training = (self.clients[0].model.training, self.server.model.training)
        self.clients[0].model.eval()
        self.server.model.eval()
        try:
            with torch.no_grad():
                z = self.clients[0].model(sample)
                n_input = int(z.flatten(1).shape[1])
                n_output = int(self.server.model(z).shape[1])
        finally:
            self.clients[0].model.train(was_training[0])
            self.server.model.train(was_training[1])

        for c in self.clients:
            head = LinearGradScalarAuxiliaryModel(
                n_input=n_input, n_output=n_output,
                server=self.server, device=self.device,
            ).to(self.device)
            head.set_optimizer_lr_scheduler(
                config_optimizer(head.parameters(), c.optimizer_options)
            )
            c.auxiliary_model = head

        # every client starts from the same head, as elsewhere in the harness
        for c in self.clients[1:]:
            c.auxiliary_model.load_state_dict(
                self.clients[0].auxiliary_model.state_dict()
            )

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        for c in self.clients:
            c.auxiliary_model.train()

    def special_models_eval_mode(self):
        for c in self.clients:
            c.auxiliary_model.eval()

    # the local auxiliary classifier head is resident on the client device
    def client_side_modules(self, i):
        return [self.clients[i].model, self.clients[i].auxiliary_model]

    def client_side_optimizers(self, i):
        return [self.clients[i].optimizer, self.clients[i].auxiliary_model.optimizer]

    # -- Algorithm 1 -------------------------------------------------------
    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        # --- Alg. 1 L4-L12: client forward + LOCAL backward ---------------
        # The client's update depends on nothing from the server. Its whole
        # graph is built and freed inside this block, so no activations are
        # retained across the cut (paper Obs. 3).
        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            self.clients[i].auxiliary_model.optimizer.zero_grad()

            z = self.clients[i].model(x)
            aux_out = self.clients[i].auxiliary_model.forward_inner(z.flatten(1))
            aux_loss = self.criterion(aux_out, y)
            aux_loss.backward()

            self.clients[i].optimizer.step()
            self.clients[i].auxiliary_model.optimizer.step()

            with torch.no_grad():
                _, aux_pred = torch.max(aux_out.data, 1)
                train_correct = aux_pred.eq(y.view_as(aux_pred)).sum().item()

        # --- Alg. 1 L8: send activations (one way, no gradient returns) ---
        smashed_data = z.detach()
        self.charge_cut_activation(smashed_data)
        self.charge_cut_labels(y)     # the global loss is evaluated server-side

        # --- Alg. 1 L13-L21: server forward + backward for its OWN weights -
        # dL/dz is computed here but never transmitted (SIII-B), so there is no
        # `charge_cut_gradient` and no backward into the client.
        with self.phase('server', i):
            self.server.optimizer.zero_grad()
            out = self.server.model(smashed_data)
            loss = self.criterion(out, y)
            loss.backward()
            self.server.optimizer.step()

        return {
            'acc': train_correct / y.size(dim=0),
            'loss': loss.item(),
            'aux_loss': aux_loss.item(),
        }

    def aggregate(self):
        # non-federated: no cross-client averaging of model or auxiliary
        # weights -- each client keeps training its own independent copy.
        t0 = time.time()
        self.aggregated_client = self.clients[0].model
        return {'client_agg_compute_time': time.time() - t0}

# ------------------------------------------------------------------------------
