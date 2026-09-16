# ------------------------------------------------------------------------------
# "Accelerating Federated Learning with Split Learning on Locally Generated
# Losses" (Han, Bhatti, Lee, Moon -- FL-ICML 2021 workshop). AI-ASSISTED
# NO-CODE REIMPLEMENTATION per CLAUDE.md's rule for methods without a working
# released implementation; not validated against the paper's own reported
# MNIST/FMNIST/CIFAR-10 numbers.
#
# Read directly from the paper (Section 3, Eq. 1-5): the model is split as
# w = [w_C, w_S]. TWO separate local loss functions are defined -- one per
# side -- and updated IN PARALLEL each round, with NO backpropagated signal
# ever crossing the cut in either direction (this is the paper's whole point:
# "obviating server-to-clients communication"):
#   - Client-side: an auxiliary network a_C is attached to the client model's
#     output; each client k minimizes F_{C,k}(w_C, a_C) = local cross-entropy
#     of a_C(w_C(x)) against the true label (Eq. 1, 3, 4), updating w_C and
#     a_C together from this loss ALONE -- the client never waits for or
#     uses any signal from the server.
#   - Server-side: given the client's REAL (non-auxiliary) smashed data
#     g_{w*_C}(x), the server minimizes its own local loss F_{S,k}(w_S, w*_C)
#     = cross-entropy of w_S(g_{w*_C}(x)) against the label (Eq. 2, 5),
#     updating w_S alone via ordinary backprop -- no gradient is sent back to
#     the client.
#   - Aggregation: the fed server FedAvgs the client-side models and
#     auxiliary networks every round (paragraph after Eq. 4); by symmetry
#     with the paper's own SplitFed baseline comparison (Table 1/Fig. 2-4,
#     both sides FedAvg'd), this port also keeps one server-side copy per
#     client and FedAvgs them every round, mirroring this harness's
#     SplitFedv1 (`sl_multi_server`) aggregation pattern.
#
# Because no gradient ever crosses the cut, `comm_load_cut` here only ever
# counts the forward smashed-data upload -- no downlink gradient bytes at
# all, unlike every gradient-crossing baseline in this harness. This is the
# real, measurable communication reduction the paper's title claims.
#
# AUXILIARY NETWORK SIZE. The paper reports its auxiliary network needing only
# ~0.1-0.6% of the full model's parameter count, and that smallness is the
# point: a_C is a cheap local error source, not a second model. An earlier
# revision of this file reused the harness's default `ResNetAuxiliary`, which
# on ResNet-18 is 2.1M parameters -- 18.8% of the full model and THREE TIMES
# the size of the client-side model it hangs off. That single choice made this
# method report the highest weight-aggregation traffic of all twelve here
# (it FedAvgs the auxiliary every round) and badly inflated its client memory.
# It now uses a flattened single-linear head (~0.18% of the full model), inside
# the paper's stated range. Note it is deliberately NOT global-average-pooled:
# pooling first would make the head ~0.01%, an order of magnitude below the
# budget the paper reports needing.
#
# DISCLOSED EXTRAPOLATION: the paper specifies FedAvg over the client-side
# models and their auxiliary networks. Keeping one server-side replica per
# client and FedAvg-ing those too is this port's own inference "by symmetry"
# with the paper's SplitFed baseline -- it is NOT in the paper, and it accounts
# for a large share of this method's weight traffic. Recorded in the fidelity
# table rather than silently folded into the number.
# ------------------------------------------------------------------------------
import time
import copy
import torch

from algos import register_algorithm, aggregate_models, FLAlgorithm
from models import config_optimizer
from models.aux_models.simple_conv import GAPLinearHead

# ------------------------------------------------------------------------------
@register_algorithm("han_locloss")
class HanLocalLoss(FLAlgorithm):
    aggregated_auxiliary: torch.nn.Module
    aggregated_server: torch.nn.Module

    def __init__(self, *args, **kwargs):
        super(HanLocalLoss, self).__init__(*args, **kwargs)
        self.servers = [copy.deepcopy(self.server) for _ in self.clients]
        self.__install_paper_auxiliary_head()

    def __install_paper_auxiliary_head(self):
        '''Swap the harness-default auxiliary for the paper-sized one.

        Width is measured, not assumed: one eval-mode forward through the client
        model gives the cut activation's channel count, and the server model
        applied to it gives the class count -- so the head is correct at every
        cut without hardcoding. Eval mode because BatchNorm rejects a 1-sample
        batch while training, and the probe must not disturb running stats.
        '''
        sample = next(iter(self.test_loader))[0][:1].to(self.device)
        cm, sm = self.clients[0].model, self.server.model
        was = (cm.training, sm.training)
        cm.eval(); sm.eval()
        try:
            with torch.no_grad():
                z = cm(sample)
                # flattened, not pooled: a pooled head would be ~0.01% of the
                # model, an order of magnitude BELOW the 0.1-0.6% the paper
                # reports needing. Flattening lands at ~0.18%, inside its range.
                in_features = int(z.flatten(1).shape[1])
                num_classes = int(sm(z).shape[1])
        finally:
            cm.train(was[0]); sm.train(was[1])

        for c in self.clients:
            head = GAPLinearHead(in_features, num_classes,
                                 pool=False).to(self.device)
            head.set_optimizer_lr_scheduler(
                config_optimizer(head.parameters(), c.optimizer_options)
            )
            c.auxiliary_model = head
        for c in self.clients[1:]:
            c.auxiliary_model.load_state_dict(
                self.clients[0].auxiliary_model.state_dict()
            )

    def full_model(self, x):
        return self.aggregated_server(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        if t > 0:
            self.aggregated_server.train()
        if t > 0:
            self.aggregated_auxiliary.train()
        for c in self.clients:
            c.auxiliary_model.train()

    def special_models_eval_mode(self):
        self.aggregated_server.eval()
        self.aggregated_auxiliary.eval()
        for c in self.clients:
            c.auxiliary_model.eval()

    # auxiliary head lives on the client; the server host holds one
    # replica per client
    def client_side_modules(self, i):
        return [self.clients[i].model, self.clients[i].auxiliary_model]

    def client_side_optimizers(self, i):
        return [self.clients[i].optimizer,
                self.clients[i].auxiliary_model.optimizer]

    def server_side_modules(self):
        return [s.model for s in self.servers]

    def server_side_optimizers(self):
        return [s.optimizer for s in self.servers]

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            self.clients[i].auxiliary_model.optimizer.zero_grad()
            self.servers[i].optimizer.zero_grad()

            # client-side local loss (Eq. 1): auxiliary head on the client's
            # own output, updates w_C + a_C alone -- no server signal involved.
            splitting_output = self.clients[i].model(x)
            aux_out = self.clients[i].auxiliary_model.forward_inner(splitting_output)
            client_loss = self.criterion(aux_out, y)
            client_loss.backward()
            self.clients[i].optimizer.step()
            self.clients[i].auxiliary_model.optimizer.step()

            with torch.no_grad():
                _, predicted = torch.max(aux_out.data, 1)
                client_correct = predicted.eq(y.view_as(predicted)).sum().item()

        # server-side local loss (Eq. 2): real smashed data, real backprop,
        # updates w_S alone -- gradient never sent back to the client.
        smashed_data = splitting_output.detach()
        self.charge_cut_activation(smashed_data)
        self.charge_cut_labels(y)     # the server computes its own loss

        # No gradient is returned to the client, and the client's graph was
        # already released above, so `client_mem_held_across_cut_mb` is 0 here.
        with self.phase('server', i):
            server_out = self.servers[i].model(smashed_data)
            server_loss = self.criterion(server_out, y)
            server_loss.backward()
            self.servers[i].optimizer.step()

            with torch.no_grad():
                _, predicted = torch.max(server_out.data, 1)
                server_correct = predicted.eq(y.view_as(predicted)).sum().item()

        return {
            'l_loss': client_loss.item(),
            'l_acc': client_correct / y.size(dim=0),
            'g_loss': server_loss.item(),
            'acc': server_correct / y.size(dim=0),
            'loss': server_loss.item(),
        }

    def aggregate(self):
        ret_dict = self.aggregate_clients()

        t0 = time.time()
        self.aggregated_auxiliary = aggregate_models(
            [c.auxiliary_model for c in self.clients], self.agg_factor, self.device
        )
        agg_aux_weights = self.aggregated_auxiliary.state_dict()
        for c in self.clients:
            c.auxiliary_model.load_state_dict(agg_aux_weights)
            self.charge_weights_roundtrip(self.aggregated_auxiliary, 'aux')
        ret_dict['auxiliary_agg_compute_time'] = time.time() - t0

        t0 = time.time()
        self.aggregated_server = aggregate_models(
            [s.model for s in self.servers], self.agg_factor, self.device
        )
        agg_server_weights = self.aggregated_server.state_dict()
        for s in self.servers:
            s.model.load_state_dict(agg_server_weights)
            self.charge_weights_roundtrip(self.aggregated_server, 'server')
        ret_dict['server_agg_compute_time'] = time.time() - t0

        return ret_dict

# ------------------------------------------------------------------------------
