# ------------------------------------------------------------------------------
# Han, Bhatti, Lee, Moon, "Accelerating Federated Learning with Split Learning
# on Locally Generated Losses", ICML 2021 Workshop on Federated Learning
# (FL-ICML'21). The paper releases no code, so this is an AI-ASSISTED NO-CODE
# REIMPLEMENTATION (CLAUDE.md). Not yet checked against the paper's own
# numbers (Table 2, p.6); that reproduction is a separate task.
#
# The paper has no algorithm box: the protocol is prose in Sec. 3.2 (p.3 right
# column -> p.4 top-left). The step numbers are ours; docs/HAN_LOCLOSS_
# BREAKDOWN.md quotes each one.
#
# LINE MAP (paper step -> code)
#   0  init w0 = [w_C, w_S], a_C; a fed server (FedAvg only) and a main
#      server (trains w_S)                   -> __init__,
#                                               __install_paper_auxiliary_head
#   1  client k downloads w_C^t, a_C^t       -> aggregate() of the previous
#                                               round (download half of the
#                                               client / aux roundtrip). The
#                                               round-0 download of w_C^0, a_C^0
#                                               is not billed (clients start
#                                               identical); instead a download
#                                               is billed after the last round.
#                                               Same count (N x R), same bytes.
#   2  forward a minibatch, upload g_{w_C,k}(x) to the main server (labels go
#      with it: Fig. 1(c), p.2)              -> client_step: charge_cut_activation,
#                                               charge_cut_labels
#   3a Eq. (3), (4): CE of a_C(w_C(x)); SGD on w_C and a_C from this local loss
#      alone, no server signal               -> client_step, client phase
#   3b fed server: w_C = 1/K sum w_C,k and a_C = 1/K sum a_C,k
#                                            -> aggregate(): aggregate_clients +
#                                               aux FedAvg (client + aux roundtrip)
#   4a Eq. (5): per client, in parallel with 3, a server copy
#      w_S,k = w_S^t - eta grad F_S,k (CE at the true model output); no
#      gradient goes back ("the downlink communication for transmitting the
#      backpropagated signals is not required", p.6)
#                                            -> client_step, server phase on
#                                               self.servers[i]
#   4b main server: w_S = 1/K sum w_S,k (p.4 top-left)
#                                            -> aggregate(): server FedAvg, NOT
#                                               charged (it never leaves the
#                                               main server)
#   5  inference with [w_C^T, w_S^T]; a_C unused
#                                            -> full_model
#
# NOT SPECIFIED in the paper (our choice)
#   - Forward schedule. Steps 2-3 read per minibatch, but Table 1's latency
#     model forwards the whole local epoch with w_C^t once. We interleave per
#     batch: forward -> upload -> client step -> server step on that batch.
#     The bytes are the same under either reading. Accuracy may differ: Eq. (5)
#     writes the server input as g_{w_C,k^t}(x) (round-start model), while our
#     server trains on activations from the client model as it is after its
#     earlier local steps in the round.
#   - Auxiliary architecture: Sec. 3.1 (p.3) says "we adopt a MLP for a_C";
#     p.6 gives only "A single FC layer with 23,050 parameters" (CNN). 23,050 =
#     2304 x 10 + 10, so our reading is Linear(2304, 10) on the flattened
#     256x3x3 client output (inferred; the paper does not say so). We
#     use flatten + Linear(2048, 10) at ResNet-18's middle cut: 20,490 params,
#     ~0.18% of the model, inside the paper's 0.1-0.6%. Not pooled: a pooled
#     head (1,290 params) would sit an order of magnitude below that range.
#   - Optimizer state. Each client, aux head and server copy keeps its own
#     momentum buffer across rounds; FedAvg replaces the weights only.
#   - Weight decay and lr schedule: none. p.6 gives only "a fixed learning
#     rate of 0.01 and a momentum of 0.9"; the theory's eta_0/(1+t) (p.5) is
#     not used in its experiments.
#   - Aux-head init: PyTorch default.
#   - BatchNorm in FedAvg: every state_dict entry is averaged, running stats
#     included (harness aggregate_models).
#   - Rounds: the paper reads Table 2 at a simulated latency budget, not a round
#     count; the benchmark runs 11 rounds.
#   - Aggregation weights: the paper writes 1/K. We use the harness's
#     data-size weights, which are equal on equal (IID) shards.
#   - Labels are uploaded with every minibatch (the server's CE needs them).
#
# DIFFERENCES FROM THE PAPER (benchmark protocol, conventions.md)
#   - ResNet-18 at the middle cut, not the AlexNet-like CNN / VGG-11 (p.6).
#   - 10 clients, all of them every round; the paper samples K = 300 of N = 1000.
#   - Batch 256; the paper uses 10 (24 local steps per round here, 6 there).
#   - Non-IID (later runs): Dirichlet alpha = 0.5; the paper uses 5 label shards
#     per client.
#   - The aux head's FedAvg traffic is charged (W3/W4). Table 1 neglects it
#     ("|a| << |w|", p.4), but it is a real transfer.
#
# MEMORY. Per device: the main server holds the global w_S^t and the copy
# being trained for one client (+ its momentum). A parallel main server holds
# the other K - 1 copies at the same time: listed in server_side_extra_memory.
# The client's graph is freed before the upload, and no gradient comes back, so
# nothing is held across the cut on the client.
# ------------------------------------------------------------------------------
import time
import copy
import torch

from algos import register_algorithm, aggregate_models, FLAlgorithm
from models.aux_models.simple_conv import GAPLinearHead
from utils.memory import module_static_bytes, optimizer_state_bytes

# ------------------------------------------------------------------------------
@register_algorithm("han_locloss")
class HanLocalLoss(FLAlgorithm):
    aggregated_auxiliary: torch.nn.Module
    aggregated_server: torch.nn.Module

    def __init__(self, *args, **kwargs):
        super(HanLocalLoss, self).__init__(*args, **kwargs)
        # Step 4a: one server-side copy per client on the main server
        self.servers = [copy.deepcopy(self.server) for _ in self.clients]
        self.__install_paper_auxiliary_head()
        # the global models the fed / main server keep between rounds
        self.aggregated_auxiliary = copy.deepcopy(self.clients[0].auxiliary_model)
        self.aggregated_server = copy.deepcopy(self.server.model)

        # p.6: SGD, lr 0.01, momentum 0.9 (weight decay NOT SPECIFIED -> 0),
        # fixed lr, for the client models, the aux heads and the server copies
        sgd = lambda params: torch.optim.SGD(
            params, lr=float(self.cfg.get('lr', 0.01)),
            momentum=float(self.cfg.get('momentum', 0.9)),
            weight_decay=float(self.cfg.get('weight_decay', 0.0)))
        for c in self.clients:
            c.optimizer = sgd(c.model.parameters())
            c.lr_scheduler = None
            c.auxiliary_model.set_optimizer_lr_scheduler(
                sgd(c.auxiliary_model.parameters()))
        for s in self.servers:
            s.optimizer = sgd(s.model.parameters())
            s.lr_scheduler = None

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
            # its SGD optimizer is set in __init__
            c.auxiliary_model = GAPLinearHead(in_features, num_classes,
                                              pool=False).to(self.device)
        for c in self.clients[1:]:
            c.auxiliary_model.load_state_dict(
                self.clients[0].auxiliary_model.state_dict()
            )

    def full_model(self, x):
        return self.aggregated_server(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        self.aggregated_server.train()
        self.aggregated_auxiliary.train()
        for c in self.clients:
            c.auxiliary_model.train()

    def special_models_eval_mode(self):
        self.aggregated_server.eval()
        self.aggregated_auxiliary.eval()
        for c in self.clients:
            c.auxiliary_model.eval()

    # -- memory partition ---------------------------------------------------------
    # Client: its model + the aux head (+ their momentum buffers).
    # Server per device: the global w_S^t + the copy being trained for one
    # client (+ its momentum). The other copies a parallel main server holds at
    # the same time are listed in server_side_extra_memory.
    def client_side_modules(self, i):
        return [self.clients[i].model, self.clients[i].auxiliary_model]

    def client_side_optimizers(self, i):
        return [self.clients[i].optimizer,
                self.clients[i].auxiliary_model.optimizer]

    def server_side_modules(self):
        return [self.aggregated_server, self.servers[0].model]

    def server_side_optimizers(self):
        return [self.servers[0].optimizer]

    def server_side_extra_memory(self):
        others = len(self.servers) - 1
        if others <= 0:
            return []
        params, grads = module_static_bytes(self.servers[0].model)
        optim = optimizer_state_bytes(self.servers[0].optimizer)
        return [{
            'item': f'server copies of the other {others} clients '
                    f'(params + grads + momentum)',
            'mb': others * (params + grads + optim) / 1024 ** 2,
            'note': 'Step 4a, p.3: the main server updates one w_S,k per '
                    'client in parallel',
        }]

    def register_meter_params(self):
        # every server copy, not only the one counted per device, so no
        # parameter view (Linear saves weight.t()) is counted as an activation
        mods = [self.server.model, getattr(self, 'aggregated_client', None),
                self.aggregated_server, self.aggregated_auxiliary]
        for i in range(len(self.clients)):
            mods += self.client_side_modules(i) + [self.servers[i].model]
        self.meter.register_params(*mods)

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
            # Not charged: Step 4b (p.4) averages the copies on the main
            # server itself, so nothing crosses a network (the same rule as
            # SplitFedv1, FedSplitX and MU-SplitFed). The copies count as
            # server memory instead.
        ret_dict['server_agg_compute_time'] = time.time() - t0

        return ret_dict

# ------------------------------------------------------------------------------
