# ------------------------------------------------------------------------------
# CSE-FSL (Mu & Shen, arXiv:2302.05599), run against the paper's pseudocode.
#
# Differences from the original implementation (changed after review):
#   D. WHICH batch is uploaded. The original code uploaded the smashed data of
#      the FIRST batch of every h-batch window (local_iter = 0, h, 2h, ...;
#      5 uploads per round at K = 24). The paper is internally ambiguous:
#        - Step 2 (p.3): "if current batch number m satisfies m mod h = 0"
#        - Eq.(6): one window is h steps, m = 0..h-1
#        - Eq.(8) and Fig. 2 ("When m=h"): the server receives
#          g_{x_c^{t,h-1}}(z) -- the client model AFTER h-1 updates, i.e. the
#          output on the h-th (LAST) batch of the window.
#      Eq.(8) and Fig. 2 agree with each other and with Eq.(6), so this file
#      uploads on the last batch of each window: local_iter = h-1, 2h-1, ...
#      (Step 2 read with m counted 1..h). At K = 24, h = 5 that is batches
#      5,10,15,20 -> 4 uploads per round (was 5), the same schedule as FSL-SAGE.
#
# Matches the paper unchanged:
#   - Step 1: x_c and a_c downloaded every round. The ledger charges the
#     download when the server broadcasts after aggregation (end of round t
#     for round t+1); the per-run COUNT is the same N*R as Step 1's.
#   - Step 4 / Eq.(9): x_c and a_c both FedAvg'd every round.
#   - Single server model, updated first-come-first-served; labels uploaded
#     with each smashed batch; no gradient ever sent back.
# ------------------------------------------------------------------------------
import time
import torch
from algos import register_algorithm, aggregate_models, FLAlgorithm
from models.aux_models import AuxiliaryModel

# ------------------------------------------------------------------------------
@register_algorithm("cse_fsl")
class CSEFSL(FLAlgorithm):
    aggregated_auxiliary : AuxiliaryModel

    def __init__(self, *args, **kwargs):
        super(CSEFSL, self).__init__(*args, **kwargs)

        self.server_update_interval = self.cfg.server_update_interval
        self.iters_per_epoch = [len(c.train_loader) for c in self.clients]

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        if t > 0: self.aggregated_auxiliary.train()
        for c in self.clients: c.auxiliary_model.train()

    def special_models_eval_mode(self):
        self.aggregated_auxiliary.eval()
        for c in self.clients: c.auxiliary_model.eval()

    # the auxiliary head is a real, permanent resident of the client device --
    # it is the price CSE-FSL pays for not needing a gradient back across the cut
    def client_side_modules(self, i):
        return [self.clients[i].model, self.clients[i].auxiliary_model]

    def client_side_optimizers(self, i):
        return [self.clients[i].optimizer, self.clients[i].auxiliary_model.optimizer]

    def should_probe_memory(self, t, i, j, k):
        # also sample the first upload (local_iter = h-1), or the server step
        # would never be measured
        return super().should_probe_memory(t, i, j, k) or (
            j == 0 and k == self.server_update_interval - 1
        )

    def client_step(self, rd_cl_ep_it, x, y):

        t, i, j, k = rd_cl_ep_it       # (round, client, epoch, iter)

        ret_dict = dict()
        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            self.clients[i].auxiliary_model.optimizer.zero_grad()

            # client feedforward
            splitting_output = self.clients[i].model(x)
            local_smashed_data = \
                splitting_output.clone().detach().requires_grad_(True)

            # client backpropagation against the LOCAL auxiliary head -- no
            # gradient ever crosses the cut, so nothing is charged here and
            # `client_mem_held_across_cut_mb` stays 0 for this method
            out = self.clients[i].auxiliary_model.forward_inner(local_smashed_data)
            loss = self.criterion(out, y)
            loss.backward()

            with torch.no_grad():
                local_loss = loss.item()
                _, predicted = torch.max(out.data, 1)
                local_correct = predicted.eq(y.view_as(predicted)).sum().item()
                ret_dict['l_loss'] = local_loss
                ret_dict['l_acc'] = local_correct / y.size(dim=0)

            self.clients[i].auxiliary_model.optimizer.step()
            splitting_output.backward(local_smashed_data.grad)
            self.clients[i].optimizer.step()

        # server model update. NOTE the clone is made AFTER the client phase
        # closes, so it is not attributed to the client's activation peak.
        local_iter = j * self.iters_per_epoch[i] + k
        # fix D: the LAST batch of each h-batch window (Eq.8 / Fig.2)
        if (local_iter + 1) % self.server_update_interval == 0:
            smashed_data = splitting_output.clone().detach().requires_grad_(True)
            self.charge_cut_activation(smashed_data)
            self.charge_cut_labels(y)     # this loss IS computed server-side

            with self.phase('server', i):
                self.server.optimizer.zero_grad()
                out = self.server.model(smashed_data)
                s_loss = self.criterion(out, y)

                with torch.no_grad():
                    global_loss = s_loss.item()
                    _, predicted = torch.max(out.data, 1)
                    global_correct = predicted.eq(y.view_as(predicted)).sum().item()
                    ret_dict['g_loss'] = global_loss
                    ret_dict['g_acc'] = global_correct / y.size(dim=0)

                s_loss.backward()
                self.server.optimizer.step()

        return ret_dict

    def aggregate(self):
        ret_dict = self.aggregate_clients()

        t0 = time.time()
        self.aggregated_auxiliary = aggregate_models(
            [c.auxiliary_model for c in self.clients],
            self.agg_factor, self.device
        )
        agg_weights = self.aggregated_auxiliary.state_dict()

        for c in self.clients:
            c.auxiliary_model.load_state_dict(agg_weights)
            self.charge_weights_roundtrip(c.auxiliary_model, 'aux')

        ret_dict['auxiliary_agg_compute_time'] = time.time() - t0
        return ret_dict

# ------------------------------------------------------------------------------