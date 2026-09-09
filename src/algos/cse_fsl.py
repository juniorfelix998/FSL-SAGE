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
        if local_iter % self.server_update_interval == 0:
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