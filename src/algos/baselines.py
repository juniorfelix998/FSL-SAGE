# ------------------------------------------------------------------------------
import time
import copy
import torch.nn as nn
import torch
from torch import optim

from algos import register_algorithm, aggregate_models, FLAlgorithm
from models import config_optimizer, config_lr_scheduler

# ------------------------------------------------------------------------------
@register_algorithm("fed_avg")
class FedAvg(FLAlgorithm):

    def __init__(self,
        *args, **kwargs 
    ):
        super(FedAvg, self).__init__(*args, **kwargs)

        # merge client and server models into one
        for c in self.clients:
            c.model = nn.Sequential(c.model, self.server.model)
            c.optimizer = config_optimizer(
                c.model.parameters(), c.optimizer_options
            )
            c.lr_scheduler = config_lr_scheduler(
                c.optimizer, c.lr_scheduler_options
            )

    def full_model(self, x):
        return self.aggregated_client(x)

    # nothing crosses a cut and there is no separate server host: the server
    # model was merged into every client's model in __init__ above.
    def server_side_modules(self):
        return []

    def server_side_optimizers(self):
        return []

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            out = self.clients[i].model(x)
            loss = self.criterion(out, y)

            with torch.no_grad():
                train_loss = loss.item()
                _, predicted = torch.max(out.data, 1)
                train_correct = predicted.eq(y.view_as(predicted)).sum().item()

            loss.backward()
            self.clients[i].optimizer.step()

        return {
            'acc' : train_correct / y.size(dim=0),
            'loss': train_loss,
        }
    
# ------------------------------------------------------------------------------
@register_algorithm("sl_single_server")
class SplitFedv2(FLAlgorithm):

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            # pass smashed data through full model
            splitting_output = self.clients[i].model(x)

        # Represents the uploaded data
        smashed_data = splitting_output.clone().detach().requires_grad_(True)
        self.charge_cut_activation(smashed_data)
        self.charge_cut_labels(y)     # loss is computed server-side

        # NOTE the client's graph (`splitting_output`) stays alive throughout the
        # server phase below -- that stall is what `client_mem_held_across_cut_mb`
        # measures, and it is the memory cost of sending a gradient back.
        with self.phase('server', i):
            self.server.optimizer.zero_grad()
            output = self.server.model(smashed_data)
            loss = self.server.criterion(output, y)

            with torch.no_grad():
                train_loss = loss.item()
                _, predicted = torch.max(output.data, 1)
                train_correct = predicted.eq(y.view_as(predicted)).sum().item()

            loss.backward()
            self.server.optimizer.step()

        self.charge_cut_gradient(smashed_data.grad)
        self.hold('client', smashed_data.grad, i=i)

        with self.phase('client', i):
            splitting_output.backward(smashed_data.grad)
            self.clients[i].optimizer.step()

        return {
            'acc' : train_correct / y.size(dim=0),
            'loss': train_loss,
        }

# ------------------------------------------------------------------------------
@register_algorithm("sl_multi_server")
class SplitFedv1(FLAlgorithm):
    aggregated_server : nn.Module

    def __init__(self, *args, **kwargs):
        super(SplitFedv1, self).__init__(*args, **kwargs)

        # split server model
        self.servers = []
        for c in self.clients:
            self.servers.append(copy.deepcopy(self.server))

    def full_model(self, x):
        return self.aggregated_server(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        if t > 0: self.aggregated_server.train()

    def special_models_eval_mode(self):
        self.aggregated_server.eval()

    # the server host really does hold one replica per client
    def server_side_modules(self):
        return [s.model for s in self.servers]

    def server_side_optimizers(self):
        return [s.optimizer for s in self.servers]

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            # pass smashed data through full model
            splitting_output = self.clients[i].model(x)

        # Represents the uploaded data
        smashed_data = splitting_output.clone().detach().requires_grad_(True)
        self.charge_cut_activation(smashed_data)
        self.charge_cut_labels(y)     # loss is computed server-side

        with self.phase('server', i):
            self.servers[i].optimizer.zero_grad()
            output = self.servers[i].model(smashed_data)
            loss = self.criterion(output, y)

            with torch.no_grad():
                train_loss = loss.item()
                _, predicted = torch.max(output.data, 1)
                train_correct = predicted.eq(y.view_as(predicted)).sum().item()

            loss.backward()
            self.servers[i].optimizer.step()

        self.charge_cut_gradient(smashed_data.grad)
        self.hold('client', smashed_data.grad, i=i)

        with self.phase('client', i):
            splitting_output.backward(smashed_data.grad)
            self.clients[i].optimizer.step()

        return {
            'acc' : train_correct / y.size(dim=0),
            'loss': train_loss,
        }

    def aggregate(self):
        ret_dict = self.aggregate_clients()

        t0 = time.time()
        self.aggregated_server = aggregate_models(
            [s.model for s in self.servers], self.agg_factor, self.device
        )
        agg_weights = self.aggregated_server.state_dict()

        for s in self.servers:
            s.model.load_state_dict(agg_weights)
            # BUGFIX: this exchange used to be charged ZERO bytes, while every
            # other multi-server method here (fedsplitx, han_locloss,
            # mu_splitfed) charges a full round trip for the identical
            # operation. That biased SplitFedv1 low on Comm-TOTAL, the value the
            # benchmark ranks on.
            self.charge_weights_roundtrip(self.aggregated_server, 'server')

        ret_dict['server_agg_compute_time'] = time.time() - t0
        return ret_dict
        
# ------------------------------------------------------------------------------