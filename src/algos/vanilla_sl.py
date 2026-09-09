# ------------------------------------------------------------------------------
# Vanilla Split Learning -- the original, non-federated split-learning protocol
# (Gupta & Raskar 2018; Vepakomma et al. 2018), included as the base/reference
# point CLAUDE.md asks for alongside SplitFedv1/v2. Unlike those two (which
# FedAvg per-client model copies every round), vanilla SL has no client-side
# aggregation at all: there is exactly ONE shared client-side model and ONE
# shared server-side model, and each "client" in this harness is simply a data
# shard taking its turn training that same pair of models with real,
# end-to-end backprop across the cut -- classic sequential/relay-style SL,
# with no weight averaging step needed since there's only one copy to begin
# with. Mirrors ho_sfl.py's `shared_model` pattern for the client side.
#
# comm_load_weights stays 0 for the whole run (no aggregation event ever
# happens), matching this harness's convention for non-federated methods.
# ------------------------------------------------------------------------------
import torch

from algos import register_algorithm, FLAlgorithm
from models import config_optimizer, config_lr_scheduler

# ------------------------------------------------------------------------------
@register_algorithm("vanilla_sl")
class VanillaSL(FLAlgorithm):

    def __init__(self, *args, **kwargs):
        super(VanillaSL, self).__init__(*args, **kwargs)

        # single shared client model across all clients -- sequential
        # training naturally carries each client's just-updated weights
        # forward to the next, with no averaging step.
        shared_model = self.clients[0].model
        for c in self.clients[1:]:
            c.model = shared_model

        shared_optimizer = config_optimizer(
            shared_model.parameters(), self.clients[0].optimizer_options
        )
        shared_lr_scheduler = config_lr_scheduler(
            shared_optimizer, self.clients[0].lr_scheduler_options
        )
        for c in self.clients:
            c.optimizer = shared_optimizer
            c.lr_scheduler = shared_lr_scheduler

        self.aggregated_client = shared_model

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            splitting_output = self.clients[i].model(x)

        smashed_data = splitting_output.clone().detach().requires_grad_(True)
        self.charge_cut_activation(smashed_data)
        self.charge_cut_labels(y)     # loss is computed server-side

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
            'acc': train_correct / y.size(dim=0),
            'loss': train_loss,
        }

    def aggregate(self):
        # non-federated: nothing to average -- there is only one shared
        # client model and one shared server model already.
        return {}

# ------------------------------------------------------------------------------
