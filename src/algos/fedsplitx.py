# ------------------------------------------------------------------------------
# FedSplitX: Federated Split Learning for Computationally-Constrained
# Heterogeneous Clients (Shin, Ahn, Kang, Kang -- arXiv:2310.14579).
# AI-ASSISTED NO-CODE REIMPLEMENTATION per CLAUDE.md's rule for methods
# without a working released implementation; not validated against the
# paper's own reported CIFAR-10/CIFAR-100/ResNet numbers.
#
# Read directly from the paper (Algorithm 1, Sec. 2.1-2.3): FedSplitX's
# defining contribution is a client population split into M "depth-levels"
# d_k in {1,...,M}, each with its OWN partition point, its OWN client/server
# sub-model sizes, and M auxiliary networks total -- a client at depth-level
# m holds m of them (client-side), the server holds the remaining M-m
# (server-side). Aggregation ("heteroavg") only averages a given layer's
# parameters across the subset of clients whose depth-level actually reaches
# that layer.
#
# This harness's benchmark protocol runs every method at ONE shared cut per
# run (CLAUDE.md's "same dataset, backbone, cut, seed" fairness rule, and the
# `cut=shallow/middle/deep` dimension added separately) -- so every client
# here has the SAME depth-level (M=1 in the paper's own notation). This is a
# genuine, disclosed simplification: FedSplitX's own defining
# multi-depth-level mechanism is not exercised. What IS faithfully ported at
# M=1 is the paper's per-client training/aggregation mechanics, which at a
# single depth-level read directly as:
#   - Client-side (Algorithm 1, Client_Update): each client trains its
#     client-side model purely from its own auxiliary/"collaborative" loss
#     (real backprop) -- never waiting for a signal from the server.
#   - Server-side: the server processes the real (non-auxiliary) smashed
#     data through its remaining layers and trains from its OWN local loss
#     (real backprop) -- no gradient is ever sent back to the client.
#   - Aggregation ("heteroavg", Sec. 2.3): with every client at the same
#     depth-level, the subset-averaging formula degenerates to plain FedAvg
#     over both the client-side and server-side models -- there is no
#     parameter subset to exclude.
#
# This makes the implementation below structurally close to han_locloss.py
# (also a real-backprop, no-cut-crossing-gradient, auxiliary-local-loss
# method) -- that is an honest consequence of both papers' mechanisms
# degenerating similarly at a single shared cut, not a coding shortcut.
# Reuses this harness's existing ResNetAuxiliary for the client-side
# auxiliary head, as CSE-FSL/FSL-SAGE/DSL-Aux/han_locloss already do.
# ------------------------------------------------------------------------------
import time
import copy
import torch

from algos import register_algorithm, aggregate_models, FLAlgorithm
from utils.utils import calculate_load

# ------------------------------------------------------------------------------
@register_algorithm("fedsplitx")
class FedSplitX(FLAlgorithm):
    aggregated_auxiliary: torch.nn.Module
    aggregated_server: torch.nn.Module

    def __init__(self, *args, **kwargs):
        super(FedSplitX, self).__init__(*args, **kwargs)
        self.servers = [copy.deepcopy(self.server) for _ in self.clients]

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

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        self.clients[i].optimizer.zero_grad()
        self.clients[i].auxiliary_model.optimizer.zero_grad()
        self.servers[i].optimizer.zero_grad()

        # Client_Update (Algorithm 1, lines 13-16): local collaborative loss
        # via the client's own auxiliary head -- at M=1 this is the client's
        # only auxiliary network, so the "sum over m auxiliary logits" in
        # Sec. 2.2's F^c_k reduces to this single term.
        splitting_output = self.clients[i].model(x)
        aux_out = self.clients[i].auxiliary_model.forward_inner(splitting_output)
        client_loss = self.criterion(aux_out, y)
        client_loss.backward()
        self.clients[i].optimizer.step()
        self.clients[i].auxiliary_model.optimizer.step()

        with torch.no_grad():
            _, predicted = torch.max(aux_out.data, 1)
            client_correct = predicted.eq(y.view_as(predicted)).sum().item()

        # server-side: real smashed data through the server's remaining
        # layers, its own local loss (Sec. 2.2's F^s_k with M-m=0 remaining
        # auxiliary terms at M=1, leaving only the final output's loss),
        # real backprop -- no gradient returned to the client.
        smashed_data = splitting_output.detach()
        self.comm_load_cut += smashed_data.numel() * smashed_data.element_size()

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
        # heteroavg (Sec. 2.3) degenerates to plain FedAvg since every
        # client shares the same depth-level in this harness's single-cut
        # protocol -- see header note.
        ret_dict = self.aggregate_clients()

        t0 = time.time()
        self.aggregated_auxiliary = aggregate_models(
            [c.auxiliary_model for c in self.clients], self.agg_factor, self.device
        )
        agg_aux_weights = self.aggregated_auxiliary.state_dict()
        for c in self.clients:
            c.auxiliary_model.load_state_dict(agg_aux_weights)
            self.comm_load_weights += 2 * calculate_load(self.aggregated_auxiliary)
        ret_dict['auxiliary_agg_compute_time'] = time.time() - t0

        t0 = time.time()
        self.aggregated_server = aggregate_models(
            [s.model for s in self.servers], self.agg_factor, self.device
        )
        agg_server_weights = self.aggregated_server.state_dict()
        for s in self.servers:
            s.model.load_state_dict(agg_server_weights)
            self.comm_load_weights += 2 * calculate_load(self.aggregated_server)
        ret_dict['server_agg_compute_time'] = time.time() - t0

        return ret_dict

# ------------------------------------------------------------------------------
