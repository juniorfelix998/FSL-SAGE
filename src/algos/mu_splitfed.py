# ------------------------------------------------------------------------------
# Ported from HKU-WILL-Lab/HO-SFL's HO_SFL_CV/src/runner/mu_splitfed_runner.py.
#
# IMPORTANT PROVENANCE NOTE: this is a THIRD-PARTY CV/ResNet18 reimplementation
# of MU-SplitFed written by the HO-SFL authors as their own baseline comparison
# -- it is NOT the original Johnny-Zip/MU-SplitFed authors' code. The original
# repo is LLM/OPT-only (no image-classification path at all) and is
# non-functional as published (imports a `cezo_fl` package that was never
# committed to that repository), so there is no way to run "their own code" on
# MNIST/CIFAR. This must be labeled as a reimplementation everywhere this
# method appears (table footnotes, plots, etc.), matching the paper's own
# convention for its "reimplemented, AI-assisted" rows.
#
# Both client and server are backprop-free (pure zeroth-order, two-point
# finite-difference gradient estimation), unlike HO-SFL which keeps real
# server backprop. Full per-client model weights (both client and server) ARE
# aggregated every round via a FedAvg-style delta-averaging with a separate
# global step size `lr_g` vs. local `lr_c`/`lr_s` -- the "unbalanced update"
# (MU) contribution -- so weight-transfer communication here is comparable in
# kind to the harness's other FedAvg-style aggregators, unlike HO-SFL's
# dimension-free scheme.
#
# Known, documented simplification vs. the reference (see plan discussion):
# one real batch per client per round -- `client_step` does real work only at
# the first (epoch, iter) of the round; every other call that round is a
# no-op, analogous to the reference's own per-round `next(iter(loader))`.
# ------------------------------------------------------------------------------
import time
import copy
import numpy as np
import torch

from algos import register_algorithm, FLAlgorithm
from utils.utils import calculate_load

# ------------------------------------------------------------------------------
def _perturb(model, seed, scale_factor):
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            u = torch.randn_like(param)
            param.add_(u, alpha=scale_factor)
    torch.set_rng_state(rng_state)

# ------------------------------------------------------------------------------
def _perturb_accumulate_grad(model, seed, scalar_weight):
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            u = torch.randn_like(param)
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad.add_(u, alpha=scalar_weight)
    torch.set_rng_state(rng_state)

# ------------------------------------------------------------------------------
@register_algorithm("mu_splitfed")
class MU_SplitFed(FLAlgorithm):
    aggregated_server: torch.nn.Module

    def __init__(self, *args, **kwargs):
        super(MU_SplitFed, self).__init__(*args, **kwargs)

        self.tau = self.cfg.tau
        self.mu = self.cfg.zo_mu
        self.eta_g = self.cfg.lr_g
        self.P = self.cfg.zo_p

        self.servers = [copy.deepcopy(self.server) for _ in self.clients]

        for c in self.clients:
            c.optimizer = torch.optim.SGD(c.model.parameters(), lr=self.cfg.lr_c)
            c.lr_scheduler = None
        for s in self.servers:
            s.optimizer = torch.optim.SGD(s.model.parameters(), lr=self.cfg.lr_s)
            s.lr_scheduler = None

        self.aggregated_client = copy.deepcopy(self.clients[0].model)
        self.aggregated_server = copy.deepcopy(self.server.model)

    def full_model(self, x):
        return self.aggregated_server(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        if t > 0:
            self.aggregated_server.train()

    def special_models_eval_mode(self):
        self.aggregated_server.eval()

    def _aggregate_weights(self, global_state, state_dicts):
        num_clients = len(state_dicts)
        new_state = {}
        for key in global_state.keys():
            if not torch.is_floating_point(global_state[key]):
                new_state[key] = global_state[key]
                continue

            delta_avg = torch.zeros_like(
                global_state[key], dtype=torch.float32, device=self.device
            )
            for state in state_dicts:
                delta_avg += (
                    state[key].to(self.device) - global_state[key].to(self.device)
                ) / num_clients

            new_param = global_state[key].to(self.device) + self.eta_g * delta_avg
            new_state[key] = new_param.to(global_state[key].dtype)
        return new_state

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        if (j, k) != (0, 0):
            return {'acc': 0.0, 'loss': 0.0}

        self.clients[i].optimizer.zero_grad()
        self.servers[i].optimizer.zero_grad()

        with torch.no_grad():
            h_fixed = self.clients[i].model(x)
        self.comm_load_cut += h_fixed.numel() * h_fixed.element_size()

        # tau local server-only zeroth-order steps, reusing h_fixed -- no
        # further client communication needed for these. Each step's
        # gradient estimate now averages P independent random directions
        # (instead of a single one) to cut finite-difference variance,
        # mirroring ho_sfl.py's zo_p multi-direction averaging.
        t0_s = time.time()
        for _ in range(self.tau):
            self.servers[i].optimizer.zero_grad()

            for _ in range(self.P):
                s_seed = int(np.random.randint(0, 1_000_000))

                _perturb(self.servers[i].model, s_seed, self.mu)
                with torch.no_grad():
                    out_p = self.servers[i].model(h_fixed)
                    loss_p = self.criterion(out_p, y)

                _perturb(self.servers[i].model, s_seed, -2 * self.mu)
                with torch.no_grad():
                    out_n = self.servers[i].model(h_fixed)
                    loss_n = self.criterion(out_n, y)

                _perturb(self.servers[i].model, s_seed, self.mu)  # restore

                scalar_s = (loss_p.item() - loss_n.item()) / (2 * self.mu)
                _perturb_accumulate_grad(
                    self.servers[i].model, s_seed, scalar_s / self.P
                )
            self.servers[i].optimizer.step()
        t_s = time.time() - t0_s

        # client-side zeroth-order step: two more client forwards per
        # direction, each a fresh activation upload. Averaged over P
        # independent random directions, same variance-reduction rationale
        # as the server loop above.
        t0_c = time.time()
        for _ in range(self.P):
            c_seed = int(np.random.randint(0, 1_000_000))

            _perturb(self.clients[i].model, c_seed, self.mu)
            with torch.no_grad():
                h_pos = self.clients[i].model(x)
            self.comm_load_cut += h_pos.numel() * h_pos.element_size()

            _perturb(self.clients[i].model, c_seed, -2 * self.mu)
            with torch.no_grad():
                h_neg = self.clients[i].model(x)
            self.comm_load_cut += h_neg.numel() * h_neg.element_size()

            _perturb(self.clients[i].model, c_seed, self.mu)  # restore

            with torch.no_grad():
                out_c_p = self.servers[i].model(h_pos)
                loss_c_p = self.criterion(out_c_p, y)
                out_c_n = self.servers[i].model(h_neg)
                loss_c_n = self.criterion(out_c_n, y)

            scalar_c = (loss_c_p.item() - loss_c_n.item()) / (2 * self.mu)
            # one scalar sent back per direction so the client can replay its
            # own update -- drives this client's own update this round, same
            # role as HO-SFL's g_a_m, so it's classified as cut traffic, not
            # an aggregation event.
            self.comm_load_cut += torch.zeros(1).element_size()

            _perturb_accumulate_grad(
                self.clients[i].model, c_seed, scalar_c / self.P
            )
        self.clients[i].optimizer.step()
        t_c = time.time() - t0_c

        with torch.no_grad():
            _, predicted = torch.max(out_c_p.data, 1)
            train_correct = predicted.eq(y.view_as(predicted)).sum().item()

        return {
            'acc': train_correct / y.size(dim=0),
            'loss': (loss_c_p.item() + loss_c_n.item()) / 2.0,
            'client_model_compute_time': t_c,
            'server_model_compute_time': t_s,
        }

    def aggregate(self):
        t0 = time.time()

        agg_client = self._aggregate_weights(
            self.aggregated_client.state_dict(),
            [c.model.state_dict() for c in self.clients]
        )
        agg_server = self._aggregate_weights(
            self.aggregated_server.state_dict(),
            [s.model.state_dict() for s in self.servers]
        )
        self.aggregated_client.load_state_dict(agg_client)
        self.aggregated_server.load_state_dict(agg_server)

        # write the freshly-aggregated global weights back into every
        # client/server copy now, so round t+1 already starts synced --
        # avoids a double-count of download bytes a literal port (which
        # reloads at the top of each client's turn instead) would introduce.
        for c in self.clients:
            c.model.load_state_dict(agg_client)
            self.comm_load_weights += 2 * calculate_load(self.aggregated_client)
        for s in self.servers:
            s.model.load_state_dict(agg_server)
            self.comm_load_weights += 2 * calculate_load(self.aggregated_server)

        return {'client_agg_compute_time': time.time() - t0}

# ------------------------------------------------------------------------------
