# ------------------------------------------------------------------------------
# Ported from HKU-WILL-Lab/HO-SFL's HO_SFL_CV/src/runner/ho_sfl_runner.py — this
# IS the paper's own proposed method (arXiv:2603.14773, "HO-SFL: Hybrid-Order
# Split Federated Learning with Backprop-Free Clients and Dimension-Free
# Aggregation"), not a third-party reimplementation.
#
# Two mechanics carried over faithfully from the reference, both load-bearing:
#   - The client model is a SINGLE shared instance across all clients (no
#     per-client weight divergence, hence nothing to aggregate as a weight
#     vector) — this *is* the paper's "dimension-free aggregation": only P
#     scalars + P seeds are combined across clients each round, not a
#     model-weight vector.
#   - The server does real backprop (first-order); the client only ever does
#     forward passes under random parameter perturbations (zeroth-order),
#     reusing the server's true activation-gradient as a fixed direction to
#     project each perturbed activation onto. This is the "hybrid-order" part.
#
# Known, documented simplifications vs. the reference (see plan discussion):
#   - Full client participation every round (no sub-round sampling/staleness
#     bookkeeping) — this harness has no sub-round client-sampling mechanism;
#     the paper's algorithm reduces exactly to this when sampled == total.
#   - One real batch per client per round: `client_step` does real work only
#     at the first (epoch, iter) of the round; every other call that round is
#     a no-op. A freshly-shuffled DataLoader's first batch is a fresh random
#     draw each round, closely analogous to the reference's own per-round
#     `next(loader)`.
# ------------------------------------------------------------------------------
import time
import numpy as np
import torch

from algos import register_algorithm, FLAlgorithm

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
@register_algorithm("ho_sfl")
class HO_SFL(FLAlgorithm):

    def __init__(self, *args, **kwargs):
        super(HO_SFL, self).__init__(*args, **kwargs)

        # single shared client model across all clients -- no per-client
        # weight divergence, nothing to aggregate as a weight vector.
        shared_model = self.clients[0].model
        for c in self.clients[1:]:
            c.model = shared_model

        shared_optimizer = self._get_optimizer(shared_model.parameters())
        for c in self.clients:
            c.optimizer = shared_optimizer
            c.lr_scheduler = None

        self.server.optimizer = self._get_optimizer(self.server.model.parameters())

        self.P = self.cfg.zo_p
        self.mu = self.cfg.zo_mu
        self._round_buf = []

    def _get_optimizer(self, params):
        if self.cfg.optimizer == 'adamw':
            return torch.optim.AdamW(
                params, lr=self.cfg.lr, betas=tuple(self.cfg.betas),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer == 'sgd':
            return torch.optim.SGD(
                params, lr=self.cfg.lr, momentum=self.cfg.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.cfg.optimizer}")

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        self._round_buf = []
        self.server.optimizer.zero_grad()
        self.clients[0].optimizer.zero_grad()
        # critical: keep the shared client model in eval() so the repeated
        # zeroth-order probe forward passes in aggregate() don't corrupt
        # BatchNorm running stats. This overrides the base class's default
        # train_mode(), which already set every client model to .train() --
        # this hook runs after that default loop, so it wins.
        self.clients[0].model.eval()

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        if (j, k) != (0, 0):
            return {'acc': 0.0, 'loss': 0.0}

        t0 = time.time()
        with torch.no_grad():
            a_m = self.clients[i].model(x)
        t1 = time.time()

        server_input = a_m.clone().detach().requires_grad_(True)
        out = self.server.model(server_input)
        loss = self.criterion(out, y)

        with torch.no_grad():
            train_loss = loss.item()
            _, predicted = torch.max(out.data, 1)
            train_correct = predicted.eq(y.view_as(predicted)).sum().item()

        t2 = time.time()
        scale = loss.new_tensor(1.0 / len(self.clients))
        loss.backward(scale)
        t_s = time.time() - t2

        g_a_m = server_input.grad.clone().detach()
        self.comm_load_cut += a_m.numel() * a_m.element_size()
        self.comm_load_cut += g_a_m.numel() * g_a_m.element_size()

        self._round_buf.append((x.detach(), a_m.detach(), g_a_m))

        return {
            'acc': train_correct / y.size(dim=0),
            'loss': train_loss,
            'client_model_compute_time': t1 - t0,
            'server_model_compute_time': t_s,
        }

    def aggregate(self):
        ret_dict = {}
        n = len(self.clients)

        t0 = time.time()
        for p in self.server.model.parameters():
            if p.grad is not None:
                p.grad.div_(n)
        self.server.optimizer.step()
        ret_dict['server_optimizer_step_time'] = time.time() - t0

        # nothing buffered this round (shouldn't normally happen) -- still
        # need to (re)establish aggregated_client for eval_mode()/full_model
        if len(self._round_buf) == 0:
            self.aggregated_client = self.clients[0].model
            return ret_dict

        t0 = time.time()
        seeds = [int(np.random.randint(0, 1_000_000)) for _ in range(self.P)]
        seeds_tensor = torch.tensor(seeds, dtype=torch.int32)

        all_v = []
        for (x_buf, a_m, g_a_m) in self._round_buf:
            v = torch.zeros(self.P, device=self.device)
            for p_idx, seed in enumerate(seeds):
                _perturb(self.clients[0].model, seed, self.mu)
                with torch.no_grad():
                    a_tilde = self.clients[0].model(x_buf)
                diff = a_tilde - a_m
                v[p_idx] = torch.sum(diff * g_a_m)
                _perturb(self.clients[0].model, seed, -self.mu)
            all_v.append(v)

        bar_v = torch.stack(all_v).mean(dim=0)
        scale = 1.0 / (self.P * self.mu)
        for p_idx, seed in enumerate(seeds):
            _perturb_accumulate_grad(
                self.clients[0].model, seed, bar_v[p_idx].item() * scale
            )
        self.clients[0].optimizer.step()
        ret_dict['zo_probe_compute_time'] = time.time() - t0

        # dimension-free aggregation: only P scalars (per client, uplink) and
        # P scalars + P seeds (broadcast back to every client, downlink) are
        # exchanged -- never a full client-model weight vector. This is the
        # number that should come out far smaller than every other method's
        # weight-transfer cost.
        n_buffered = len(self._round_buf)
        self.comm_load_weights += v.numel() * v.element_size() * n_buffered
        self.comm_load_weights += bar_v.numel() * bar_v.element_size() * n_buffered
        self.comm_load_weights += \
            seeds_tensor.numel() * seeds_tensor.element_size() * n_buffered

        self.aggregated_client = self.clients[0].model
        return ret_dict

# ------------------------------------------------------------------------------
