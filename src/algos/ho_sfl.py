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
from utils.comm import tensor_bytes
from models.pretrained import (
    pretrained_resnet18_state_dict, load_pretrained_client,
    load_pretrained_server, freeze_batchnorm_affine
)

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

        # ImageNet-pretrained init + frozen BN affine, matching the
        # reference's conf/base.yaml (`model.use_pretrained: True`,
        # `model.freeze_bn: True`), applied BEFORE the model is shared out
        # below so every client sees the same weights.
        #
        # This is load-bearing, not cosmetic: the client here is updated ONLY
        # by a P-direction zeroth-order estimator, whose convergence scales
        # with d/P. At ~683k client params and P=5 it is untrainable from
        # scratch in a few hundred rounds -- the reference works because the
        # client starts near-optimal and the first-order server does the
        # learning. Compounding that, the client is pinned to .eval() for the
        # whole run (see special_models_train_mode below, faithful to the
        # reference), so from a random init its BatchNorm running stats stay at
        # mean 0 / var 1 forever and the stem never normalises at all. Omitting
        # this was the dominant cause of this method's near-chance accuracy.
        self.use_pretrained = self.cfg.get('use_pretrained', True)
        self.freeze_bn = self.cfg.get('freeze_bn', True)
        if self.use_pretrained:
            full_state = pretrained_resnet18_state_dict()
            load_pretrained_client(self.clients[0].model, full_state)
            load_pretrained_server(self.server.model, full_state)
        if self.freeze_bn:
            freeze_batchnorm_affine(self.clients[0].model)
            freeze_batchnorm_affine(self.server.model)

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

        # The single-shared-client-model premise above IS the paper's
        # dimension-free aggregation, and aggregate() below relies on it by
        # only ever stepping clients[0]. Assert it so a future change that
        # gives a client its own model/optimizer fails loudly instead of
        # silently training one client and reporting it as all of them.
        assert all(c.model is self.clients[0].model for c in self.clients), \
            "HO-SFL requires one shared client model across all clients"
        assert all(c.optimizer is self.clients[0].optimizer for c in self.clients), \
            "HO-SFL requires one shared client optimizer across all clients"

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

        # client forward under no_grad -- a zeroth-order client retains no
        # autograd activations, so its activation peak is ~0 by construction
        with self.phase('client', i):
            with torch.no_grad():
                a_m = self.clients[i].model(x)

        server_input = a_m.clone().detach().requires_grad_(True)
        with self.phase('server', i):
            out = self.server.model(server_input)
            loss = self.criterion(out, y)

            with torch.no_grad():
                train_loss = loss.item()
                _, predicted = torch.max(out.data, 1)
                train_correct = predicted.eq(y.view_as(predicted)).sum().item()

            scale = loss.new_tensor(1.0 / len(self.clients))
            loss.backward(scale)

        g_a_m = server_input.grad.clone().detach()
        self.charge_cut_activation(a_m)
        self.charge_cut_labels(y)     # loss is computed server-side
        self.charge_cut_gradient(g_a_m)

        # This buffer is CLIENT-resident: the zeroth-order probe in aggregate()
        # replays the client forward locally against the downloaded g_a_m, so
        # the client must keep x, a_m and g_a_m alive until then.
        self._round_buf.append((x.detach(), a_m.detach(), g_a_m))
        self.hold('client', x, a_m, g_a_m, i=i)

        return {
            'acc': train_correct / y.size(dim=0),
            'loss': train_loss,
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

        # The probe loop below is CLIENT-side compute even though it lives in
        # aggregate(): it replays the client's own forward against the already
        # downloaded g_a_m. It must be bracketed as such, or HO-SFL's client
        # memory/time would read as "static only" -- a fake result that would
        # flatter the method.
        all_v = []
        with self.phase('client', 0):
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
        self.charge_weights_scalars(
            (tensor_bytes(v) + tensor_bytes(bar_v) + tensor_bytes(seeds_tensor))
            * n_buffered
        )

        self.aggregated_client = self.clients[0].model
        return ret_dict

# ------------------------------------------------------------------------------
