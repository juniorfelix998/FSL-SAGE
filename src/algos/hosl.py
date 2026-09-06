# ------------------------------------------------------------------------------
# HOSL: Hybrid-Order Split Learning for Memory-Constrained Edge Training
# (arXiv:2601.10940). AI-ASSISTED NO-CODE REIMPLEMENTATION per CLAUDE.md's
# rule for methods without a working released implementation; not validated
# against the paper's own reported CIFAR-10/CIFAR-100/ImageNet numbers.
#
# NOTE: this is a DIFFERENT paper from `ho_sfl.py` in this same directory
# ("HO-SFL: Hybrid-Order Split Federated Learning with Backprop-Free Clients
# and Dimension-Free Aggregation", arXiv:2603.14773) -- confirmed distinct
# arXiv ids and distinct scope: HO-SFL is federated (dimension-free
# aggregation across a client population sharing one model, real gradient
# projection); HOSL is explicitly single-client/non-federated and targets
# per-device memory savings, not aggregation efficiency.
#
# Read from the paper: "hybrid-order" means an asymmetric split --
#   - Client-side: zeroth-order (gradient-free), via single-sided
#     finite-difference: grad_hat(f)(x) = (f(x+delta*u) - f(x))/delta * u,
#     u a random direction, averaged over m perturbations to reduce
#     variance. No backprop, no activation caching needed on the client --
#     this is the paper's stated memory-saving mechanism for edge devices.
#   - Server-side: ordinary first-order backpropagation.
# Single client-server pair (non-federated) -- no gradient is ever sent back
# to the client (it never needed one), only a per-perturbation scalar loss
# difference. `num_clients=1` is this method's paper-faithful configuration,
# per its own single-client framing; running with more clients in this
# harness just means each trains an independent, unaggregated copy (same
# pattern as this harness's other non-federated methods, e.g. dsl_aux.py).
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
@register_algorithm("hosl")
class HOSL(FLAlgorithm):

    def __init__(self, *args, **kwargs):
        super(HOSL, self).__init__(*args, **kwargs)
        self.num_pert = self.cfg.num_pert
        self.mu = self.cfg.zo_mu
        self.aggregated_client = self.clients[0].model

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        self.clients[i].optimizer.zero_grad()
        self.server.optimizer.zero_grad()

        with torch.no_grad():
            h_fixed = self.clients[i].model(x)
        self.comm_load_cut += h_fixed.numel() * h_fixed.element_size()

        # real server-side update on the real (unperturbed) smashed data --
        # ordinary backprop, server-only; no gradient returned to the client.
        smashed_data = h_fixed.clone().detach().requires_grad_(True)
        out = self.server.model(smashed_data)
        loss = self.criterion(out, y)
        loss.backward()
        self.server.optimizer.step()

        with torch.no_grad():
            baseline_loss = loss.item()
            _, predicted = torch.max(out.data, 1)
            train_correct = predicted.eq(y.view_as(predicted)).sum().item()

        # client-side zeroth-order update: m single-sided perturbations,
        # each evaluated against the (already-updated) server -- every
        # evaluation this round shares the same, now-fixed server state.
        for _ in range(self.num_pert):
            seed = int(np.random.randint(0, 1_000_000))

            _perturb(self.clients[i].model, seed, self.mu)
            with torch.no_grad():
                h_pert = self.clients[i].model(x)
                out_pert = self.server.model(h_pert)
                loss_pert = self.criterion(out_pert, y)
            _perturb(self.clients[i].model, seed, -self.mu)  # restore

            self.comm_load_cut += h_pert.numel() * h_pert.element_size()
            # one scalar loss difference sent back per perturbation
            self.comm_load_cut += torch.zeros(1).element_size()

            scalar = (loss_pert.item() - baseline_loss) / self.mu
            _perturb_accumulate_grad(
                self.clients[i].model, seed, scalar / self.num_pert
            )

        self.clients[i].optimizer.step()

        return {
            'acc': train_correct / y.size(dim=0),
            'loss': baseline_loss,
        }

    def aggregate(self):
        # non-federated: no cross-client averaging (paper's own single
        # client-server framing) -- comm_load_weights stays 0.
        t0 = time.time()
        self.aggregated_client = self.clients[0].model
        return {'client_agg_compute_time': time.time() - t0}

# ------------------------------------------------------------------------------
