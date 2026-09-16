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

from omegaconf import OmegaConf

from algos import register_algorithm, FLAlgorithm
from models import config_optimizer

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
def _zo_update(model, seed, step):
    """Algorithm 3's ZOUPDATE: theta <- theta - step * z, with z REGENERATED
    from its seed rather than stored.

    This is what makes a zeroth-order client cheap in memory. The paper is
    explicit that ZO needs no gradient storage at all (Eq. 15, M_grad = 0) and
    no stored activations (Eq. 18, M_act = 0); an earlier version of this file
    accumulated a real `.grad` per parameter and stepped a torch optimizer,
    which allocated exactly the buffers the method exists to avoid and inflated
    the one metric it is measured on.
    """
    rng_state = torch.get_rng_state()
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in model.parameters():
            if not param.requires_grad:
                continue
            u = torch.randn_like(param)
            param.add_(u, alpha=-step)
    torch.set_rng_state(rng_state)

# ------------------------------------------------------------------------------
@register_algorithm("hosl")
class HOSL(FLAlgorithm):

    def __init__(self, *args, **kwargs):
        super(HOSL, self).__init__(*args, **kwargs)
        self.Q = int(self.cfg.num_pert)
        self.mu = float(self.cfg.zo_mu)
        self.client_lr = float(self.cfg.client_lr)
        self.aggregated_client = self.clients[0].model

        # Appendix VII-A: "Both client and server use SGD without momentum."
        # The client no longer holds an optimizer at all (its update is applied
        # in place by _zo_update), so only the server's is rebuilt here.
        self.server.optimizer = config_optimizer(
            self.server.model.parameters(),
            OmegaConf.create({'name': 'sgd',
                              'options': {'lr': self.cfg.server_lr}})
        )

    # the ZO client keeps no gradient and no optimizer state -- Eq. 15/18
    def client_side_optimizers(self, i):
        return []

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        # --- Alg. 1 Phase 1 (L6-25): ZO gradient estimation, Q perturbations.
        # The server stays in INFERENCE mode throughout and its parameters are
        # unchanged (Alg. 2 L11), so all 2Q probes are evaluated against one
        # fixed server state -- that is what makes the estimate unbiased.
        grads = []
        for _ in range(self.Q):
            seed = int(np.random.randint(0, 1_000_000))

            # Step 1.1: positive perturbation, theta_c + eps*z
            _perturb(self.clients[i].model, seed, self.mu)
            with self.phase('client', i):
                with torch.no_grad():
                    h_pos = self.clients[i].model(x)
            self.charge_cut_activation(h_pos)
            with self.phase('server', i):
                with torch.no_grad():
                    loss_pos = self.criterion(self.server.model(h_pos), y)
            self.charge_cut_scalar(1, direction='down')

            # Step 1.2: negative perturbation, theta_c - eps*z
            _perturb(self.clients[i].model, seed, -2 * self.mu)
            with self.phase('client', i):
                with torch.no_grad():
                    h_neg = self.clients[i].model(x)
            self.charge_cut_activation(h_neg)
            with self.phase('server', i):
                with torch.no_grad():
                    loss_neg = self.criterion(self.server.model(h_neg), y)
            self.charge_cut_scalar(1, direction='down')

            # Step 1.3: restore, and store the projected gradient. Eq. 6-7's
            # SYMMETRIC two-point estimate: (L(+eps) - L(-eps)) / 2*eps, averaged
            # over Q directions. An earlier version used the one-sided
            # (L(theta+d) - L(theta))/d form, which carries an O(eps) bias the
            # symmetric estimator cancels to O(eps^2).
            _perturb(self.clients[i].model, seed, self.mu)
            grads.append(
                (seed, (loss_pos.item() - loss_neg.item()) / (2 * self.mu * self.Q))
            )

        # --- Alg. 1 Phase 2 (L27-30): server FO update on the ORIGINAL client
        # parameters, before the client applies anything. `compute_grad` pass.
        with self.phase('client', i):
            with torch.no_grad():
                h = self.clients[i].model(x)
        self.charge_cut_activation(h)
        self.charge_cut_labels(y)     # the loss is evaluated server-side

        smashed_data = h.detach()
        with self.phase('server', i):
            self.server.optimizer.zero_grad()
            out = self.server.model(smashed_data)
            loss = self.criterion(out, y)
            loss.backward()
            self.server.optimizer.step()

            with torch.no_grad():
                baseline_loss = loss.item()
                _, predicted = torch.max(out.data, 1)
                train_correct = predicted.eq(y.view_as(predicted)).sum().item()

        # --- Alg. 1 Phase 3 (L32-35): the client applies its Q stored ZO
        # estimates, regenerating each direction from its seed. No gradient
        # buffer and no optimizer state are ever allocated.
        with self.phase('client', i):
            for seed, g_hat in grads:
                _zo_update(self.clients[i].model, seed, self.client_lr * g_hat)

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
