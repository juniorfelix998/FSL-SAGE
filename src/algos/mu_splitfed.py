# ------------------------------------------------------------------------------
# MU-SplitFed (Liang et al., "Towards Straggler-Resilient Split Federated
# Learning: An Unbalanced Update Approach", NeurIPS 2025, arXiv:2510.21155),
# run against the paper's Algorithm 1 and adapted to vision.
#
# PROVENANCE. The authors' code (Johnny-Zip/MU-SplitFed) is LLM/OPT-only,
# single-client, has no aggregation, and does not run as published (it imports
# a `cezo_fl` package that is not in the repository). The vision baseline here
# is therefore ported from the third-party ResNet-18 reimplementation in
# HKU-WILL-Lab/HO-SFL (HO_SFL_CV/src/runner/mu_splitfed_runner.py), and checked
# step by step against the paper. Label it a reimplementation wherever it is
# reported.
#
# VISION ADAPTATION. MU-SplitFed itself uses no LoRA and no LLM-specific loss:
# it is full-parameter zeroth-order (ZO) training. So it maps directly onto
# this benchmark's ResNet-18 split (client = conv1..layer2, server =
# layer3..fc, the same middle cut as every method), with cross-entropy loss
# and the cut activation in place of the LLM hidden state.
#
# ONE PAPER ROUND (Alg.1) = one iteration here, over the sampled clients S:
#   L3     each sampled client pulls the global client model x_c (download);
#          the Split Server starts that client's server copy from x_s (local)
#   Eq.4   client: clean h, h+ = f(x_c + mu*u), h- = f(x_c - mu*u), forward
#          only (no backward); sends h, h+, h- and the labels  -> 3 activations
#   Eq.5   Split Server: tau ZO steps on that client's copy, on the clean h
#          (two-sided, one random direction each)
#   Eq.6   server sends ONE scalar back: F(x_s^tau, h+) - F(x_s^tau, h-),
#          evaluated with the post-tau copy; the client takes one ZO step
#   L11    each client uploads its model; Eq.7 (as in the paper's proof; the
#          printed sign is a typo): x <- x + eta_g * mean over S of (x_m - x),
#          for the client models AND the server copies (eta_g = lr_g = 0.3)
# No activation gradient ever crosses the cut. Hyperparameters are the paper's
# Table 6: tau = 2, mu = 0.005, lr_c = 0.005, lr_s = 0.01, lr_g = 0.3, one
# perturbation (P = 1).
#
# Differences from the original port (changed after review):
#   - Aggregation after EVERY iteration, as in Alg.1 (was: once per local
#     epoch, i.e. 24 client steps between aggregations).
#   - Client sampling (`sampled_clients`, default 5 of 10 = the paper's 50%
#     partial participation, p.8). Each iteration takes one batch from each of
#     K clients drawn at random among the clients with batches left this round
#     (fair sampling: every client still uses all its data every round, like
#     every other method). K = 1 runs one client per iteration (client_step),
#     still aggregating every iteration.
#   - Client-model traffic is charged per participation: one download (L3) and
#     one upload (L11). The server-copy averaging is NOT charged: in the paper
#     it happens inside the Split Server (no network). (SplitFedv1 is still
#     charged under the benchmark's older convention.)
#   - Random init (use_pretrained/freeze_bn = false), the same as every method
#     in the benchmark; BatchNorm trains in train mode (as in the reference).
#
# Deliberate differences from the paper:
#   - The random direction u is Gaussian (both codebases); the paper's theory
#     draws it uniformly from a sphere.
#   - Batches of size 1 are skipped (BatchNorm in train mode cannot run them),
#     as in the reference; no such batch occurs at batch 32 or 256 on MNIST.
#   - Benchmark settings: 10 clients, one round = one pass over every client's
#     data. The reported run uses batch 256 (the batch every method uses).
#     The paper's batch 32 (5 x 32 = 160 images per update) diverges here
#     (loss -> NaN) from random init on MNIST.
#
# Expected accuracy: the HO-SFL paper (arXiv:2603.14773, Fig.3) reports this
# reimplementation near chance (~10-15%) on CIFAR-10. Report what it gets; do
# not tune it.
# ------------------------------------------------------------------------------
import time
import copy
import numpy as np
import torch

from algos import register_algorithm, FLAlgorithm
from utils.memory import module_static_bytes
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
def _zo_update(model, seed, step):
    """theta <- theta - step * u, with u REGENERATED from its seed.

    Mathematically identical to what this file used to do -- accumulate
    `scalar * u` into `param.grad` and call `SGD(lr).step()` -- because the
    reference uses plain SGD with no momentum or weight decay, so its update is
    exactly `param -= lr * grad`. The difference is purely in what gets
    allocated: no `.grad` buffer is created for any parameter. That matters here
    because client memory is the metric a backprop-free method is measured on,
    and a full-size gradient tensor is precisely the thing it claims not to
    need."""
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
@register_algorithm("mu_splitfed")
class MU_SplitFed(FLAlgorithm):
    aggregated_server: torch.nn.Module

    def __init__(self, *args, **kwargs):
        super(MU_SplitFed, self).__init__(*args, **kwargs)

        self.tau = self.cfg.tau
        self.mu = self.cfg.zo_mu
        self.eta_g = self.cfg.lr_g
        self.use_pretrained = self.cfg.get('use_pretrained', False)
        self.freeze_bn = self.cfg.get('freeze_bn', False)

        # Initialization: random by default (benchmark fairness); the vision
        # reference's pretrained + frozen-BN setting is available via config.
        if self.use_pretrained:
            full_state = pretrained_resnet18_state_dict()
            for c in self.clients:
                load_pretrained_client(c.model, full_state)
            load_pretrained_server(self.server.model, full_state)
        if self.freeze_bn:
            for c in self.clients:
                freeze_batchnorm_affine(c.model)
            freeze_batchnorm_affine(self.server.model)

        # every client starts from the same global client model x_c
        for c in self.clients[1:]:
            c.model.load_state_dict(self.clients[0].model.state_dict())

        # The Split Server keeps one server-model copy per client (paper p.3);
        # each is re-started from the global x_s when its client is sampled.
        self.servers = [copy.deepcopy(self.server) for _ in self.clients]

        # Both sides are zeroth-order and updated in place by _zo_update, so no
        # `.grad` is ever allocated. The SGD objects exist only because the
        # shared loop reads their learning rate (plain SGD: no state).
        self.lr_c = float(self.cfg.lr_c)
        self.lr_s = float(self.cfg.lr_s)
        for c in self.clients:
            c.optimizer = torch.optim.SGD(c.model.parameters(), lr=self.lr_c)
            c.lr_scheduler = None
        for srv in self.servers:
            srv.optimizer = torch.optim.SGD(srv.model.parameters(), lr=self.lr_s)
            srv.lr_scheduler = None

        # the global models x_c and x_s (Fed Server / Split Server)
        self.aggregated_client = copy.deepcopy(self.clients[0].model)
        self.aggregated_server = copy.deepcopy(self.server.model)

        # Alg.1: clients sampled per iteration (paper: 50% of 10 = 5)
        self.K = int(self.cfg.get('sampled_clients', 1))
        self.group_log = []
        self._agg_time = 0.0

    def full_model(self, x):
        return self.aggregated_server(self.aggregated_client(x))

    # -- memory partition ---------------------------------------------------
    # Per-device server value: the global x_s plus ONE client's server copy
    # being trained. The other K-1 sampled clients' copies, alive during the
    # same iteration, are listed separately (server_side_extra_memory).
    def server_side_modules(self):
        return [self.aggregated_server, self.servers[0].model]

    def server_side_optimizers(self):
        return [self.servers[0].optimizer]

    def server_side_extra_memory(self):
        params, grads = module_static_bytes(self.servers[0].model)
        others = max(self.K - 1, 0)
        return [{
            'item': f'server copies of the other {others} sampled clients (params)',
            'mb': others * (params + grads) / 1024 ** 2,
            'note': 'paper p.3: one server-model copy per client, kept by the Split Server during an iteration',
        }]

    def special_models_train_mode(self, t):
        # BatchNorm stays in train mode during every perturbed forward, as in
        # the reference runner (a blanket .train() every round).
        if t > 0:
            self.aggregated_server.train()

    def special_models_eval_mode(self):
        self.aggregated_server.eval()

    def _aggregate_weights(self, global_state, state_dicts):
        '''Eq.7 (as used in the paper's proof): x + eta_g * mean(x_m - x).'''
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

    # -- one paper round (Alg.1) over the sampled clients ---------------------
    def group_size(self):
        return self.K

    def client_step(self, rd_cl_ep_it, x, y):
        # K = 1: one client per iteration, still aggregating every iteration
        i = rd_cl_ep_it[1]
        return self.group_step([(rd_cl_ep_it, x, y)]).get(i, {})

    def group_step(self, batch):
        '''One MU-SplitFed round (paper Alg.1) over the sampled clients in
        `batch` (a list of ((t, i, j, k), x, y)). Returns {client: metrics}.'''
        mets = {}
        self.meter.enabled = bool(self.cfg.get('measure_memory', True)) and \
            any(self.should_probe_memory(*ix) for ix, _, _ in batch)
        g_client = self.aggregated_client.state_dict()
        g_server = self.aggregated_server.state_dict()
        S = []

        def timed(i, side, fn):
            t0 = time.time()
            with self.phase(side, i):
                out = fn()
            key = f'{side}_model_compute_time'
            mets[i][key] = mets[i].get(key, 0.0) + time.time() - t0
            return out

        for (t, i, j, k), x, y in batch:
            # BatchNorm in train mode cannot run a batch of one (reference
            # L138-139); skipped batches send nothing and take no part.
            if x.size(0) == 1:
                continue
            S.append(i)
            mets[i] = {'client_model_compute_time': 0.0,
                       'server_model_compute_time': 0.0}
            client = self.clients[i].model
            server = self.servers[i].model
            self.hold('client', x, y, i=i)

            # L3: pull the global client model (download); the Split Server
            # starts this client's server copy from x_s (local, not charged)
            client.load_state_dict(g_client)
            server.load_state_dict(g_server)
            self.charge_weights_model(self.aggregated_client, 'client', 'down')

            # Eq.4: clean h, forward only
            def fwd():
                with torch.no_grad():
                    return client(x)
            h_fixed = timed(i, 'client', fwd)
            self.charge_cut_activation(h_fixed)
            self.charge_cut_labels(y)     # every loss is computed server-side

            # Eq.5: tau ZO steps on this client's server copy, on the clean h
            def server_steps():
                for _ in range(self.tau):
                    s_seed = int(np.random.randint(0, 1_000_000))
                    _perturb(server, s_seed, self.mu)
                    with torch.no_grad():
                        loss_p = self.criterion(server(h_fixed), y)
                    _perturb(server, s_seed, -2 * self.mu)
                    with torch.no_grad():
                        loss_n = self.criterion(server(h_fixed), y)
                    _perturb(server, s_seed, self.mu)            # restore
                    scalar_s = (loss_p.item() - loss_n.item()) / (2 * self.mu)
                    _zo_update(server, s_seed, self.lr_s * scalar_s)
            timed(i, 'server', server_steps)

            # Eq.4: perturbed forwards h+ and h- (each sent as computed)
            c_seed = int(np.random.randint(0, 1_000_000))
            _perturb(client, c_seed, self.mu)
            h_pos = timed(i, 'client', fwd)
            self.charge_cut_activation(h_pos)
            _perturb(client, c_seed, -2 * self.mu)
            h_neg = timed(i, 'client', fwd)
            self.charge_cut_activation(h_neg)
            _perturb(client, c_seed, self.mu)                    # restore

            # Eq.6: one scalar from the post-tau server copy
            def evaluate_pm():
                with torch.no_grad():
                    out_p = server(h_pos)
                    out_n = server(h_neg)
                    return out_p, self.criterion(out_p, y), self.criterion(out_n, y)
            out_c_p, loss_c_p, loss_c_n = timed(i, 'server', evaluate_pm)
            scalar_c = (loss_c_p.item() - loss_c_n.item()) / (2 * self.mu)
            self.charge_cut_scalar(1, direction='down')

            # client ZO step, then upload for aggregation (L10-11)
            timed(i, 'client', lambda: _zo_update(client, c_seed, self.lr_c * scalar_c))
            self.charge_weights_model(client, 'client', 'up')

            with torch.no_grad():
                correct = out_c_p.argmax(1).eq(y).sum().item()
            mets[i]['acc'] = correct / y.size(0)
            mets[i]['loss'] = (loss_c_p.item() + loss_c_n.item()) / 2.0

        # Eq.7: aggregate the client models and the server copies over S. The
        # server copies are averaged inside the Split Server: not charged.
        if S:
            t0 = time.time()
            self.aggregated_client.load_state_dict(self._aggregate_weights(
                g_client, [self.clients[i].model.state_dict() for i in S]))
            self.aggregated_server.load_state_dict(self._aggregate_weights(
                g_server, [self.servers[i].model.state_dict() for i in S]))
            self._agg_time += time.time() - t0
            self.group_log.append(S)
        self.meter.end_step()
        self._phase_times = {}
        return mets

    def aggregate(self):
        # Everything was aggregated per iteration in group_step (Eq.7); keep
        # every client's copy in sync with the final global model for eval.
        for c in self.clients:
            c.model.load_state_dict(self.aggregated_client.state_dict())
        out = {'client_agg_compute_time': self._agg_time}
        self._agg_time = 0.0
        return out

# ------------------------------------------------------------------------------
