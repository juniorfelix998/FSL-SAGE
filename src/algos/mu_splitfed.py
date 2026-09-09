# ------------------------------------------------------------------------------
# Ported from HKU-WILL-Lab/HO-SFL's HO_SFL_CV/src/runner/mu_splitfed_runner.py
# (+ its src/models/registry.py and src/models/split_wrapper.py, which build
# the pretrained/frozen-BN backbone this runner trains), verified directly
# against the actual repo source and its conf/mu_splitfed.yaml.
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
# IMPORTANT EXPECTATION-SETTING, per the HO-SFL paper itself (arXiv:2603.14773,
# Figure 3): the paper's OWN reported validation accuracy for MU-SplitFed on
# CIFAR-10 stays FLAT near chance level (~10-15%) for the entire run (160k
# processed samples), on both IID and Non-IID splits, while every other method
# in that figure climbs to 60-80%. This is the paper's own point -- MU-SplitFed
# is a deliberately weak backprop-free baseline that HO-SFL is shown to
# dramatically improve on. A near-chance accuracy for this method in this
# benchmark is the EXPECTED, paper-confirmed result, not a sign of a broken
# port. Do not tune this method to chase a "good" accuracy number.
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
# Matches the reference's own recipe (its conf/mu_splitfed.yaml sets
# `use_pretrained: True`, `freeze_bn: True`) -- both are ported here as
# `use_pretrained`/`freeze_bn` config fields:
#   - `use_pretrained`: loads ImageNet-pretrained torchvision ResNet18 weights
#     into the client/server split at this harness's existing middle-cut
#     boundary (conv1/bn1/layer1/layer2 | layer3/layer4/fc), matching the
#     reference's registry.get_model()+split_wrapper.split_model() split
#     point (split_point=6, the same boundary). The server's `fc` is left at
#     this harness's own random init (pretrained fc is 1000-way ImageNet,
#     this benchmark's is num_classes-way).
#   - `freeze_bn`: sets `requires_grad=False` on every BatchNorm2d's affine
#     `weight`/`bias`, matching the reference's src/models/registry.py. This
#     does NOT put BatchNorm in eval mode or freeze its running mean/var --
#     verified directly against the reference runner, which calls a blanket
#     `.train()` on the whole model every round, re-enabling ordinary
#     train-mode running-stat updates regardless of this flag. The only
#     lasting effect is that BatchNorm's affine params are invisible to the
#     ZO perturbation loop (`_perturb`/`_perturb_accumulate_grad` already skip
#     any `requires_grad=False` param below), so they stay pinned at their
#     pretrained values while running stats keep adapting normally. (An
#     earlier version of this file forced BatchNorm into permanent .eval()
#     mode, assuming that's what "freeze" meant -- that's wrong and caused
#     the loss to explode to NaN on a from-scratch model with no calibrated
#     running stats to fall back on; this version matches the reference
#     exactly instead.)
#
# Known, documented simplification vs. the reference (see plan discussion):
# one real batch per client per round -- `client_step` does real work only at
# the first (epoch, iter) of the round; every other call that round is a
# no-op, analogous to the reference's own per-round `next(iter(loader))`. Uses
# a single zeroth-order perturbation direction per gradient estimate (no
# averaging), exactly matching the reference -- an earlier version of this
# file added multi-direction averaging as an from-scratch stabilization
# attempt; that's a real deviation from the reference and has been reverted
# for a faithful, fair comparison.
# ------------------------------------------------------------------------------
import time
import copy
import numpy as np
import torch

from algos import register_algorithm, FLAlgorithm
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
@register_algorithm("mu_splitfed")
class MU_SplitFed(FLAlgorithm):
    aggregated_server: torch.nn.Module

    def __init__(self, *args, **kwargs):
        super(MU_SplitFed, self).__init__(*args, **kwargs)

        self.tau = self.cfg.tau
        self.mu = self.cfg.zo_mu
        self.eta_g = self.cfg.lr_g
        self.use_pretrained = self.cfg.use_pretrained
        self.freeze_bn = self.cfg.freeze_bn

        # matches the reference's own registry.get_model(): pretrained
        # backbone loaded (and BN affine frozen) BEFORE the servers/aggregated
        # deep copies below, so every copy inherits the same weights/frozen
        # flags without re-doing this per copy.
        if self.use_pretrained:
            full_state = pretrained_resnet18_state_dict()
            for c in self.clients:
                load_pretrained_client(c.model, full_state)
            load_pretrained_server(self.server.model, full_state)

        if self.freeze_bn:
            for c in self.clients:
                freeze_batchnorm_affine(c.model)
            freeze_batchnorm_affine(self.server.model)

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

    # the server host holds one replica per client
    def server_side_modules(self):
        return [s.model for s in self.servers]

    def server_side_optimizers(self):
        return [s.optimizer for s in self.servers]

    def special_models_train_mode(self, t):
        # Deliberately does NOT force the client/server into eval() around the
        # zeroth-order probe loop, unlike ho_sfl. Each batch runs 2*tau + 2
        # perturbed server forwards and 2 perturbed client forwards in train
        # mode, so BatchNorm running stats are updated with perturbed weights.
        # That looks like a bug, and was investigated as one -- but the
        # reference (HKU-WILL-Lab/HO-SFL's mu_splitfed_runner.py) calls a
        # blanket .train() on the whole model every round and behaves
        # identically, so this is FAITHFUL. Do not "fix" it without also
        # changing the reference comparison.
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

        # client forward under no_grad -- a zeroth-order client retains no
        # autograd activations, so its activation peak is ~0 by construction
        with self.phase('client', i):
            with torch.no_grad():
                h_fixed = self.clients[i].model(x)
        self.charge_cut_activation(h_fixed)
        self.charge_cut_labels(y)     # every loss below is computed server-side

        # tau local server-only zeroth-order steps, reusing h_fixed -- no
        # further client communication needed for these.
        with self.phase('server', i):
            for _ in range(self.tau):
                self.servers[i].optimizer.zero_grad()
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
                _perturb_accumulate_grad(self.servers[i].model, s_seed, scalar_s)
                self.servers[i].optimizer.step()

        # client-side zeroth-order step: two more client forwards, each a
        # fresh activation upload. The perturbed forwards are the CLIENT's
        # compute and the evaluations against the server are the SERVER's --
        # bracketed separately so neither is charged the other's time.
        c_seed = int(np.random.randint(0, 1_000_000))

        _perturb(self.clients[i].model, c_seed, self.mu)
        with self.phase('client', i):
            with torch.no_grad():
                h_pos = self.clients[i].model(x)
        self.charge_cut_activation(h_pos)

        _perturb(self.clients[i].model, c_seed, -2 * self.mu)
        with self.phase('client', i):
            with torch.no_grad():
                h_neg = self.clients[i].model(x)
        self.charge_cut_activation(h_neg)

        _perturb(self.clients[i].model, c_seed, self.mu)  # restore

        with self.phase('server', i):
            with torch.no_grad():
                out_c_p = self.servers[i].model(h_pos)
                loss_c_p = self.criterion(out_c_p, y)
                out_c_n = self.servers[i].model(h_neg)
                loss_c_n = self.criterion(out_c_n, y)

        scalar_c = (loss_c_p.item() - loss_c_n.item()) / (2 * self.mu)
        # one scalar sent back so the client can replay its own update --
        # drives this client's own update this round, same role as HO-SFL's
        # g_a_m, so it's classified as cut traffic, not an aggregation event.
        self.charge_cut_scalar(1, direction='down')

        with self.phase('client', i):
            _perturb_accumulate_grad(self.clients[i].model, c_seed, scalar_c)
            self.clients[i].optimizer.step()

        with torch.no_grad():
            _, predicted = torch.max(out_c_p.data, 1)
            train_correct = predicted.eq(y.view_as(predicted)).sum().item()

        return {
            'acc': train_correct / y.size(dim=0),
            'loss': (loss_c_p.item() + loss_c_n.item()) / 2.0,
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
            self.charge_weights_roundtrip(self.aggregated_client, 'client')
        for s in self.servers:
            s.model.load_state_dict(agg_server)
            self.charge_weights_roundtrip(self.aggregated_server, 'server')

        return {'client_agg_compute_time': time.time() - t0}

# ------------------------------------------------------------------------------
