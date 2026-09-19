#!/usr/bin/env python
# ------------------------------------------------------------------------------
# Correctness gate for the benchmark's measurement layer.
#
#     python test/check_accounting.py
#
# Exits non-zero on the first failure. Run this BEFORE a sweep -- it takes
# seconds, while a sweep takes hours, and a silently-wrong byte counter
# invalidates every number the sweep produces.
#
# It checks three things:
#   1. MemoryMeter unit invariants (exact byte counts on known shapes, zero
#      under no_grad, release after backward, no double-counting of views).
#   2. Communication counts against CLOSED-FORM expectations, per method. This
#      is the check that actually answers "are we measuring comm correctly" --
#      everything else is inspection.
#   3. Cross-method orderings that the paper's claims depend on (a BP-free
#      method must show zero returned-gradient bytes; a zeroth-order client must
#      retain ~zero activations).
#
# Deliberately does NOT go through hydra's CLI: it builds an OmegaConf config
# directly and drives the real shared training loop, so it exercises exactly the
# code path a real run uses while staying runnable in environments where the
# hydra argparse entry point is unavailable.
# ------------------------------------------------------------------------------
import inspect
import math
import os
import re
import sys

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, SRC)

from models import Client, Server                                    # noqa: E402
from utils.memory import (                                           # noqa: E402
    MemoryMeter, module_static_bytes, optimizer_state_bytes
)
import algos                                                         # noqa: E402
from algos import ALGORITHM_REGISTRY, _run_fl_algorithm              # noqa: E402

FAILURES = []
CHECKS = 0


def check(label, condition, detail=''):
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label}" + (f"\n          {detail}" if detail else ''))
        FAILURES.append(label)


def check_eq(label, got, want):
    check(label, got == want, f"got {got!r}, want {want!r}")


# ==============================================================================
# 1. MemoryMeter unit invariants
# ==============================================================================
def test_memory_meter():
    print("\n[1] MemoryMeter unit invariants")

    # -- exact value on a known shape ------------------------------------
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
                      nn.Conv2d(8, 8, 3, padding=1))
    x = torch.randn(4, 3, 32, 32)
    meter = MemoryMeter()
    meter.register_params(m)
    with meter.phase(('client', 0)):
        out = m(x)
    # conv1 saves its input x; relu saves its own output; conv2 saves the SAME
    # storage as the relu output (inplace-free but shared), so it counts once.
    expected = 4 * 3 * 32 * 32 * 4 + 4 * 8 * 32 * 32 * 4
    check_eq("exact activation bytes on 2-conv toy",
             meter.peak[('client', 0)], expected)

    # -- release after backward ------------------------------------------
    # Exact release: freeing the graph drops every saved slot, so nothing is
    # still counted as autograd-retained -- even though `x` and `out` are still
    # live Python locals here. That distinction is the whole point of the
    # sentinel-based release (a weakref-based one would still count `x`, and
    # would report a decoupled method as holding bytes across the cut).
    out.sum().backward()
    check_eq("all saved bytes released after backward, despite live locals",
             meter.live_bytes(('client', 0)), 0)
    check("live locals are still usable after release accounting",
          x.shape == (4, 3, 32, 32) and out.requires_grad)

    # -- zero under no_grad: the whole ZO discrimination rests on this ----
    meter2 = MemoryMeter()
    meter2.register_params(m)
    with meter2.phase(('client', 0)):
        with torch.no_grad():
            m(x)
    check_eq("peak is exactly 0 under torch.no_grad()",
             meter2.peak.get(('client', 0), 0), 0)

    # -- parameter views must not be double counted ----------------------
    # nn.Linear saves weight.t(), a non-leaf sharing the parameter's storage.
    lin = nn.Linear(16, 4)
    xv = torch.randn(2, 16)
    meter3 = MemoryMeter()
    meter3.register_params(lin)
    with meter3.phase(('client', 0)):
        lin(xv).sum()
    check_eq("Linear's weight.t() contributes no activation bytes",
             meter3.peak.get(('client', 0), 0), 2 * 16 * 4)

    # -- a reshape shares storage and must count once --------------------
    meter4 = MemoryMeter()
    with meter4.phase(('client', 0)):
        base = torch.randn(8, 8, requires_grad=True)
        y = (base * 2).view(64)
        y.sum().backward()
    check("a view() adds no extra bytes",
          meter4.peak.get(('client', 0), 0) <= 8 * 8 * 4,
          f"peak={meter4.peak.get(('client', 0), 0)}")

    # -- meter disabled is a true no-op ----------------------------------
    meter5 = MemoryMeter(enabled=False)
    with meter5.phase(('client', 0)):
        m(x)
    check_eq("disabled meter records nothing", meter5.peak, {})

    # -- static helpers --------------------------------------------------
    lin2 = nn.Linear(10, 5)                       # 50 + 5 params
    params, grads = module_static_bytes(lin2)
    check_eq("module_static_bytes params", params, (10 * 5 + 5) * 4)
    check_eq("module_static_bytes grads (none yet)", grads, 0)
    lin2(torch.randn(1, 10)).sum().backward()
    _, grads = module_static_bytes(lin2)
    check_eq("module_static_bytes grads (after backward)", grads, (10 * 5 + 5) * 4)

    opt = torch.optim.SGD(lin2.parameters(), lr=0.1)
    check_eq("plain SGD holds no optimizer state",
             optimizer_state_bytes(opt), 0)
    adam = torch.optim.Adam(lin2.parameters(), lr=0.1)
    adam.step()
    two_x = 2 * (10 * 5 + 5) * 4
    got = optimizer_state_bytes(adam)
    # exp_avg + exp_avg_sq is 2x the parameters; Adam also keeps a small
    # per-parameter `step` tensor, so 2x is a floor rather than an equality
    check("Adam holds at least 2x params of state",
          two_x <= got <= two_x + 256, f"got {got}, want ~{two_x}")


# ==============================================================================
# 2 + 3. Communication closed forms, driven through the real training loop
# ==============================================================================
# A deliberately tiny split model, so the expected byte counts are hand-checkable.
CUT_CHANNELS, CUT_HW = 4, 4
IN_CH, IMG, NUM_CLASSES = 3, 8, 5


class TinyClient(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(IN_CH, CUT_CHANNELS, 3, padding=1),
                                 nn.ReLU(), nn.AdaptiveAvgPool2d(CUT_HW))

    def forward(self, x):
        return self.net(x)


class TinyServer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(CUT_CHANNELS * CUT_HW * CUT_HW, NUM_CLASSES)

    def forward(self, x):
        return torch.log_softmax(self.fc(x.flatten(1)), dim=1)


class TinyAux(nn.Module):
    '''Stands in for the harness's AuxiliaryModel: a local head that can emit
    either a class prediction (forward_inner) or a gradient at the cut
    (forward), which is the interface cse_fsl / fsl_sage / dsl_aux use.'''

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(CUT_CHANNELS * CUT_HW * CUT_HW, NUM_CLASSES)
        self.criterion = nn.NLLLoss()
        self.optimizer = None
        self.server = None
        self.data_x = torch.tensor([])
        self.data_y = torch.tensor([])
        self.data_labels = torch.tensor([], dtype=torch.long)

    def set_optimizer_lr_scheduler(self, optimizer, lr_scheduler):
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def forward_inner(self, x):
        return torch.log_softmax(self.fc(x.flatten(1)), dim=1)

    def forward(self, x, label):
        x = x.requires_grad_(True)
        loss = self.criterion(self.forward_inner(x), label)
        return torch.autograd.grad(loss, x, create_graph=True)[0]

    # -- the alignment-buffer interface fsl_sage drives ------------------
    def add_datapoint(self, x, label):
        self.data_x = torch.cat((self.data_x, x), dim=0)
        self.data_labels = torch.cat((self.data_labels, label), dim=0).long()

    def refresh_data(self):
        '''Mirrors AuxiliaryModel.refresh_data: a real server forward/backward
        over the stored buffer to obtain true cut-layer gradients. Runs
        server-side, which is why fsl_sage brackets it as a server phase and
        charges only the aligned surrogate's download.'''
        if self.data_x.numel() == 0:
            return
        self.server.optimizer.zero_grad()
        ins = self.data_x.clone().detach().requires_grad_(True)
        loss = self.server.criterion(self.server.model(ins), self.data_labels)
        loss.backward()
        self.data_y = ins.grad.clone().detach()
        self.server.optimizer.zero_grad()

    def align(self):
        if self.data_x.numel() == 0:
            return
        self.optimizer.zero_grad()
        pred = self.forward(self.data_x.clone().detach(), self.data_labels)
        nn.functional.mse_loss(pred, self.data_y).backward()
        self.optimizer.step()


def make_cfg(algo_name, num_clients, rounds, batch_size, extra_algo=None):
    cfg = OmegaConf.create({
        'algorithm': {'name': algo_name},
        'model': {
            'name': 'tiny',
            'client': {'epoch': 1, 'batch_size': batch_size},
            'server': {}, 'auxiliary': {},
        },
        'dataset': {'name': 'synthetic', 'distribution': 'iid'},
        'num_clients': num_clients,
        'rounds': rounds,
        'seed': 0,
        'save': False,
        'checkpoint_interval': 10 ** 9,
        'agg_factor': [1.0 / num_clients] * num_clients,
        'use_64bit': False,
        'comm_threshold_mb': float('inf'),
        'measure_memory': True,
        'mem_probe_batches': 10 ** 9,       # instrument every batch in the test
        'device': 'cpu',
    })
    if extra_algo:
        cfg.algorithm.update(extra_algo)
    return cfg


def build_run(algo_name, num_clients, samples_per_client, batch_size, rounds,
              extra_algo=None):
    torch.manual_seed(0)
    cfg = make_cfg(algo_name, num_clients, rounds, batch_size, extra_algo)

    server = Server(TinyServer(), OmegaConf.create({
        'optimizer': {'name': 'sgd', 'options': {'lr': 0.01}}
    }), device='cpu')

    client_cfg = OmegaConf.create({
        'epoch': 1, 'batch_size': batch_size,
        'optimizer': {'name': 'sgd', 'options': {'lr': 0.01}},
    })
    clients = []
    for i in range(num_clients):
        xs = torch.randn(samples_per_client, IN_CH, IMG, IMG)
        ys = torch.randint(0, NUM_CLASSES, (samples_per_client,))
        loader = DataLoader(TensorDataset(xs, ys), batch_size=batch_size,
                            shuffle=False)
        c = Client(i, loader, TinyClient(), client_cfg, device='cpu')
        aux = TinyAux()
        aux.server = server          # main.py passes server= to the constructor
        c.init_auxiliary(aux, OmegaConf.create({
            'optimizer': {'name': 'sgd', 'options': {'lr': 0.01}}
        }))
        clients.append(c)

    xs = torch.randn(2 * batch_size, IN_CH, IMG, IMG)
    ys = torch.randint(0, NUM_CLASSES, (2 * batch_size,))
    test_loader = DataLoader(TensorDataset(xs, ys), batch_size=batch_size)

    results = _run_fl_algorithm(
        cfg, server, clients, test_loader, None, torch.device('cpu'),
        logger_fn=lambda *a, **kw: None
    )
    return cfg, results


def batch_sizes(n, b):
    '''Actual per-batch sizes, including a ragged final batch.'''
    return [min(b, n - s) for s in range(0, n, b)]


def act_bytes(bs):
    return bs * CUT_CHANNELS * CUT_HW * CUT_HW * 4


def label_bytes(bs):
    return bs * 8            # int64 labels


def client_model_bytes():
    return sum(p.numel() * p.element_size() for p in TinyClient().parameters())


def server_model_bytes():
    return sum(p.numel() * p.element_size() for p in TinyServer().parameters())


def aux_model_bytes():
    return sum(p.numel() * p.element_size() for p in TinyAux().parameters())


def last(bd, key):
    return bd[-1][key]


def test_comm_closed_form():
    print("\n[2] Communication vs. closed form")

    M, N, B, R = 2, 10, 4, 2           # clients, samples/client, batch, rounds
    bs_list = batch_sizes(N, B)
    per_round_act = M * sum(act_bytes(b) for b in bs_list)
    per_round_lbl = M * sum(label_bytes(b) for b in bs_list)

    # ---- SplitFedv2: activation up + gradient down, every batch --------
    cfg, res = build_run('sl_single_server', M, N, B, R)
    bd = res.comm_breakdown
    check_eq("SplitFedv2 cut.act_up", last(bd, 'cut.act_up'), R * per_round_act)
    check_eq("SplitFedv2 cut.grad_down", last(bd, 'cut.grad_down'),
             R * per_round_act)
    check_eq("SplitFedv2 cut.labels_up", last(bd, 'cut.labels_up'),
             R * per_round_lbl)
    check_eq("SplitFedv2 weights (client FedAvg round trip)",
             last(bd, 'weights.client_up') + last(bd, 'weights.client_down'),
             R * 2 * M * client_model_bytes())
    check_eq("SplitFedv2 charges no server weight traffic (shared server)",
             last(bd, 'weights.server_up') + last(bd, 'weights.server_down'), 0)
    check_eq("SplitFedv2 cut + weights == total",
             res.comm_load_cut[-1] + res.comm_load_weights[-1],
             res.comm_load[-1])
    splitfedv2 = res

    # ---- SplitFedv1: same cut, PLUS server-replica aggregation ---------
    cfg, res = build_run('sl_multi_server', M, N, B, R)
    bd = res.comm_breakdown
    check_eq("SplitFedv1 cut.act_up", last(bd, 'cut.act_up'), R * per_round_act)
    check_eq("SplitFedv1 cut.grad_down", last(bd, 'cut.grad_down'),
             R * per_round_act)
    # the bug this fixes: server aggregation used to be charged ZERO
    check_eq("SplitFedv1 DOES charge server aggregation (was 0 before)",
             last(bd, 'weights.server_up') + last(bd, 'weights.server_down'),
             R * 2 * M * server_model_bytes())
    check("SplitFedv1 total exceeds SplitFedv2 total (extra server FedAvg)",
          res.comm_load[-1] > splitfedv2.comm_load[-1],
          f"v1={res.comm_load[-1]} v2={splitfedv2.comm_load[-1]}")

    # ---- CSE-FSL: activation only, 1 batch in every q ------------------
    Q = 3
    cfg, res = build_run('cse_fsl', M, N, B, R, extra_algo={
        'server_update_interval': Q
    })
    bd = res.comm_breakdown
    uploaded = [bs_list[it] for it in range(0, len(bs_list), Q)]
    check_eq("CSE-FSL cut.act_up (only every q-th batch)",
             last(bd, 'cut.act_up'), R * M * sum(act_bytes(b) for b in uploaded))
    check_eq("CSE-FSL cut.grad_down is ZERO (no gradient crosses back)",
             last(bd, 'cut.grad_down'), 0)
    check_eq("CSE-FSL charges an auxiliary FedAvg round trip",
             last(bd, 'weights.aux_up') + last(bd, 'weights.aux_down'),
             R * 2 * M * aux_model_bytes())
    check("CSE-FSL cut traffic is far below SplitFedv2's",
          res.comm_load_cut[-1] < splitfedv2.comm_load_cut[-1] / 2,
          f"cse={res.comm_load_cut[-1]} v2={splitfedv2.comm_load_cut[-1]}")
    check_eq("CSE-FSL cut + weights == total",
             res.comm_load_cut[-1] + res.comm_load_weights[-1],
             res.comm_load[-1])
    cse = res

    # ---- FSL-SAGE: activation only, one-way auxiliary download ---------
    cfg, res = build_run('fsl_sage', M, N, B, R, extra_algo={
        'server_update_interval': Q, 'align_interval': 1
    })
    bd = res.comm_breakdown
    check_eq("FSL-SAGE cut.act_up (only every q-th batch)",
             last(bd, 'cut.act_up'), R * M * sum(act_bytes(b) for b in uploaded))
    check_eq("FSL-SAGE cut.grad_down is ZERO (surrogate replaces it)",
             last(bd, 'cut.grad_down'), 0)
    check_eq("FSL-SAGE auxiliary is charged DOWNLOAD only (aligned server-side)",
             (last(bd, 'weights.aux_down'), last(bd, 'weights.aux_up')),
             (R * M * aux_model_bytes(), 0))

    # ---- Vanilla SL: non-federated, so zero weight traffic ------------
    cfg, res = build_run('vanilla_sl', M, N, B, R)
    bd = res.comm_breakdown
    # Vanilla SL never AGGREGATES, but it does MOVE its one client model from
    # client to client (the sequential relay), so its weight traffic is the
    # handover, not zero. One transmission per handover, `num_clients` handovers
    # per round -- see the header note in src/algos/vanilla_sl.py.
    from utils.utils import calculate_load as _load
    relay_per_round = M * _load(res.client_list[0].model)
    check_eq("Vanilla-SL charges its relay handover (NOT zero)",
             last(bd, 'weights.client_relay'), R * relay_per_round)
    check_eq("Vanilla-SL's weight traffic is the relay and nothing else",
             res.comm_load_weights[-1], R * relay_per_round)
    check("Vanilla-SL weight traffic is non-zero",
          res.comm_load_weights[-1] > 0)
    check_eq("Vanilla-SL cut.act_up", last(bd, 'cut.act_up'), R * per_round_act)

    # ---- DSL-Aux: decoupled -- activations up only, non-federated -----
    # arXiv:2601.19261 SIII-B: the server computes dL/dz but "this gradient is
    # not transmitted to the client". Obs. 2: that halves communication versus
    # conventional SL. Both are asserted here against Vanilla-SL, run just above
    # on the identical config, so a regression to plain SL cannot pass silently.
    cfg, res_dsl = build_run('dsl_aux', 1, N, B, R)
    bd = res_dsl.comm_breakdown
    one_round = sum(act_bytes(b) for b in bs_list)
    check_eq("DSL-Aux cut.act_up", last(bd, 'cut.act_up'), R * one_round)
    check_eq("DSL-Aux cut.grad_down is ZERO (no gradient crosses back)",
             last(bd, 'cut.grad_down'), 0)
    check_eq("DSL-Aux charges no weight traffic (non-federated)",
             res_dsl.comm_load_weights[-1], 0)
    # the paper's ~50% claim, stated exactly. Paired against Vanilla-SL on an
    # IDENTICAL config (same client count), so the only difference is the
    # protocol: DSL pays activations + labels, conventional SL pays those plus
    # an equal-sized gradient coming back.
    _, res_sl1 = build_run('vanilla_sl', 1, N, B, R)
    check_eq("Vanilla-SL (1 client) cut.grad_down is the gradient half",
             last(res_sl1.comm_breakdown, 'cut.grad_down'), R * one_round)
    check_eq("DSL-Aux cut total == Vanilla-SL's minus the gradient half",
             res_dsl.comm_load_cut[-1],
             res_sl1.comm_load_cut[-1] - R * one_round)
    check("DSL-Aux cut traffic is ~50% of Vanilla-SL's (paper Obs. 2)",
               abs(res_dsl.comm_load_cut[-1] / res_sl1.comm_load_cut[-1] - 0.5) < 0.02)

    # ---- HOSL: 2Q probe activations + 1 base, 2Q scalars back ---------
    # Eq. 7 is a SYMMETRIC two-point estimate, so each of the Q perturbation
    # vectors costs two forward passes (Alg. 1 L13 and L19 both send an
    # activation), plus one unperturbed upload for the server's first-order
    # pass in Phase 2.
    Q = 2
    cfg, res = build_run('hosl', 1, N, B, R, extra_algo={
        'num_pert': Q, 'zo_mu': 1e-3, 'client_lr': 1e-3, 'server_lr': 1e-3
    })
    bd = res.comm_breakdown
    check_eq("HOSL cut.act_up (1 base + 2Q two-sided probes)",
             last(bd, 'cut.act_up'), R * (1 + 2 * Q) * one_round)
    check_eq("HOSL cut.scalar_down (one scalar per probe forward)",
             last(bd, 'cut.scalar_down'), R * 2 * Q * len(bs_list) * 4)
    check_eq("HOSL charges no weight traffic (non-federated)",
             res.comm_load_weights[-1], 0)

    # ---- HO-SFL: dimension-free aggregation ---------------------------
    cfg, res = build_run('ho_sfl', M, N, B, R, extra_algo={
        'optimizer': 'adamw', 'lr': 1e-3, 'betas': [0.9, 0.999],
        'weight_decay': 1e-4, 'zo_p': 3, 'zo_mu': 1e-3,
        # the tiny stand-in model is not a ResNet-18, so the pretrained loader
        # (which a real run does use -- see src/models/pretrained.py) is off here
        'use_pretrained': False, 'freeze_bn': False,
    })
    bd = res.comm_breakdown
    # matched rounds: every batch of the local epoch, like every other method
    check_eq("HO-SFL cut.act_up (every batch, matched rounds)",
             last(bd, 'cut.act_up'), R * M * sum(act_bytes(b) for b in bs_list))
    check("HO-SFL weight traffic is scalars only, never a weight vector",
          last(bd, 'weights.scalars') == res.comm_load_weights[-1]
          and res.comm_load_weights[-1] > 0,
          f"scalars={last(bd, 'weights.scalars')} total={res.comm_load_weights[-1]}")
    # Dimension-free aggregation: HO-SFL sends (scalar, seed) pairs per update
    # rather than a weight vector, so its aggregation traffic scales with the
    # NUMBER OF UPDATES and not with the model's parameter count. The margin over
    # SplitFedv2 therefore grows with model size -- huge on ResNet-18, modest on
    # this deliberately tiny stand-in model -- so assert the direction, which is
    # the paper's actual claim, rather than a ratio that is an artifact of the
    # test model's size.
    check("HO-SFL weight traffic is below SplitFedv2's",
          res.comm_load_weights[-1] < splitfedv2.comm_load_weights[-1],
          f"ho={res.comm_load_weights[-1]} v2={splitfedv2.comm_load_weights[-1]}")

    return splitfedv2, cse


def test_memory_orderings(splitfedv2, cse):
    print("\n[3] Cross-method memory orderings")

    v2 = splitfedv2.memory_metrics
    cs = cse.memory_metrics

    check("SplitFedv2 holds its autograd graph across the cut (client stalls)",
          v2['client_mem_held_across_cut_mb'] > 0,
          f"held={v2['client_mem_held_across_cut_mb']}")
    check_eq("CSE-FSL holds NOTHING across the cut (decoupled)",
             cs['client_mem_held_across_cut_mb'], 0.0)

    # Per the plan's B0 correction: the aux-model family is NOT cheaper on peak
    # client bytes -- it holds the client graph AND an auxiliary head at once.
    check("CSE-FSL peak client memory is >= SplitFedv2's (aux head + graph)",
          cs['peak_client_mem_mb'] >= v2['peak_client_mem_mb'],
          f"cse={cs['peak_client_mem_mb']:.4f} v2={v2['peak_client_mem_mb']:.4f}")

    check("SplitFedv2 client retains a non-zero activation peak",
          v2['client_act_peak_mem_mb'] > 0)
    check("server-side peak memory is reported",
          v2['peak_server_mem_mb'] > 0)

    # zeroth-order clients run their forward under no_grad -> ~0 retained
    for algo, extra in (
        ('hosl', {'num_pert': 2, 'zo_mu': 1e-3, 'client_lr': 1e-3, 'server_lr': 1e-3}),
        ('mu_splitfed', {'tau': 1, 'zo_mu': 5e-3, 'lr_c': 5e-3, 'lr_s': 1e-2,
                         'lr_g': 0.3, 'use_pretrained': False, 'freeze_bn': False}),
    ):
        _, res = build_run(algo, 2, 8, 4, 1, extra_algo=extra)
        mm = res.memory_metrics
        check_eq(f"{algo}: zeroth-order client retains 0 activation bytes",
                 mm['client_act_peak_mem_mb'], 0.0)
        check(f"{algo}: client memory is still non-zero (params/grads/optimizer)",
              mm['peak_client_mem_mb'] > 0)


# ==============================================================================
# 4. Every method runs end-to-end on the REAL split ResNet-18
# ==============================================================================
# The tiny stand-in model above keeps the closed-form byte counts hand-checkable,
# but it does not exercise the real client/server/auxiliary modules. This pass
# runs all 13 registered methods through the shared loop on the actual
# ResNet-18 split (shallow cut, 2 clients, 8 samples, batch 4, 1 round) and
# asserts each one completes and produces a self-consistent ledger. It is the
# safety net for `main.py`, whose hydra CLI entry point cannot be driven in
# every environment.
ALGO_EXTRAS = {
    'fed_avg':          {},
    'sl_single_server': {},
    'sl_multi_server':  {},
    'vanilla_sl':       {},
    'cse_fsl':          {'server_update_interval': 2},
    'fsl_sage':         {'server_update_interval': 2, 'align_interval': 1},
    'dsl_aux':          {},   # lambda = 0 per the paper; no weight to tune
    'han_locloss':      {},
    'fedsplitx':        {},
    'locfedmix_sl':     {'mixup_alpha': 1.0, 'mixup_partners': 1,
                          'decoder_lr': 1e-3},
    'hosl':             {'num_pert': 2, 'zo_mu': 1e-3, 'client_lr': 1e-3, 'server_lr': 1e-3},
    'ho_sfl':           {'optimizer': 'adamw', 'lr': 1e-3,
                          'betas': [0.9, 0.999], 'weight_decay': 1e-4,
                          'zo_p': 2, 'zo_mu': 1e-3,
                          # exercised separately in test_pretrained_loader
                          'use_pretrained': False, 'freeze_bn': False},
    'mu_splitfed':      {'tau': 1, 'zo_mu': 5e-3, 'lr_c': 5e-3, 'lr_s': 1e-2,
                          'lr_g': 0.3, 'use_pretrained': False,
                          'freeze_bn': False},
}

# methods where no gradient crosses the cut, so cut.grad_down must be zero
NO_GRADIENT_BACK = ('cse_fsl', 'fsl_sage', 'han_locloss', 'fedsplitx',
                    'hosl', 'mu_splitfed', 'fed_avg', 'dsl_aux')
# methods that run the client forward under no_grad, so it retains nothing
ZERO_ORDER_CLIENTS = ('hosl', 'ho_sfl', 'mu_splitfed')
# Zeroth-order methods whose update is applied IN PLACE from a seed-regenerated
# perturbation, so no `.grad` tensor is ever allocated -- HOSL Eq. 15 states
# M_grad = 0 outright. `ho_sfl` is deliberately absent: its own reference uses
# AdamW, whose moment state the algorithm genuinely needs, so its client
# legitimately carries gradient and optimizer memory.
NO_CLIENT_GRAD = ('hosl', 'mu_splitfed')
# non-federated methods, so weight traffic must be zero
# Methods that move NO model at all -- every client owns an independent copy
# that is never shared, averaged, or handed on. `vanilla_sl` is deliberately NOT
# here: it also never aggregates, but it relays one shared model between clients,
# which is a real transfer. "Does it aggregate?" is the wrong question; "does a
# model cross the wire?" is the right one.
NON_FEDERATED = ('dsl_aux', 'hosl')
# MATCHED ROUNDS: no method may skip batches. ho_sfl and mu_splitfed used to do
# real work only at (j,k)==(0,0), which made a "round" mean 1 optimizer step for
# them and 24 for everyone else. The benchmark now defines a round as one local
# epoch for every method, so that gate must not come back -- checked below
# against every registered algorithm's source.


def build_real_run(algo_name, extra_algo, num_clients=2, samples=8,
                    batch_size=4, rounds=1, cut_layers=1):
    from models import (CLIENT_SERVER_MODEL_REGISTRY, aux_models,
                        Client, Server)
    torch.manual_seed(0)
    cfg = make_cfg(algo_name, num_clients, rounds, batch_size, extra_algo)

    cli_ctor, srv_ctor = CLIENT_SERVER_MODEL_REGISTRY['resnet18']
    aux_ctor = aux_models.AUXILIARY_MODEL_REGISTRY['resnet18']

    opt_cfg = OmegaConf.create({
        'optimizer': {'name': 'sgd', 'options': {'lr': 0.01}}
    })
    server = Server(srv_ctor(client_layers=cut_layers, num_classes=NUM_CLASSES),
                    opt_cfg, device='cpu')

    client_cfg = OmegaConf.create({
        'epoch': 1, 'batch_size': batch_size,
        'optimizer': {'name': 'sgd', 'options': {'lr': 0.01}},
    })
    clients = []
    for i in range(num_clients):
        xs = torch.randn(samples, 3, 32, 32)
        ys = torch.randint(0, NUM_CLASSES, (samples,))
        loader = DataLoader(TensorDataset(xs, ys), batch_size=batch_size,
                            shuffle=False)
        c = Client(i, loader, cli_ctor(client_layers=cut_layers), client_cfg,
                   device='cpu')
        c.init_auxiliary(
            aux_ctor(server=server, client_layers=cut_layers,
                     num_classes=NUM_CLASSES, device='cpu',
                     max_dataset_size=64, align_batch_size=16),
            OmegaConf.create({
                'optimizer': {'name': 'adam', 'options': {'lr': 1e-3}},
                'align_epochs': 1,
            })
        )
        clients.append(c)
    for c in clients:
        c.model.load_state_dict(clients[0].model.state_dict())
        c.auxiliary_model.load_state_dict(clients[0].auxiliary_model.state_dict())

    xs = torch.randn(batch_size, 3, 32, 32)
    ys = torch.randint(0, NUM_CLASSES, (batch_size,))
    test_loader = DataLoader(TensorDataset(xs, ys), batch_size=batch_size)

    return _run_fl_algorithm(
        cfg, server, clients, test_loader, None, torch.device('cpu'),
        logger_fn=lambda *a, **kw: None
    )


def test_all_methods_on_real_model():
    print("\n[4] All methods end-to-end on the real split ResNet-18")

    registered = set(ALGORITHM_REGISTRY.keys())
    covered = set(ALGO_EXTRAS.keys())
    check(f"every registered method is covered ({len(covered)}/{len(registered)})",
          registered <= covered,
          f"uncovered: {sorted(registered - covered)}")

    for algo in ALGO_EXTRAS:
        if algo not in ALGORITHM_REGISTRY:
            continue
        try:
            res = build_real_run(algo, ALGO_EXTRAS[algo])
        except Exception as exc:
            check(f"{algo}: completes a round on the real model", False,
                  f"{type(exc).__name__}: {exc}")
            continue

        bd = res.comm_breakdown[-1]
        ok = (
            abs(res.comm_load_cut[-1] + res.comm_load_weights[-1]
                - res.comm_load[-1]) < 1e-6
            and len(res.test_acc if hasattr(res, 'test_acc') else res.accuracy) == 1
        )
        check(f"{algo}: completes a round, ledger balances", ok)

        if algo in ('han_locloss', 'fedsplitx'):
            # Both papers specify a LIGHTWEIGHT auxiliary -- Han et al. report
            # theirs at 0.1-0.6% of full-model parameters. The harness default
            # (`ResNetAuxiliary`) is a whole mirrored ResNet stage, ~19% of a
            # ResNet-18 and three times the client-side model, which inflated
            # both these methods' client memory and their weight-aggregation
            # traffic. Guarded by size rather than by class name so any
            # future substitution is checked too.
            cm = res.memory_metrics
            check(f"{algo}: auxiliary is small beside the client model",
                  cm['client_param_mem_mb'] < 2.0 * _client_only_param_mb(),
                  f"client-side params {cm['client_param_mem_mb']:.4f} MiB "
                  f"vs client model alone {_client_only_param_mb():.4f} MiB")

        if algo == 'fedsplitx':
            # arXiv:2310.14579 Fig. 1 / Sec. 2.1: an auxiliary network at EVERY
            # partition point. A 4-stage ResNet has M = 3 of them (after
            # layer1/2/3; layer4's output is l_{M+1}), split between the sides
            # by the client's depth-level. Asserting the count is what keeps
            # this method from collapsing back to the single-head M=1 case that
            # made it a duplicate of han_locloss.
            tr = res.train_metrics[0]
            n_c, n_s = tr['n_client_aux'][0], tr['n_server_aux'][0]
            check_eq("fedsplitx: M = 3 partition points on ResNet-18",
                     n_c + n_s, 3)
            check("fedsplitx: auxiliary networks on BOTH sides of the cut",
                  n_c >= 1 and n_s >= 1, f"client={n_c} server={n_s}")

        if algo in NO_GRADIENT_BACK:
            check_eq(f"{algo}: no gradient crosses the cut",
                     bd['cut.grad_down'], 0.0)
        if algo in NON_FEDERATED:
            check_eq(f"{algo}: no weight traffic (non-federated)",
                     res.comm_load_weights[-1], 0.0)
        if algo in ZERO_ORDER_CLIENTS:
            check_eq(f"{algo}: zeroth-order client retains 0 activation bytes",
                     res.memory_metrics['client_act_peak_mem_mb'], 0.0)
        if algo in NO_CLIENT_GRAD:
            check_eq(f"{algo}: client allocates NO gradient buffer",
                     res.memory_metrics['client_grad_mem_mb'], 0.0)
            check_eq(f"{algo}: client holds NO optimizer state",
                     res.memory_metrics['client_optim_mem_mb'], 0.0)
        check(f"{algo}: reports non-zero client memory",
              res.memory_metrics['peak_client_mem_mb'] > 0)

        check_every_batch_trains(algo, res)
        if algo == 'fed_avg':
            # fed_avg merges the server model into every client's model, so
            # there is no separate server host to charge -- 0 is correct here
            check_eq("fed_avg: no separate server host to charge",
                     res.memory_metrics['peak_server_mem_mb'], 0.0)
        else:
            check(f"{algo}: reports non-zero server memory",
                  res.memory_metrics['peak_server_mem_mb'] > 0)


def check_every_batch_trains(algo, res):
    '''The matched-rounds invariant: one round is one local epoch, the same for
    every method, so no client_step may skip batches.

    This is the guard on the benchmark's central fairness claim. When ho_sfl and
    mu_splitfed no-op'd on all but the round's first batch, an equal round budget
    silently handed them 1/24 of everyone else's optimizer steps -- which the
    sweep then compensated for with a 24x round multiplier, making Comm-total and
    Latency incomparable across rows. Checked structurally against the source, so
    the gate cannot be reintroduced without this failing.
    '''
    alg = ALGORITHM_REGISTRY[algo]
    src = inspect.getsource(alg.client_step)
    gate = re.search(r'if \(j,\s*k\)\s*!=\s*\(0,\s*0\)', src)
    check(f"{algo}: trains on every batch (no one-batch-per-round gate)",
          gate is None,
          "client_step still early-returns on batches other than (0,0)")
    # one training-metric entry per round, i.e. the round really did run.
    # Only meaningful for methods that report a 'loss' key at all -- some report
    # their own named losses instead.
    tr = res.train_metrics[0]
    if 'loss' in tr:
        check(f"{algo}: one training-metric entry per round",
              len(tr['loss']) == len(res.accuracy),
              f"loss entries={len(tr['loss'])} rounds={len(res.accuracy)}")


# ==============================================================================
# 5. The pretrained zeroth-order init (Part D)
# ==============================================================================
def test_pretrained_loader():
    print("\n[5] Pretrained ResNet-18 init for the zeroth-order clients")
    from models import CLIENT_SERVER_MODEL_REGISTRY
    from models.pretrained import (
        pretrained_resnet18_state_dict, load_pretrained_client,
        load_pretrained_server, freeze_batchnorm_affine
    )
    try:
        state = pretrained_resnet18_state_dict()
    except Exception as exc:
        print(f"  SKIP  pretrained weights unavailable ({type(exc).__name__})")
        return

    cli_ctor, srv_ctor = CLIENT_SERVER_MODEL_REGISTRY['resnet18']
    for cl in (1, 2, 3):
        c = cli_ctor(client_layers=cl)
        srv = srv_ctor(client_layers=cl, num_classes=NUM_CLASSES)
        load_pretrained_client(c, state)
        load_pretrained_server(srv, state)
        check(f"cut={cl}: client conv1 matches torchvision's pretrained weights",
              torch.allclose(c.conv1.weight, state['conv1.weight']))
        freeze_batchnorm_affine(c)
        frozen = [n for n, p in c.named_parameters() if not p.requires_grad]
        check(f"cut={cl}: BatchNorm affine params are frozen (not perturbed)",
              len(frozen) > 0 and all('bn' in n or 'downsample.1' in n
                                      for n in frozen),
              f"frozen={frozen}")
        out = srv(c(torch.randn(2, 3, 32, 32)))
        check_eq(f"cut={cl}: split model still produces the right output shape",
                 tuple(out.shape), (2, NUM_CLASSES))



# ==============================================================================
def test_no_cloned_implementations():
    """No two methods may share an implementation.

    THIS IS THE CHECK THAT WAS MISSING. Every other assertion in this file is
    per-method: it can confirm that `fedsplitx` charges the bytes `fedsplitx`
    is supposed to charge, and still not notice that `fedsplitx.py` was a
    byte-for-byte copy of `han_locloss.py` with the class name changed. Two
    identical implementations produce two identical benchmark rows, which reads
    as an independent corroboration when it is really one number printed twice.

    Compared on source text rather than on measured output, because two genuinely
    distinct methods may legitimately coincide on a toy model while differing on
    the real backbone -- source identity is unambiguous either way.
    """
    print("\n[6] No two methods share an implementation")

    def normalise(fn):
        src = inspect.getsource(fn)
        src = re.sub(r'#[^\n]*', '', src)              # strip comments
        src = re.sub(r'\s+', ' ', src)                  # collapse whitespace
        return src.strip()

    bodies = {}
    for name, cls in sorted(ALGORITHM_REGISTRY.items()):
        step = getattr(cls, 'client_step', None)
        if step is None:
            continue
        try:
            bodies[name] = normalise(step)
        except (OSError, TypeError):
            continue

    seen = {}
    clones = []
    for name, body in bodies.items():
        # a class name appearing inside the body would mask an otherwise exact
        # clone, so compare with every registry key neutralised
        key = body
        for other in bodies:
            key = key.replace(other, '<ALGO>')
        if key in seen:
            clones.append((seen[key], name))
        else:
            seen[key] = name

    check("no two algorithms share a client_step implementation",
          not clones,
          "; ".join(f"{a} == {b}" for a, b in clones))
    print(f"  ({len(bodies)} algorithms compared)")



def _client_only_param_mb(_cache={}):
    '''Params of the client-side model alone, for the auxiliary-size guard.'''
    if 'v' not in _cache:
        from models import CLIENT_SERVER_MODEL_REGISTRY
        cli_ctor, _ = CLIENT_SERVER_MODEL_REGISTRY['resnet18']
        m = cli_ctor(client_layers=1)
        _cache['v'] = sum(p.numel() * p.element_size()
                          for p in m.parameters()) / (1024 ** 2)
    return _cache['v']


# ==============================================================================
def test_system_peak_memory():
    """The simultaneity peak: does conventional SL really hold both sides at once?

    `peak_client_mem_mb` and `peak_server_mem_mb` are maxed per side
    INDEPENDENTLY, so neither can express that conventional SL keeps the client's
    activations alive WHILE the server runs, whereas a decoupled method frees
    them first. That difference is the whole of DSL-Aux's memory claim
    (arXiv:2601.19261 Obs. 3, "up to 58%"), and it is what `peak_system_live_mb`
    measures. Asserted as a RELATION between two methods on an identical config,
    not against an absolute figure, so it holds on any model size.
    """
    print("\n[7] System (simultaneity) peak memory")

    sl = build_real_run('vanilla_sl', ALGO_EXTRAS['vanilla_sl'], cut_layers=2)
    dsl = build_real_run('dsl_aux', ALGO_EXTRAS['dsl_aux'], cut_layers=2)

    sl_m, dsl_m = sl.memory_metrics, dsl.memory_metrics

    for name, m in (('vanilla_sl', sl_m), ('dsl_aux', dsl_m)):
        check(f"{name}: reports a non-zero system act peak",
              m['peak_system_live_mb'] > 0)
        # A simultaneity peak must be at least as large as either side alone.
        # (It can exceed the sum of the two *activation* peaks, because it also
        # counts held buffers -- the input batch, the uploaded activation, the
        # downloaded gradient -- which the activation-only metrics exclude.)
        check(f"{name}: system peak >= each side's activation peak",
              m['peak_system_live_mb'] >= max(m['client_act_peak_mem_mb'],
                                              m['server_act_peak_mem_mb']) - 1e-6,
              f"system={m['peak_system_live_mb']} "
              f"client={m['client_act_peak_mem_mb']} "
              f"server={m['server_act_peak_mem_mb']}")

    # Conventional SL holds the client graph across the cut, so its simultaneity
    # peak must exceed the decoupled method's on the same config. This is the
    # assertion that would fail if dsl_aux ever regressed to holding its graph.
    check("vanilla_sl's system peak EXCEEDS dsl_aux's (the decoupling saving)",
          sl_m['peak_system_live_mb'] > dsl_m['peak_system_live_mb'],
          f"sl={sl_m['peak_system_live_mb']:.4f} "
          f"dsl={dsl_m['peak_system_live_mb']:.4f}")

    # ...and the mechanism: SL holds a client graph across the cut, DSL does not
    check_eq("dsl_aux holds nothing across the cut",
             dsl_m['client_mem_held_across_cut_mb'], 0.0)
    check("vanilla_sl DOES hold across the cut",
          sl_m['client_mem_held_across_cut_mb'] > 0)

    # On CUDA the allocator gives an independent ground truth: whatever we
    # attribute must be bytes it actually handed out. No-op on CPU.
    if torch.cuda.is_available():
        for name, m in (('vanilla_sl', sl_m), ('dsl_aux', dsl_m)):
            sides = max(m['peak_client_mem_mb'], m['peak_server_mem_mb'])
            check(f"{name}: system peak dominates either side alone",
                  m['peak_system_mem_mb'] >= sides - 1e-6,
                  f"system={m['peak_system_mem_mb']} max(side)={sides}")

    saving = 100.0 * (1 - dsl_m['peak_system_live_mb'] / sl_m['peak_system_live_mb'])
    print(f"  (system live peak: vanilla_sl {sl_m['peak_system_live_mb']:.4f} MiB, "
          f"dsl_aux {dsl_m['peak_system_live_mb']:.4f} MiB -> {saving:.1f}% saving)")



# ==============================================================================
def test_held_across_cut_reflects_the_algorithm():
    """`held_across_cut` must measure a STALL, not statement order.

    The metric samples client-owned autograd bytes at the instant a server phase
    opens. That makes it sensitive to where a method happens to put its local
    backward -- two methods here used to report a large hold purely because
    their backward sat below the server block, with no data dependency forcing
    it there. These assertions pin the corrected ordering.
    """
    print("\n[8] held-across-cut reflects the algorithm, not statement order")

    sl = build_real_run('vanilla_sl', ALGO_EXTRAS['vanilla_sl'])
    sl_held = sl.memory_metrics['client_mem_held_across_cut_mb']

    # FSL-SAGE depends on the server ONLY on an alignment iteration, where the
    # surrogate is refreshed server-side and the client's gradient must come
    # from the refreshed copy. Its backward is now hoisted above the server
    # phase on every other iteration.
    #
    # Note what this does and does not change. `held_across_cut` is a MAX over
    # the run, and `t % align_interval == 0` is true at t = 0 for any interval,
    # so every run contains an alignment iteration and the max is unchanged: the
    # backbone, exactly what vanilla SL holds. What the hoist changes is how
    # OFTEN the client stalls -- once per `align_interval` rounds instead of
    # every iteration -- which a max cannot express. The assertion here is
    # therefore the honest one: FSL-SAGE never holds MORE than the backbone.
    fs = build_real_run('fsl_sage', ALGO_EXTRAS['fsl_sage'])
    check("fsl_sage never holds more than the backbone across the cut",
          fs.memory_metrics['client_mem_held_across_cut_mb'] <= sl_held + 1e-6,
          f"fsl_sage={fs.memory_metrics['client_mem_held_across_cut_mb']} "
          f"vanilla={sl_held}")

    # LocFedMix-SL genuinely waits for the server's gradient (Eq. 4), so it
    # holds its backbone graph -- but ONLY the backbone. Its Infopro decoder
    # subgraph has no reason to survive the round trip, and on ResNet-18 that
    # subgraph is dominated by a single interpolate back to input resolution.
    # Holding exactly what vanilla SL holds is the precise statement of that.
    lfm = build_real_run('locfedmix_sl', ALGO_EXTRAS['locfedmix_sl'])
    check("locfedmix_sl holds only the backbone, same as vanilla SL",
          abs(lfm.memory_metrics['client_mem_held_across_cut_mb']
              - sl_held) < 1e-6,
          f"locfedmix={lfm.memory_metrics['client_mem_held_across_cut_mb']} "
          f"vanilla={sl_held}")


# ==============================================================================
def main():
    print("=" * 74)
    print("Measurement-layer checks (communication + per-side memory)")
    print("=" * 74)
    test_memory_meter()
    v2, cse = test_comm_closed_form()
    test_memory_orderings(v2, cse)
    test_all_methods_on_real_model()
    test_pretrained_loader()
    test_no_cloned_implementations()
    test_system_peak_memory()
    test_held_across_cut_reflects_the_algorithm()

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"FAILED {len(FAILURES)}/{CHECKS} checks:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"All {CHECKS} checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
