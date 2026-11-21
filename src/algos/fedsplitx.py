# ------------------------------------------------------------------------------
# FedSplitX: Federated Split Learning for Computationally-Constrained
# Heterogeneous Clients (Shin, Ahn, Kang, Kang -- arXiv:2310.14579).
#
# PROVENANCE. The authors released no code, so this is an AI-ASSISTED
# REIMPLEMENTATION written from the paper alone (Algorithm 1 p.3, Sec. 2.1-2.3,
# Sec. 3, Appendix A.1). Label it a reimplementation wherever it is reported.
# Every place the paper leaves a choice open is marked NOT SPECIFIED below,
# with the choice made here.
#
# THE MODEL (Sec. 2.1, Fig. 1, Appendix A.1). The network is cut at M partition
# points; an auxiliary network a_[i] sits at every one of them. For ResNet-18
# the paper's partition points are after the stem (conv1+bn+relu+maxpool),
# after layer1 and after layer2 (M = 3); a client at depth level d_k holds the
# layers up to point d_k plus a_[1..d_k], the server the rest plus a_[d_k+1..M]
# and the model output. The benchmark's middle cut (client = stem..layer2) is
# depth level 3 (Table 3: 689.4K client params; ours 683,072 + BN buffers), so
#   client: stem, layer1, layer2 + a_[1] (stem), a_[2] (layer1), a_[3] (layer2)
#   server: layer3, layer4, fc   -- no auxiliary network, only the output
# cut=shallow is depth level 2 (client a_[1], a_[2]; server a_[3] after
# layer2). cut=deep is not a paper partition point: the client keeps a_[1..3]
# and the extra layer3 has no head. A backbone without these submodules (the
# test model) falls back to one head at the cut (M = 1).
#
# ALGORITHM 1, line by line, and where it is below:
#   L3    P_t <- random set of clients, a fraction C        begin_round()
#   L5    client downloads the global client-side model     client_step, 1st batch
#         (printed as w^s[:d_k]; a typo for w^c)            -> client_down, aux_down
#   L6    server copy w^s_k <- w^s[d_k:] for that client     (on the main server: no traffic)
#   L7    s^c_k, y_k <- GetSmashedData(w^c_k, D_k)           client_step, 1st batch
#   L18     forward ALL of D_k through the DOWNLOADED model,  -> ONE act_up + ONE
#           send the smashed data and labels to the server       labels_up per round
#   L8    Client_Update (L13-16): E epochs; loss             client_step, every batch
#         sum_{i=1..d_k} CE(a_i(f[:i](x)), y), no weights;
#         updates the client model and a_[1..d_k]
#   L9-11 server trains w^s_k for E epochs on s^c_k with     client_step, every batch
#         sum_{i=d_k+1..M+1} CE (output only at level 3)
#   L12   heteroavg: each parameter is averaged over the      aggregate()
#         clients that hold it (plain mean); the same for    -> client_up, aux_up
#         the server-side models (on the main server: no traffic)
# No gradient ever crosses the cut (p.4, B.2, Fig. 1). Inference (Sec. 3): the
# ensemble of every auxiliary network and the model output; the paper reports
# the client-side accuracy as well, logged here each round.
#
# NOT SPECIFIED in the paper, and the choice made here:
#   - auxiliary network architecture: global average pool + one Linear layer
#     (GAPLinearHead); 2,590 params in total at the middle cut
#   - how the auxiliary networks are aggregated: averaged with the client
#     model; their transfers are charged on separate rows (aux_up/aux_down)
#   - loss: cross-entropy (the harness's NLLLoss on log-probabilities)
#   - ensemble rule: mean of the heads' probabilities
#   - learning rate / momentum / weight decay: SGD lr 0.01, no momentum, no
#     weight decay (the paper only says SGD); no LR schedule
#   - BatchNorm during GetSmashedData: batch statistics (train mode, as in
#     Client_Update), with the running statistics restored afterwards, so
#     Client_Update starts from exactly the downloaded model
#   - which clients are averaged: the sampled clients P_t only
#   - how the server walks its uploaded data: one step per client batch, on
#     the matching slice of the upload, E = 1
#
# Benchmark settings (decisions recorded with the user):
#   - every client at the benchmark's middle cut (depth level 3), so heteroavg
#     is a plain average; the paper's heterogeneous depth levels are not used
#   - client_fraction C = 1.0: ALL 10 clients train every round, so FedSplitX
#     sees the same data per round as every other method and the L12 average
#     is over 10 models. (The paper's C = 0.1 with our 10 clients would be ONE
#     client per round: 1/10 of the data, and an average of a single model.)
#     Set algorithm.client_fraction=C to sample S = max(1, round(C * N)).
#   - batch 256 (the batch every method uses; the paper uses 64)
#   - communication counts only what crosses a network. Sec. 2.3: "The
#     aggregation of server-side model parameters is performed at the main
#     server", which also holds the global w^s and every copy w^s_k, so making
#     a copy (L6) and averaging the copies (L12) never leave that machine and
#     are not charged (the same rule as MU-SplitFed's Split Server and
#     SplitFedv1's main server). The client-side models and aux networks are
#     averaged at the fed server, across the network: charged (W1-W4). The
#     copies still count as server memory (server_side_extra_memory).
#   - random init, the same as every method
#
# OPTIONAL heterogeneous depth levels (OFF by default; depth_level_mix or
# depth_levels in the yaml). Each client k gets its own level d_k (1 = stem,
# 2 = + layer1, 3 = + layer2) and holds w^c[: d_k] + a_[1..d_k]; its server copy
# holds w^s[d_k :] + a_[d_k+1..M]. The global w^c is the largest client model
# (level 3) and w^s the largest server model (level 1: layer1 -> fc), so layer1,
# layer2, a2, a3 exist in a client-trained and a server-trained version
# (Sec. 2.1). heteroavg (L12) is _heteroavg_into: each parameter averaged over
# the models that hold it. Evaluation (Sec. 3, Table 1) runs the level-m split
# w^c[: m] + w^s[m :] for m = 1, 2, 3: a client-side and a full-model accuracy
# each; the returned accuracy is the mean of the 3 full-model accuracies. The
# breakdown adds a per-level table (extra_report). With the mode off, every
# client holds every parameter and all of this reduces to the single-level code.
# ------------------------------------------------------------------------------
import time
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from algos import register_algorithm, FLAlgorithm
from utils.memory import module_static_bytes
from utils.utils import calculate_load
from models import CLIENT_SERVER_MODEL_REGISTRY
from models.aux_models.simple_conv import GAPLinearHead as _AuxHead

# Appendix A.1: ResNet-18 partition points, in order (stem output = maxpool)
_PARTITION_POINTS = ('maxpool', 'layer1', 'layer2')


def _taps(model):
    '''Names of the paper's partition points present in this model. Names,
    not module objects: each client forwards its own copy, so the hook has to
    go on the copy actually being run.'''
    return [n for n in _PARTITION_POINTS
            if isinstance(getattr(model, n, None), nn.Module)]


def _forward_taps(model, tap_names, x):
    '''Forward `model`, capturing the output of each tapped submodule.'''
    feats = []
    handles = [getattr(model, n).register_forward_hook(
                   lambda _m, _i, o: feats.append(o))
               for n in tap_names]
    try:
        out = model(x)
    finally:
        for h in handles:
            h.remove()
    return feats, out


def _named_feats(model, x, cut_fallback):
    '''One forward of `model`: (names, features, output). `names` are the
    partition points present in the model, which key the aux heads. A client
    model with none of them (the tiny test model) uses its own output under
    the name 'cut' (M = 1); a server model with none has no heads.'''
    names = _taps(model)
    feats, out = _forward_taps(model, names, x)
    if not names and cut_fallback:
        names, feats = ['cut'], [out]
    return names, feats, out


def _load_subset(model, global_state):
    '''L5 / L6: load into `model` the part of a global model it holds, i.e.
    its depth level's cut: w^c[: d_k] for a client, w^s[d_k :] for a server
    copy, and the aux heads at its partition points. With every client at
    one depth level this is the whole global model.'''
    own = model.state_dict()
    missing = [k for k in own if k not in global_state]
    assert not missing, f'not in the global model: {missing[:3]}'
    model.load_state_dict({k: global_state[k] for k in own})


def _heteroavg_into(target, models):
    '''L12 heteroavg (Sec. 2.3): every parameter of the global model
    `target` is averaged over the models that HOLD it (the same parameter
    name), so each segment between partition points is averaged only over the
    clients (or server copies) that have it. With every client at one depth
    level every model holds every name, and this is the plain mean. Integer
    buffers (BatchNorm's num_batches_tracked) come from the first holder; a
    parameter no model holds keeps its value.'''
    states = [m.state_dict() for m in models]
    new = {}
    for key, v in target.state_dict().items():
        holders = [s[key] for s in states if key in s]
        if not holders:
            new[key] = v
        elif torch.is_floating_point(v):
            new[key] = sum(holders) / len(holders)
        else:
            new[key] = holders[0].clone()
    target.load_state_dict(new)


def _levels_from_mix(mix, n):
    '''Depth level (1..M) of each of n clients from the fraction of clients
    at each level, assigned in order. Counts use largest-remainder rounding
    so they add up to n: [0.32, 0.32, 0.36] gives 10 -> 3 / 3 / 4 and
    50 -> 16 / 16 / 18 (the paper's mix).'''
    total = float(sum(mix))
    quotas = [f / total * n for f in mix]
    counts = [int(q) for q in quotas]
    spare = n - sum(counts)
    for j in sorted(range(len(mix)), key=lambda j: quotas[j] - counts[j],
                    reverse=True)[:spare]:
        counts[j] += 1
    return [lvl for lvl, c in enumerate(counts, start=1) for _ in range(c)]


def _ensemble(logps):
    '''Mean of the heads' probabilities, returned as a log-probability (the
    harness's criterion is NLLLoss): log(mean_i exp(logp_i)).'''
    stacked = torch.stack(logps, dim=0)
    return torch.logsumexp(stacked, dim=0) - np.log(stacked.shape[0])


# ------------------------------------------------------------------------------
@register_algorithm("fedsplitx")
class FedSplitX(FLAlgorithm):
    aggregated_server: torch.nn.Module

    def __init__(self, *args, **kwargs):
        super(FedSplitX, self).__init__(*args, **kwargs)
        n = len(self.clients)

        # L3: fraction C of the clients per round. Default 1.0 = every client
        # (benchmark setting); the paper (p.5) uses C = 10% of 50 clients.
        self.C = float(self.cfg.get('client_fraction', 1.0))
        self.S = max(1, min(n, int(round(self.C * n))))
        self.sampled = list(range(n))

        # Optional heterogeneous depth levels (off by default). depth_levels
        # gives each client's level explicitly; otherwise depth_level_mix gives
        # the fraction of clients at levels 1, 2, 3. Off = every client at the
        # harness's cut (level 3 at cut=middle), the benchmark setting.
        levels = self.cfg.get('depth_levels') or None
        mix = self.cfg.get('depth_level_mix') or None
        if levels is None and mix is not None:
            levels = _levels_from_mix([float(f) for f in mix], n)
        self.hetero = levels is not None
        self.levels = [int(l) for l in levels] if self.hetero else None
        if self.hetero:
            assert len(self.levels) == n and set(self.levels) <= {1, 2, 3}, \
                f'one depth level in 1..3 per client, got {self.levels}'
            assert _taps(self.clients[0].model) == list(_PARTITION_POINTS), \
                'heterogeneous depth levels need ResNet-18 at cut=middle (level 3)'

        # every client starts from the same client model; one server copy per
        # client (L6), re-extracted from the global w^s whenever it is sampled
        for c in self.clients[1:]:
            c.model.load_state_dict(self.clients[0].model.state_dict())
        self.servers = [copy.deepcopy(self.server) for _ in self.clients]

        self.__build_heads()

        # the global models the Fed server keeps between rounds
        self.aggregated_client = copy.deepcopy(self.clients[0].model)
        self.aggregated_aux = copy.deepcopy(self.client_aux[0])
        self.aggregated_server = copy.deepcopy(self.server.model)
        self.aggregated_server_aux = copy.deepcopy(self.server_aux[0])
        if self.hetero:
            self.__build_levels()

        # SGD (p.5); lr / momentum / weight decay NOT SPECIFIED -> config
        sgd = lambda params: torch.optim.SGD(
            params, lr=float(self.cfg.get('lr', 0.01)),
            momentum=float(self.cfg.get('momentum', 0.0)),
            weight_decay=float(self.cfg.get('weight_decay', 0.0)))
        for c, heads in zip(self.clients, self.client_aux):
            c.optimizer = sgd(c.model.parameters())
            c.lr_scheduler = None
        self.client_aux_optims = [sgd(h.parameters()) for h in self.client_aux]
        for s, heads in zip(self.servers, self.server_aux):
            s.optimizer = sgd(s.model.parameters())
            s.lr_scheduler = None
        self.server_aux_optims = [sgd(h.parameters()) if len(h) else None
                                  for h in self.server_aux]

        # per sampled client: the round's uploaded smashed data (L7/L18), and
        # its size in bytes (activations + labels), kept for the memory report
        self._upload = {}
        self._upload_bytes = {}

    # -- Sec. 2.1 / Appendix A.1: an auxiliary network at each point ---------
    def __build_heads(self):
        client_model, server_model = self.clients[0].model, self.server.model
        sample = next(iter(self.test_loader))[0][:2].to(self.device)
        was = (client_model.training, server_model.training)
        client_model.eval(); server_model.eval()
        with torch.no_grad():
            c_names, c_feats, z = _named_feats(client_model, sample, True)
            s_names, s_feats, out = _named_feats(server_model, z, False)
        client_model.train(was[0]); server_model.train(was[1])

        num_classes = int(out.shape[1])
        dim = lambda f: int(f.shape[1]) if f.dim() == 4 else int(f.flatten(1).shape[1])

        # one head per partition point, keyed by the point's name (maxpool ->
        # a1, layer1 -> a2, layer2 -> a3), so a client or server copy at any
        # depth level holds exactly the heads at its own points
        first = nn.ModuleDict({nm: _AuxHead(dim(f), num_classes)
                               for nm, f in zip(c_names, c_feats)}).to(self.device)
        self.client_aux = [first] + [copy.deepcopy(first) for _ in self.clients[1:]]
        first = nn.ModuleDict({nm: _AuxHead(dim(f), num_classes)
                               for nm, f in zip(s_names, s_feats)}).to(self.device)
        self.server_aux = [first] + [copy.deepcopy(first) for _ in self.clients[1:]]

    # -- optional heterogeneous depth levels (Sec. 2.1, App. A.1) ---------------
    def __build_levels(self):
        '''Give each client a client model, aux heads and server copy at its
        own depth level d_k (client_layers = d_k - 1: level 1 = stem only).

        The global models are the largest of each kind (Sec. 2.1): w^c = a
        level-3 client model + a1, a2, a3 (already built), and w^s = a level-1
        server model (layer1 -> fc) + a2, a3. They start as one initial full
        model: w^s's layer1 / layer2 and heads a2 / a3 start from w^c's, and
        layer3 -> fc from the harness's server model.'''
        cli_ctor, srv_ctor = CLIENT_SERVER_MODEL_REGISTRY['resnet18']
        n_ch = self.clients[0].model.conv1.in_channels
        n_cls = self.server.model.fc.out_features
        dev = self.device
        mk_client = lambda lvl: cli_ctor(n_channels=n_ch, client_layers=lvl - 1).to(dev)
        mk_server = lambda lvl: srv_ctor(client_layers=lvl - 1, num_classes=n_cls).to(dev)

        ws = mk_server(1)
        wc_state = self.aggregated_client.state_dict()
        _load_subset(ws, {**self.server.model.state_dict(),
                          **{k: v for k, v in wc_state.items()
                             if k.startswith(('layer1.', 'layer2.'))}})
        self.aggregated_server = ws
        self.aggregated_server_aux = nn.ModuleDict(
            {nm: copy.deepcopy(self.aggregated_aux[nm]) for nm in _taps(ws)})

        for i, lvl in enumerate(self.levels):
            if lvl != 3:                          # level 3 = the harness's model
                self.clients[i].model = mk_client(lvl)
            _load_subset(self.clients[i].model, wc_state)
            self.servers[i].model = mk_server(lvl)
            _load_subset(self.servers[i].model, ws.state_dict())
            self.client_aux[i] = nn.ModuleDict(
                {nm: copy.deepcopy(self.aggregated_aux[nm])
                 for nm in _taps(self.clients[i].model)})
            self.server_aux[i] = nn.ModuleDict(
                {nm: copy.deepcopy(self.aggregated_server_aux[nm])
                 for nm in _taps(self.servers[i].model)})

        # Sec. 3 / Table 1: one evaluation model per level, w^c[:m] + w^s[m:]
        self._eval_models = {m: (mk_client(m), mk_server(m)) for m in (1, 2, 3)}
        self.level_acc = {}
        self.level_bytes = {m: {'upload': 0, 'client': 0, 'aux': 0} for m in (1, 2, 3)}

    def _count(self, i, key, nbytes):
        '''Per-level communication (heterogeneous mode only).'''
        if self.hetero:
            self.level_bytes[self.levels[i]][key] += nbytes

    # -- L3: sample P_t ---------------------------------------------------------
    def begin_round(self, t):
        self.sampled = sorted(int(i) for i in np.random.choice(
            len(self.clients), size=self.S, replace=False))

    def participates(self, t, i):
        return i in self.sampled

    def sampled_per_round(self):
        return self.S

    # -- L7 / L18: GetSmashedData -----------------------------------------------
    def __get_smashed_data(self, i):
        '''Forward ALL of client i's data through the model it just
        downloaded, before Client_Update, and send it in one upload.'''
        model = self.clients[i].model
        bn_state = {k: v.clone() for k, v in model.state_dict().items()}
        smashed, labels = [], []
        with torch.no_grad():
            for x, y in self.clients[i].train_loader:
                x = x.to(self.device).double() if self.use_64bit \
                    else x.to(self.device).float()
                smashed.append(model(x))
                labels.append(y.to(self.device).long())
        # BatchNorm used batch statistics above; put the running statistics
        # back so Client_Update starts from exactly the downloaded model
        model.load_state_dict(bn_state)
        s, y = torch.cat(smashed), torch.cat(labels)
        offsets = np.cumsum([0] + [t.shape[0] for t in labels]).tolist()
        return s, y, offsets

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it
        client, server = self.clients[i].model, self.servers[i].model
        mets = {}

        # Alg.1 L4 runs the sampled clients in parallel; here they run one
        # after another. The RESULT is the same: each client starts from the
        # global models, which only change in aggregate() (L12), and shares no
        # model, server copy or optimizer state with another client (checked:
        # running the clients in reverse order gives identical weights).
        # MEMORY is measured per client: one server copy and one upload at a
        # time. The other clients' copies and uploads, which a parallel server
        # holds at the same time, are listed in server_side_extra_memory().
        if j == 0 and k == 0:
            # L5: download the global client-side model cut at this client's
            # depth level, w^c[: d_k] (+ its aux networks a_[1..d_k])
            _load_subset(client, self.aggregated_client.state_dict())
            _load_subset(self.client_aux[i], self.aggregated_aux.state_dict())
            self.charge_weights_model(client, 'client', 'down')
            self.charge_weights_model(self.client_aux[i], 'aux', 'down')
            self._count(i, 'client', calculate_load(client))
            self._count(i, 'aux', calculate_load(self.client_aux[i]))
            # L6: the server copy for this client, w^s[d_k :] (+ a_[d_k+1..M])
            _load_subset(server, self.aggregated_server.state_dict())
            _load_subset(self.server_aux[i], self.aggregated_server_aux.state_dict())
            # (on the main server, which holds w^s and every copy: not charged)
            # L7 / L18: smashed data + labels of all of D_k, one upload
            with self.phase('client', i):
                s, ys, offsets = self.__get_smashed_data(i)
            self.charge_cut_activation(s)
            self.charge_cut_labels(ys)
            self._upload[i] = (s, ys, offsets)
            self._upload_bytes[i] = (s.numel() * s.element_size()
                                     + ys.numel() * ys.element_size())
            self._count(i, 'upload', self._upload_bytes[i])

        # --- L8, Client_Update (L13-16): sum of CE over a_[1..d_k] ----------
        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            self.client_aux_optims[i].zero_grad()
            c_names, c_feats, z = _named_feats(client, x, True)
            logps = [self.client_aux[i][nm](f) for nm, f in zip(c_names, c_feats)]
            client_loss = sum(self.criterion(lp, y) for lp in logps)
            client_loss.backward()
            self.clients[i].optimizer.step()
            self.client_aux_optims[i].step()
            with torch.no_grad():
                client_acc = _ensemble(logps).argmax(1).eq(y).float().mean().item()
        del z, c_feats, logps

        # --- L9-11: server trains its copy on the uploaded smashed data -----
        s, ys, offsets = self._upload[i]
        n_chunks = len(offsets) - 1
        c = k % n_chunks
        sx, sy = s[offsets[c]:offsets[c + 1]], ys[offsets[c]:offsets[c + 1]]
        with self.phase('server', i):
            self.hold('server', s, ys, i=i)       # the round's upload stays resident
            opt_aux = self.server_aux_optims[i]
            self.servers[i].optimizer.zero_grad()
            if opt_aux is not None:
                opt_aux.zero_grad()
            s_names, s_feats, out = _named_feats(server, sx, False)
            server_loss = self.criterion(out, sy)
            for nm, f in zip(s_names, s_feats):
                server_loss = server_loss + self.criterion(self.server_aux[i][nm](f), sy)
            server_loss.backward()
            self.servers[i].optimizer.step()
            if opt_aux is not None:
                opt_aux.step()
            with torch.no_grad():
                server_acc = out.argmax(1).eq(sy).float().mean().item()

        mets.update({
            'l_loss': float(client_loss.item()), 'l_acc': client_acc,
            'g_loss': float(server_loss.item()), 'acc': server_acc,
            'loss': float(server_loss.item()),
            'n_client_aux': len(self.client_aux[i]),
            'n_server_aux': len(self.server_aux[i]),
        })
        return mets

    # -- L12: heteroavg over P_t -----------------------------------------------
    def aggregate(self):
        P = self.sampled
        t0 = time.time()
        # heteroavg at the fed server: each parameter of w^c and of the client
        # aux heads over the clients that hold it
        _heteroavg_into(self.aggregated_client, [self.clients[i].model for i in P])
        _heteroavg_into(self.aggregated_aux, [self.client_aux[i] for i in P])
        for i in P:
            self.charge_weights_model(self.clients[i].model, 'client', 'up')
            self.charge_weights_model(self.client_aux[i], 'aux', 'up')
            self._count(i, 'client', calculate_load(self.clients[i].model))
            self._count(i, 'aux', calculate_load(self.client_aux[i]))
        t1 = time.time()
        # heteroavg at the main server: each parameter of w^s and of the
        # server aux heads over the server copies that hold it
        _heteroavg_into(self.aggregated_server, [self.servers[i].model for i in P])
        if len(self.aggregated_server_aux):
            _heteroavg_into(self.aggregated_server_aux, [self.server_aux[i] for i in P])
        # (Sec. 2.3: averaged at the main server, which already holds every
        # copy: no network traffic, not charged)
        self._upload = {}                      # the round's uploads are spent
        return {'client_agg_compute_time': t1 - t0,
                'server_agg_compute_time': time.time() - t1}

    # -- Sec. 3: inference = ensemble of every head + the output ------------------
    def __heads_and_output(self, x, level=None):
        '''The global heads' predictions and the output for a batch. level m
        (heterogeneous mode): the level-m split, w^c[: m] + w^s[m :].'''
        cm, sm = (self.aggregated_client, self.aggregated_server) if level is None \
            else self._eval_models[level]
        c_names, c_feats, z = _named_feats(cm, x, True)
        s_names, s_feats, out = _named_feats(sm, z, False)
        client_logps = [self.aggregated_aux[nm](f) for nm, f in zip(c_names, c_feats)]
        server_logps = [self.aggregated_server_aux[nm](f) for nm, f in zip(s_names, s_feats)]
        return client_logps, server_logps, out

    def full_model(self, x):
        c, s, out = self.__heads_and_output(x, 3 if self.hetero else None)
        return _ensemble(c + s + [out])

    def evaluate(self):
        '''The benchmark accuracy is the full-model ensemble (every head plus
        the output). The client-side ensemble (a_[1..d_k] only), which the
        paper also reports, is logged alongside.'''
        self.eval_mode()
        if self.hetero:
            return self.__evaluate_levels()
        correct, client_correct, losses = 0, 0, []
        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc="Test Batch", unit='batch',
                             leave=False):
                x = x.to(self.device).double() if self.use_64bit \
                    else x.to(self.device).float()
                y = y.to(self.device).long()
                c, s, out = self.__heads_and_output(x)
                full = _ensemble(c + s + [out])
                losses.append(self.criterion(full, y).item())
                correct += full.argmax(1).eq(y).sum().item()
                client_correct += _ensemble(c).argmax(1).eq(y).sum().item()
        n = len(self.test_loader.dataset)
        self.client_side_acc = client_correct / n
        logging.info(f' > FedSplitX client-side ensemble acc (a_[1..{len(self.aggregated_aux)}]): '
                     f'{100. * self.client_side_acc:.2f}%, full-model ensemble acc: '
                     f'{100. * correct / n:.2f}%, sampled clients: {self.sampled}')
        return correct / n, sum(losses) / len(losses)

    def __evaluate_levels(self):
        '''Heterogeneous mode (Sec. 3, Table 1): for each depth level m, the
        level-m split w^c[: m] + w^s[m :], with the client-side ensemble (a1..a_m)
        and the full-model ensemble (every head + the output): 6 accuracies.
        Returned: the mean of the 3 full-model accuracies and losses.'''
        for m, (cm, sm) in self._eval_models.items():
            _load_subset(cm, self.aggregated_client.state_dict())
            _load_subset(sm, self.aggregated_server.state_dict())
            cm.eval(); sm.eval()
        hits = {m: [0, 0] for m in self._eval_models}
        losses = {m: [] for m in self._eval_models}
        with torch.no_grad():
            for x, y in tqdm(self.test_loader, desc="Test Batch", unit='batch',
                             leave=False):
                x = x.to(self.device).double() if self.use_64bit \
                    else x.to(self.device).float()
                y = y.to(self.device).long()
                for m in self._eval_models:
                    c, s, out = self.__heads_and_output(x, m)
                    full = _ensemble(c + s + [out])
                    losses[m].append(self.criterion(full, y).item())
                    hits[m][0] += _ensemble(c).argmax(1).eq(y).sum().item()
                    hits[m][1] += full.argmax(1).eq(y).sum().item()
        n = len(self.test_loader.dataset)
        self.level_acc = {m: (h[0] / n, h[1] / n) for m, h in hits.items()}
        logging.info(' > FedSplitX per depth level (client-side / full model): ' + ', '.join(
            f'level {m}: {100 * a[0]:.2f}% / {100 * a[1]:.2f}%' for m, a in self.level_acc.items()))
        full_acc = float(np.mean([a[1] for a in self.level_acc.values()]))
        full_loss = float(np.mean([np.mean(l) for l in losses.values()]))
        return full_acc, full_loss

    def extra_report(self):
        '''Per-depth-level table for the breakdown (heterogeneous mode only).'''
        if not self.hetero:
            return None
        MB = 1024 ** 2
        per_client = self.memory_report()['peak_client_mem_mb_per_client']
        rows, data = [], {}
        for m in (1, 2, 3):
            idx = [i for i, l in enumerate(self.levels) if l == m]
            if not idx:
                continue
            params, grads = module_static_bytes(self.servers[idx[0]].model,
                                                self.server_aux[idx[0]])
            acc = self.level_acc.get(m, (float('nan'), float('nan')))
            b = self.level_bytes[m]
            d = {'clients': len(idx), 'acc_client_side': acc[0], 'acc_full': acc[1],
                 'client_mem_mb': max(per_client[i] for i in idx),
                 'server_copy_mb': (params + grads) / MB,
                 'upload_mb': b['upload'] / MB, 'client_model_mb': b['client'] / MB,
                 'aux_mb': b['aux'] / MB}
            data[m] = d
            rows.append([m, d['clients'], f"{100 * acc[0]:.2f}% / {100 * acc[1]:.2f}%",
                         f"{d['client_mem_mb']:.2f}", f"{d['server_copy_mb']:.2f}",
                         f"{d['upload_mb']:.2f}", f"{d['client_model_mb']:.2f}",
                         f"{d['aux_mb']:.2f}"])
        mean_full = float(np.mean([d['acc_full'] for d in data.values()]))
        return {
            'title': 'Per depth level (FedSplitX heterogeneous mode)',
            'columns': ['Level', 'Clients', 'Accuracy: client side / full model',
                        'Client mem (max)', 'Server copy (params + grads)',
                        'Upload', 'Client model up + down', 'Aux heads up + down'],
            'rows': rows,
            'note': f'levels {self.levels}; headline accuracy = mean of the full-model '
                    f'accuracies = {100 * mean_full:.2f}%; MiB',
            'data': data,
        }

    def special_models_train_mode(self, t):
        for m in [self.aggregated_server, self.aggregated_aux,
                  self.aggregated_server_aux] + [s.model for s in self.servers] \
                 + self.client_aux + self.server_aux:
            m.train()

    def special_models_eval_mode(self):
        for m in [self.aggregated_server, self.aggregated_aux,
                  self.aggregated_server_aux] + [s.model for s in self.servers] \
                 + self.client_aux + self.server_aux:
            m.eval()

    # -- memory partition ---------------------------------------------------------
    # Client: its model + a_[1..d_k] (SGD without momentum: no optimizer state).
    # Server per device: the global x_s + the copy being trained for one client
    # (+ their aux networks when the server holds any); that client's upload is
    # held as server working memory. The other sampled clients' copies and
    # uploads, held at the same time in Alg.1's parallel round, are listed
    # separately (server_side_extra_memory).
    def client_side_modules(self, i):
        return [self.clients[i].model, self.client_aux[i]]

    def client_side_optimizers(self, i):
        return [self.clients[i].optimizer, self.client_aux_optims[i]]

    def _trained_copy(self):
        '''Index of a server copy that has actually been trained (a sampled
        client), so its gradients are allocated and counted: the largest one
        (heterogeneous mode: a level-1 copy), the first when all are equal.'''
        return max(self.sampled, key=lambda i: sum(
            p.numel() for p in self.servers[i].model.parameters()))

    def server_side_modules(self):
        c = self._trained_copy()
        mods = [self.aggregated_server, self.servers[c].model]
        if len(self.aggregated_server_aux):
            mods += [self.aggregated_server_aux, self.server_aux[c]]
        return mods

    def server_side_optimizers(self):
        c = self._trained_copy()
        return [o for o in (self.servers[c].optimizer, self.server_aux_optims[c])
                if o is not None]

    def server_side_extra_memory(self):
        others = self.S - 1
        if others <= 0:
            return []
        c = self._trained_copy()
        copies = sum(sum(module_static_bytes(self.servers[i].model, self.server_aux[i]))
                     for i in self.sampled if i != c)
        lines = [{
            'item': f'server copies of the other {others} sampled clients (params + grads)',
            'mb': copies / 1024 ** 2,
            'note': 'Alg.1 L6: one server-side copy per sampled client in a round',
        }]
        # their uploads (activations + labels of all their data, L7/L18): the
        # per-device line already counts one upload, so list the other S - 1
        sizes = sorted(self._upload_bytes[i] for i in self.sampled
                       if i in self._upload_bytes)
        if len(sizes) > 1:
            lines.append({
                'item': f'uploads of the other {len(sizes) - 1} sampled clients '
                        f'(smashed data + labels of all their images)',
                'mb': (sum(sizes) - sizes[-1]) / 1024 ** 2,
                'note': 'Alg.1 L7/L18: each sampled client\'s upload is held '
                        'until its server copy has trained on it',
            })
        return lines

    def register_meter_params(self):
        # every client's and every server copy's parameters, so no parameter
        # view (Linear saves weight.t()) is ever counted as an activation
        mods = [self.server.model, self.aggregated_client, self.aggregated_aux,
                self.aggregated_server, self.aggregated_server_aux]
        for i in range(len(self.clients)):
            mods += [self.clients[i].model, self.client_aux[i],
                     self.servers[i].model, self.server_aux[i]]
        if self.hetero:
            mods += [m for pair in self._eval_models.values() for m in pair]
        self.meter.register_params(*mods)

    def _module_label(self, m, i=None):
        if m is self.aggregated_server_aux:
            return 'global server-side aux networks'
        if any(m is h for h in self.server_aux):
            return 'server-side aux networks of the copy being trained'
        if i is not None and m is self.client_aux[i]:
            return f'client aux networks a_[1..{len(m)}] (GAP + Linear)'
        return super()._module_label(m, i)

# ------------------------------------------------------------------------------
