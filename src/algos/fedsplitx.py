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
#   L6    server copy w^s_k <- w^s[d_k:] for that client     -> server_down
#   L7    s^c_k, y_k <- GetSmashedData(w^c_k, D_k)           client_step, 1st batch
#   L18     forward ALL of D_k through the DOWNLOADED model,  -> ONE act_up + ONE
#           send the smashed data and labels to the server       labels_up per round
#   L8    Client_Update (L13-16): E epochs; loss             client_step, every batch
#         sum_{i=1..d_k} CE(a_i(f[:i](x)), y), no weights;
#         updates the client model and a_[1..d_k]
#   L9-11 server trains w^s_k for E epochs on s^c_k with     client_step, every batch
#         sum_{i=d_k+1..M+1} CE (output only at level 3)
#   L12   heteroavg: each parameter is averaged over the      aggregate()
#         clients that hold it (plain mean); the same for    -> client_up, aux_up,
#         the server-side models                                 server_up
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
#   - the server-copy averaging (L12) is charged as Server-model FedAvg, one
#     upload (L12) and one download (L6) of x_s per sampled client per round
#   - random init, the same as every method
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


def _mean_into(target, models):
    '''L12 heteroavg with every client at one depth level: the plain
    (unweighted) mean of the given models, written into `target`. Integer
    buffers (BatchNorm's num_batches_tracked) are taken from the first.'''
    states = [m.state_dict() for m in models]
    new = {}
    for key, v in states[0].items():
        if torch.is_floating_point(v):
            new[key] = sum(s[key] for s in states) / len(states)
        else:
            new[key] = v.clone()
    target.load_state_dict(new)


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

        # every client starts from the same client model; one server copy per
        # client (L6), re-extracted from the global w^s whenever it is sampled
        for c in self.clients[1:]:
            c.model.load_state_dict(self.clients[0].model.state_dict())
        self.servers = [copy.deepcopy(self.server) for _ in self.clients]

        self._client_taps = _taps(self.clients[0].model)
        self._server_taps = _taps(self.server.model)
        self.__build_heads()

        # the global models the Fed server keeps between rounds
        self.aggregated_client = copy.deepcopy(self.clients[0].model)
        self.aggregated_aux = copy.deepcopy(self.client_aux[0])
        self.aggregated_server = copy.deepcopy(self.server.model)
        self.aggregated_server_aux = copy.deepcopy(self.server_aux[0])

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

        # per sampled client: the round's uploaded smashed data (L7/L18)
        self._upload = {}

    # -- Sec. 2.1 / Appendix A.1: an auxiliary network at each point ---------
    def __build_heads(self):
        client_model, server_model = self.clients[0].model, self.server.model
        sample = next(iter(self.test_loader))[0][:2].to(self.device)
        was = (client_model.training, server_model.training)
        client_model.eval(); server_model.eval()
        with torch.no_grad():
            c_feats, z = _forward_taps(client_model, self._client_taps, sample)
            if not self._client_taps:
                c_feats = [z]                 # M = 1: one head at the cut
            s_feats, out = _forward_taps(server_model, self._server_taps, z)
        client_model.train(was[0]); server_model.train(was[1])

        num_classes = int(out.shape[1])
        dim = lambda f: int(f.shape[1]) if f.dim() == 4 else int(f.flatten(1).shape[1])
        c_dims, s_dims = [dim(f) for f in c_feats], [dim(f) for f in s_feats]

        first = nn.ModuleList([_AuxHead(d, num_classes) for d in c_dims]).to(self.device)
        self.client_aux = [first] + [copy.deepcopy(first) for _ in self.clients[1:]]
        first = nn.ModuleList([_AuxHead(d, num_classes) for d in s_dims]).to(self.device)
        self.server_aux = [first] + [copy.deepcopy(first) for _ in self.clients[1:]]

    @staticmethod
    def __server_side_bytes(model, heads):
        '''One transfer of a server-side model w^s_k: its layers plus its aux
        networks a_[d_k+1..M] (none at the middle cut).'''
        return calculate_load(model) + calculate_load(heads)

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

        if j == 0 and k == 0:
            # L5: download the global client-side model (+ its aux networks)
            client.load_state_dict(self.aggregated_client.state_dict())
            self.client_aux[i].load_state_dict(self.aggregated_aux.state_dict())
            self.charge_weights_model(self.aggregated_client, 'client', 'down')
            self.charge_weights_model(self.aggregated_aux, 'aux', 'down')
            # L6: the server copy for this client, from the global w^s
            server.load_state_dict(self.aggregated_server.state_dict())
            self.server_aux[i].load_state_dict(self.aggregated_server_aux.state_dict())
            self.ledger.charge_weights('server_down', self.__server_side_bytes(
                self.aggregated_server, self.aggregated_server_aux))
            # L7 / L18: smashed data + labels of all of D_k, one upload
            with self.phase('client', i):
                s, ys, offsets = self.__get_smashed_data(i)
            self.charge_cut_activation(s)
            self.charge_cut_labels(ys)
            self._upload[i] = (s, ys, offsets)

        # --- L8, Client_Update (L13-16): sum of CE over a_[1..d_k] ----------
        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            self.client_aux_optims[i].zero_grad()
            c_feats, z = _forward_taps(client, self._client_taps, x)
            if not self._client_taps:
                c_feats = [z]
            logps = [h(f) for h, f in zip(self.client_aux[i], c_feats)]
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
            s_feats, out = _forward_taps(server, self._server_taps, sx)
            server_loss = self.criterion(out, sy)
            for h, f in zip(self.server_aux[i], s_feats):
                server_loss = server_loss + self.criterion(h(f), sy)
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
        _mean_into(self.aggregated_client, [self.clients[i].model for i in P])
        _mean_into(self.aggregated_aux, [self.client_aux[i] for i in P])
        for i in P:
            self.charge_weights_model(self.clients[i].model, 'client', 'up')
            self.charge_weights_model(self.client_aux[i], 'aux', 'up')
        t1 = time.time()
        _mean_into(self.aggregated_server, [self.servers[i].model for i in P])
        if len(self.aggregated_server_aux):
            _mean_into(self.aggregated_server_aux, [self.server_aux[i] for i in P])
        for i in P:
            self.ledger.charge_weights('server_up', self.__server_side_bytes(
                self.servers[i].model, self.server_aux[i]))
        self._upload = {}                      # the round's uploads are spent
        return {'client_agg_compute_time': t1 - t0,
                'server_agg_compute_time': time.time() - t1}

    # -- Sec. 3: inference = ensemble of every head + the output ------------------
    def __heads_and_output(self, x):
        c_feats, z = _forward_taps(self.aggregated_client, self._client_taps, x)
        if not self._client_taps:
            c_feats = [z]
        s_feats, out = _forward_taps(self.aggregated_server, self._server_taps, z)
        client_logps = [h(f) for h, f in zip(self.aggregated_aux, c_feats)]
        server_logps = [h(f) for h, f in zip(self.aggregated_server_aux, s_feats)]
        return client_logps, server_logps, out

    def full_model(self, x):
        c, s, out = self.__heads_and_output(x)
        return _ensemble(c + s + [out])

    def evaluate(self):
        '''The benchmark accuracy is the full-model ensemble (every head plus
        the output). The client-side ensemble (a_[1..d_k] only), which the
        paper also reports, is logged alongside.'''
        self.eval_mode()
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
    # (+ their aux networks when the server holds any); the round's upload is
    # held as server working memory. Copies for the other sampled clients are
    # listed separately.
    def client_side_modules(self, i):
        return [self.clients[i].model, self.client_aux[i]]

    def client_side_optimizers(self, i):
        return [self.clients[i].optimizer, self.client_aux_optims[i]]

    def _trained_copy(self):
        '''Index of a server copy that has actually been trained (a sampled
        client), so its gradients are allocated and counted.'''
        return self.sampled[0]

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
        params, grads = module_static_bytes(self.servers[0].model, self.server_aux[0])
        return [{
            'item': f'server copies of the other {others} sampled clients (params + grads)',
            'mb': others * (params + grads) / 1024 ** 2,
            'note': 'Alg.1 L6: one server-side copy per sampled client in a round',
        }]

    def register_meter_params(self):
        # every client's and every server copy's parameters, so no parameter
        # view (Linear saves weight.t()) is ever counted as an activation
        mods = [self.server.model, self.aggregated_client, self.aggregated_aux,
                self.aggregated_server, self.aggregated_server_aux]
        for i in range(len(self.clients)):
            mods += [self.clients[i].model, self.client_aux[i],
                     self.servers[i].model, self.server_aux[i]]
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
