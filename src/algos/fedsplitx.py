# ------------------------------------------------------------------------------
# FedSplitX: Federated Split Learning for Computationally-Constrained
# Heterogeneous Clients (Shin, Ahn, Kang, Kang -- arXiv:2310.14579).
# AI-ASSISTED NO-CODE REIMPLEMENTATION per CLAUDE.md's rule for methods without
# a released implementation; not validated against the paper's own reported
# CIFAR-10/CIFAR-100 numbers.
#
# WHAT DEFINES THIS METHOD (Sec. 2.1-2.2, Fig. 1): the model is cut at M
# PARTITION POINTS, and an auxiliary network a_[i] hangs off EVERY one of them.
# A client at depth-level d_k holds a_[1:d_k] on its side; the server holds
# a_[d_k+1:M] plus the model's own output head. Both sides then train on a
# COLLABORATIVE LOSS -- a SUM of losses over all the intermediate logits they
# own, not a single loss at the cut:
#
#   client (Sec. 2.2): F^c_k = 1/|D_k| * sum_j sum_{i=1..d_k} l_c(l^k_{i,j}, y_j)
#                      with l^k_i = a_[i]( f^c_k[:i](w^c; x) )
#   server (Sec. 2.2): F^s_k = 1/|D_k| * sum_j sum_{i=d_k+1..M+1} l_s(l^k_i, y_j)
#                      with l^k_{M+1} = f^k(w^s; s^c_k), the real output
#
# No gradient ever crosses the cut: "clients can compute local-loss and perform
# backward propagation using a_[m] without waiting for gradients from the
# server" (Sec. 2.2). Inference uses the ENSEMBLE of all auxiliary networks
# (Sec. 3), not the final head alone.
#
# HOW IT MAPS ONTO THIS HARNESS. CLAUDE.md's protocol runs every method at ONE
# shared cut, so every client here sits at the SAME depth-level -- FedSplitX's
# client HETEROGENEITY (clients clustered by compute into M depth-levels) is the
# one part of the paper deliberately not exercised, and `heteroavg` (Sec. 2.3)
# correspondingly degenerates to plain FedAvg because there is no depth-level
# subset to exclude. Everything else IS exercised, because the partition points
# are a property of the architecture, not of the client population: for a
# 4-stage ResNet the partition points are the stage boundaries, so M = 3
# (after layer1, layer2, layer3), with layer4's output supplying l_{M+1}. The
# harness's `cut` then selects the depth-level shared by all clients:
#
#   cut=shallow (client_layers=1): client holds a_[1];        server a_[2], a_[3]
#   cut=middle  (client_layers=2): client holds a_[1], a_[2]; server a_[3]
#   cut=deep    (client_layers=3): client holds a_[1..3];     server none
#
# An EARLIER REVISION of this file was a byte-for-byte copy of han_locloss.py:
# it used a single auxiliary head at the cut, which is the M=1 degenerate case
# and is genuinely indistinguishable from Han et al. That produced two identical
# rows in the benchmark. The multi-partition collaborative loss above is what
# actually separates the two methods, and test/check_accounting.py now refuses
# to let any two algorithms share a client_step implementation.
#
# Auxiliary network architecture: the paper does not specify one, so this uses
# the standard lightweight auxiliary classifier -- global average pool to a
# vector, then a single linear layer to the class logits. Kept deliberately
# small, since the paper's premise is that auxiliary networks are cheap enough
# to sit on a compute-constrained client.
# ------------------------------------------------------------------------------
import time
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from algos import register_algorithm, aggregate_models, FLAlgorithm
from models import config_optimizer
from models.aux_models.simple_conv import GAPLinearHead as _AuxHead

_STAGE_NAMES = ('layer1', 'layer2', 'layer3', 'layer4')


# One FedSplitX auxiliary network a_[i] per partition point. Shared with
# han_locloss so there is a single definition of "the paper-sized auxiliary
# head" in the harness rather than one per method.


def _stage_modules(model):
    '''The architecture's partition points, in order.

    Returns the NAMES of the ResNet stage submodules present on this side of
    the cut -- names, not module objects, because every client holds its own
    copy of the model and a hook must be attached to the copy actually being
    forwarded. Empty for a backbone with no stage structure, in which case the
    caller falls back to the cut itself as the single partition point (the
    paper's M=1 case).
    '''
    return [n for n in _STAGE_NAMES
            if isinstance(getattr(model, n, None), nn.Module)]


# ------------------------------------------------------------------------------
@register_algorithm("fedsplitx")
class FedSplitX(FLAlgorithm):
    aggregated_auxiliary: torch.nn.Module
    aggregated_server: torch.nn.Module

    def __init__(self, *args, **kwargs):
        super(FedSplitX, self).__init__(*args, **kwargs)
        self.servers = [copy.deepcopy(self.server) for _ in self.clients]
        self.__build_partition_heads()
        self.aggregated_auxiliary = self.client_aux[0]
        self.aggregated_server = self.servers[0].model
        self.aggregated_server_aux = self.server_aux[0]

    # -- Sec. 2.1: an auxiliary network at every partition point -----------
    def __build_partition_heads(self):
        client_model = self.clients[0].model
        server_model = self.servers[0].model

        # partition points owned by each side. The client's stages each end at
        # a partition point (the last one IS the cut); on the server every stage
        # but the final one does, since the final stage feeds the real output.
        self._client_taps = _stage_modules(client_model)
        server_stages = _stage_modules(server_model)
        self._server_taps = server_stages[:-1] if server_stages else []

        sample = next(iter(self.test_loader))[0][:1].to(self.device)
        c_feats, s_feats, was = self.__probe(client_model, server_model, sample)

        num_classes = int(self._probe_out.shape[1])
        c_dims = [self.__feat_dim(f) for f in c_feats]
        s_dims = [self.__feat_dim(f) for f in s_feats]

        self.client_aux, self.server_aux = [], []
        for c in self.clients:
            heads = nn.ModuleList([_AuxHead(d, num_classes) for d in c_dims])
            self.client_aux.append(heads.to(self.device))
        for _ in self.servers:
            heads = nn.ModuleList([_AuxHead(d, num_classes) for d in s_dims])
            self.server_aux.append(heads.to(self.device))

        # every client starts from the same auxiliary networks
        for heads in self.client_aux[1:]:
            heads.load_state_dict(self.client_aux[0].state_dict())
        for heads in self.server_aux[1:]:
            heads.load_state_dict(self.server_aux[0].state_dict())

        self.client_aux_optims = [
            config_optimizer(h.parameters(), c.optimizer_options)
            for h, c in zip(self.client_aux, self.clients)
        ] if c_dims else []
        self.server_aux_optims = [
            config_optimizer(h.parameters(), s.optimizer_options)
            for h, s in zip(self.server_aux, self.servers)
        ] if s_dims else []

    def __probe(self, client_model, server_model, sample):
        '''One eval-mode forward to learn each partition point's width.'''
        was = (client_model.training, server_model.training)
        client_model.eval(); server_model.eval()
        try:
            with torch.no_grad():
                c_feats, z = self.__forward_taps(client_model, self._client_taps, sample)
                if not self._client_taps:
                    c_feats = [z]
                s_feats, out = self.__forward_taps(server_model, self._server_taps, z)
                self._probe_out = out
        finally:
            client_model.train(was[0]); server_model.train(was[1])
        return c_feats, s_feats, was

    @staticmethod
    def __feat_dim(f):
        return int(f.shape[1]) if f.dim() == 4 else int(f.flatten(1).shape[1])

    @staticmethod
    def __forward_taps(model, tap_names, x):
        '''Forward `model`, capturing the output of each tapped stage.

        Hooks are attached to THIS model's submodules by name on every call.
        Caching the module objects instead would silently tap client 0's copy
        while client i>0 was the one actually running, yielding no features.
        '''
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

    def full_model(self, x):
        '''Sec. 3: inference uses the ENSEMBLE of all auxiliary networks.

        Each head emits log-probabilities, so the ensemble is the log of the
        mean probability -- logsumexp over the heads minus log(n) -- which keeps
        the result a valid log-prob for the harness's NLLLoss.
        '''
        c_feats, z = self.__forward_taps(
            self.aggregated_client, self._client_taps, x
        )
        if not self._client_taps:
            c_feats = [z]
        s_feats, out = self.__forward_taps(
            self.aggregated_server, self._server_taps, z
        )

        logits = [h(f) for h, f in zip(self.aggregated_auxiliary, c_feats)]
        logits += [h(f) for h, f in zip(self.aggregated_server_aux, s_feats)]
        logits.append(out)
        stacked = torch.stack(logits, dim=0)
        return torch.logsumexp(stacked, dim=0) - torch.log(
            torch.tensor(float(stacked.shape[0]), device=stacked.device)
        )

    def special_models_train_mode(self, t):
        if t > 0:
            self.aggregated_server.train()
            self.aggregated_auxiliary.train()
        for h in self.client_aux + self.server_aux:
            h.train()

    def special_models_eval_mode(self):
        self.aggregated_server.eval()
        self.aggregated_auxiliary.eval()
        self.aggregated_server_aux.eval()
        for h in self.client_aux + self.server_aux:
            h.eval()

    # the client owns its model plus the a_[1:d_k] auxiliary networks; the
    # server host owns one model replica and a_[d_k+1:M] per client
    def client_side_modules(self, i):
        return [self.clients[i].model, self.client_aux[i]]

    def client_side_optimizers(self, i):
        opt = [self.clients[i].optimizer]
        if self.client_aux_optims:
            opt.append(self.client_aux_optims[i])
        return opt

    def server_side_modules(self):
        return [s.model for s in self.servers] + list(self.server_aux)

    def server_side_optimizers(self):
        return [s.optimizer for s in self.servers] + list(self.server_aux_optims)

    # -- Algorithm 1 -------------------------------------------------------
    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        # --- Client_Update (Alg. 1 L13-16): collaborative loss over the
        # client's own d_k partition points. Purely local -- never waits on
        # the server, and its graph is released before the server runs.
        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            if self.client_aux_optims:
                self.client_aux_optims[i].zero_grad()

            c_feats, z = self.__forward_taps(
                self.clients[i].model, self._client_taps, x
            )
            if not self._client_taps:
                c_feats = [z]

            client_loss = 0.0
            last_logits = None
            for head, f in zip(self.client_aux[i], c_feats):
                last_logits = head(f)
                client_loss = client_loss + self.criterion(last_logits, y)

            client_loss.backward()
            self.clients[i].optimizer.step()
            if self.client_aux_optims:
                self.client_aux_optims[i].step()

            with torch.no_grad():
                _, predicted = torch.max(last_logits.data, 1)
                client_correct = predicted.eq(y.view_as(predicted)).sum().item()

        # --- Alg. 1 L7-8: upload smashed data. One way -- no gradient returns.
        smashed_data = z.detach()
        self.charge_cut_activation(smashed_data)
        self.charge_cut_labels(y)     # the server evaluates its own losses

        # --- Alg. 1 L9-11: server's collaborative loss over a_[d_k+1:M] plus
        # the model's own output l_{M+1}.
        with self.phase('server', i):
            self.servers[i].optimizer.zero_grad()
            if self.server_aux_optims:
                self.server_aux_optims[i].zero_grad()

            s_feats, server_out = self.__forward_taps(
                self.servers[i].model, self._server_taps, smashed_data
            )
            server_loss = self.criterion(server_out, y)
            for head, f in zip(self.server_aux[i], s_feats):
                server_loss = server_loss + self.criterion(head(f), y)

            server_loss.backward()
            self.servers[i].optimizer.step()
            if self.server_aux_optims:
                self.server_aux_optims[i].step()

            with torch.no_grad():
                _, predicted = torch.max(server_out.data, 1)
                server_correct = predicted.eq(y.view_as(predicted)).sum().item()

        return {
            'l_loss': float(client_loss.item()),
            'l_acc': client_correct / y.size(dim=0),
            'g_loss': float(server_loss.item()),
            'acc': server_correct / y.size(dim=0),
            'loss': float(server_loss.item()),
            'n_client_aux': len(self.client_aux[i]),
            'n_server_aux': len(self.server_aux[i]),
        }

    def aggregate(self):
        # heteroavg (Sec. 2.3) degenerates to plain FedAvg here: every client
        # shares one depth-level under this harness's single-cut protocol, so
        # there is no parameter subset to exclude from an average.
        ret_dict = self.aggregate_clients()

        t0 = time.time()
        self.aggregated_auxiliary = aggregate_models(
            list(self.client_aux), self.agg_factor, self.device
        )
        agg_aux_weights = self.aggregated_auxiliary.state_dict()
        for heads in self.client_aux:
            heads.load_state_dict(agg_aux_weights)
            self.charge_weights_roundtrip(self.aggregated_auxiliary, 'aux')
        ret_dict['auxiliary_agg_compute_time'] = time.time() - t0

        t0 = time.time()
        self.aggregated_server = aggregate_models(
            [s.model for s in self.servers], self.agg_factor, self.device
        )
        agg_server_weights = self.aggregated_server.state_dict()
        for s in self.servers:
            s.model.load_state_dict(agg_server_weights)
            self.charge_weights_roundtrip(self.aggregated_server, 'server')

        # the server-side auxiliary networks are averaged on the main server
        # alongside the server models ("the main server averages the parameters
        # of the server-side models in the same way", Sec. 2.3)
        self.aggregated_server_aux = aggregate_models(
            list(self.server_aux), self.agg_factor, self.device
        )
        if len(self.aggregated_server_aux) > 0:
            agg_saux_weights = self.aggregated_server_aux.state_dict()
            for heads in self.server_aux:
                heads.load_state_dict(agg_saux_weights)
                self.charge_weights_roundtrip(self.aggregated_server_aux, 'server')
        ret_dict['server_agg_compute_time'] = time.time() - t0

        return ret_dict

# ------------------------------------------------------------------------------
