# ------------------------------------------------------------------------------
# LocFedMix-SL: Localize, Federate, and Mix for Improved Scalability,
# Convergence, and Latency in Split Learning (Oh, Park, Vepakomma, Baek,
# Raskar, Bennis, Kim -- ACM WWW 2022). AI-ASSISTED NO-CODE REIMPLEMENTATION
# per CLAUDE.md's rule for methods without a working released implementation;
# not validated against the paper's own reported accuracy/scalability numbers.
#
# Read directly from the paper (Sec. 2.2-3.3, Eq. 1, 4-6, 8-10). Five
# component techniques, combined:
#   1) Smashed-data FP + 2) Global gradient BP (parallel SL baseline, Eq. 1,
#      4): every client forwards its own batch, the server computes a real
#      per-client loss L_i = CE(server(s_i), y_i) and backpropagates a real
#      grad_at_cut to that client -- an ordinary SplitFed-style real gradient
#      DOES cross the cut both ways here. Each client's own optimizer step
#      uses ONLY this real gradient plus the local regularizer below (Eq. 8)
#      -- the mixup loss (next) is explicitly stated in the paper as
#      "detached from the lower model segment", i.e. it never reaches any
#      client.
#   3) Smashed-data Mixup (Eq. 5-6): after every client's real smashed data
#      has been collected this round, pairs of different clients' smashed
#      data are linearly interpolated (s_mix = lam*s_i + (1-lam)*s_j, lam ~
#      Beta(alpha,alpha)) and a mixup loss CE(server(s_mix), mixed one-hot
#      target) is computed -- purely as ADDITIONAL server-side training
#      signal, at no extra communication cost (the server already has every
#      client's real smashed data from step 1-2; nothing new is uploaded).
#      The paper's Eq. 6 combines every client's real loss L_i AND every
#      mixup loss L~_i into ONE dataset-size-weighted (delta_i) sum before a
#      single server update per round -- this port accumulates the server's
#      gradient across all clients' real losses AND all mixup pairs (scaled
#      by this harness's own `agg_factor`, which already serves the same
#      dataset-size-weighting role as the paper's delta_i) and steps the
#      server optimizer once per round, in `aggregate()`.
#   4) Regularized Local Gradient (Eq. 8, "Infopro"): each client also trains
#      a small per-client DECODER network that reconstructs the raw input
#      from its own smashed data, minimizing an L2 reconstruction loss L_hat_i
#      = ||x - h(s_i)||^2 -- this maximizes mutual information between the
#      smashed data and the raw input (a regularizer, NOT a classifier head,
#      unlike this harness's other auxiliary-loss methods). The decoder
#      updates from L_hat_i alone; the client's own model updates from
#      L_hat_i AND the real task loss's gradient combined (Eq. 8).
#   5) Local (Federated) Model Averaging (Eq. 9-10): the client-side models
#      AND the per-client decoders are FedAvg'd. The paper explicitly studies
#      a tunable averaging interval T_c as a communication/accuracy
#      trade-off (Fig. 6) -- this port always averages every round (the
#      T_c=1 endpoint of that trade-off), a disclosed simplification.
#
# The decoder is a lightweight stand-in (a couple of conv layers plus
# `F.interpolate` back to the raw input's spatial size) for the paper's own
# "Infopro" reconstruction network, whose exact architecture wasn't
# specified in the sections read -- sufficient for the "runs + logs
# correctly, non-chance accuracy on MNIST" bar. `mixup_partners` (n_s in the
# paper) defaults to 1 (pairwise mixup, matching the paper's own worked
# two-client example) rather than exploring larger n_s as the paper's own
# ablation does.
# ------------------------------------------------------------------------------
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algos import register_algorithm, FLAlgorithm
from utils.utils import calculate_load
from models.resnet import _STAGE_OUT_PLANES

# ------------------------------------------------------------------------------
class _ReconstructionDecoder(nn.Module):
    '''Lightweight stand-in for the paper's "Infopro" decoder h(.): maps
    smashed-data activations back to a reconstruction of the raw input, for
    the L2 mutual-information regularizer in Eq. 8.'''

    def __init__(self, in_channels, out_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, smashed_data, target_size):
        x = F.interpolate(
            smashed_data, size=target_size, mode='bilinear', align_corners=False
        )
        return self.net(x)

# ------------------------------------------------------------------------------
@register_algorithm("locfedmix_sl")
class LocFedMixSL(FLAlgorithm):

    def __init__(self, *args, **kwargs):
        super(LocFedMixSL, self).__init__(*args, **kwargs)
        self.mixup_alpha = self.cfg.mixup_alpha
        self.mixup_partners = self.cfg.mixup_partners

        client_layers = getattr(self.clients[0].model, 'client_layers', 2)
        in_channels = _STAGE_OUT_PLANES[client_layers - 1]

        decoder0 = _ReconstructionDecoder(in_channels).to(self.device)
        self.decoders = [decoder0] + [
            copy.deepcopy(decoder0) for _ in self.clients[1:]
        ]
        self.decoder_optimizers = [
            torch.optim.Adam(d.parameters(), lr=self.cfg.decoder_lr)
            for d in self.decoders
        ]
        self.aggregated_decoder = decoder0

        self._round_buf = []

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        self._round_buf = []
        self.server.optimizer.zero_grad()
        for d in self.decoders:
            d.train()

    def special_models_eval_mode(self):
        for d in self.decoders:
            d.eval()

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        self.clients[i].optimizer.zero_grad()
        self.decoder_optimizers[i].zero_grad()

        # (1) real client forward -- graph stays attached so the local
        # regularizer's gradient and the real injected grad_at_cut both
        # accumulate into this client model's own parameters (Eq. 8).
        splitting_output = self.clients[i].model(x)

        # (4) Infopro reconstruction regularizer: decoder updates from this
        # loss alone; the client model's update below adds this loss's
        # gradient on top of the real task gradient.
        recon = self.decoders[i](splitting_output, x.shape[-2:])
        recon_loss = F.mse_loss(recon, x)
        recon_loss.backward(retain_graph=True)

        # (2) real per-client loss/gradient (Eq. 1, 4) -- a real leaf sent to
        # the server, real backprop, real grad_at_cut returned to the client.
        smashed_data = splitting_output.detach().requires_grad_(True)
        self.comm_load_cut += smashed_data.numel() * smashed_data.element_size()

        delta_i = self.agg_factor[i]
        out = self.server.model(smashed_data)
        loss = self.criterion(out, y)
        (delta_i * loss).backward()
        # server optimizer is NOT stepped here -- gradients accumulate across
        # every client (and every mixup pair) this round, stepped once in
        # aggregate() per Eq. 6's single dataset-size-weighted server update.

        grad_at_cut = smashed_data.grad.clone().detach()
        self.comm_load_cut += grad_at_cut.numel() * grad_at_cut.element_size()
        splitting_output.backward(grad_at_cut)

        self.clients[i].optimizer.step()
        self.decoder_optimizers[i].step()

        with torch.no_grad():
            _, predicted = torch.max(out.data, 1)
            train_correct = predicted.eq(y.view_as(predicted)).sum().item()

        self._round_buf.append((i, smashed_data.detach(), y.detach()))

        return {
            'acc': train_correct / y.size(dim=0),
            'loss': loss.item(),
            'recon_loss': recon_loss.item(),
        }

    def aggregate(self):
        # (3) smashed-data mixup: purely additive server-side signal, no
        # extra communication (reuses this round's already-uploaded smashed
        # data), gradient never reaches any client.
        t0 = time.time()
        n = len(self._round_buf)
        mixup_loss_total = 0.0
        if n >= 2:
            for idx in range(n):
                client_a, s_a, y_a = self._round_buf[idx]
                delta_i = self.agg_factor[client_a]
                for p in range(1, self.mixup_partners + 1):
                    # advance until we land on a different client's smashed
                    # data -- the buffer is filled one client's local batches
                    # at a time, so a plain (idx+p)%n mostly re-pairs a
                    # client with its own other batches, whereas the paper's
                    # mixup (Eq. 5) is defined between two DIFFERENT clients.
                    partner_idx = (idx + p) % n
                    tries = 0
                    while self._round_buf[partner_idx][0] == client_a and tries < n:
                        partner_idx = (partner_idx + 1) % n
                        tries += 1
                    _, s_b, y_b = self._round_buf[partner_idx]
                    m = min(s_a.size(0), s_b.size(0))
                    lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                    mixed = lam * s_a[:m] + (1 - lam) * s_b[:m]

                    out = self.server.model(mixed)
                    mix_loss = lam * self.criterion(out, y_a[:m]) \
                        + (1 - lam) * self.criterion(out, y_b[:m])
                    (delta_i * mix_loss / self.mixup_partners).backward()
                    mixup_loss_total += mix_loss.item()

        self.server.optimizer.step()
        mixup_compute_time = time.time() - t0

        # (5) local federated averaging of client models AND decoders, every
        # round (see header note on the paper's own tunable interval).
        ret_dict = self.aggregate_clients()

        t0 = time.time()
        agg_state = {}
        for key in self.decoders[0].state_dict().keys():
            agg_state[key] = sum(
                self.agg_factor[i] * self.decoders[i].state_dict()[key]
                for i in range(len(self.decoders))
            )
        self.aggregated_decoder.load_state_dict(agg_state)
        for d in self.decoders:
            d.load_state_dict(agg_state)
            self.comm_load_weights += 2 * calculate_load(self.aggregated_decoder)
        ret_dict['decoder_agg_compute_time'] = time.time() - t0
        ret_dict['server_mixup_compute_time'] = mixup_compute_time
        if n >= 2:
            ret_dict['mixup_loss'] = mixup_loss_total / (n * self.mixup_partners)

        self._round_buf = []
        return ret_dict

# ------------------------------------------------------------------------------
