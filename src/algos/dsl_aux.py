# ------------------------------------------------------------------------------
# DSL-Aux -- "Decoupled Split Learning via Auxiliary Loss" (arXiv:2601.19261).
#
# IMPORTANT PROVENANCE NOTE: this is an AI-ASSISTED, NO-CODE REIMPLEMENTATION
# per CLAUDE.md's rule for methods without a working released implementation.
# It is structurally adapted from a partial third-party reference,
# juniorfelix998/sl-fl-dgl's `cifar-100/sl/split-2/sl_dsl.py`, specifically
# that file's `train_standard_split` cut/communication pattern (a real
# activation uploaded client->server, and a real gradient sent back
# server->client across the cut) -- NOT its `train_dgl` function, which never
# sends a gradient across the cut at all and is the wrong training mode for
# this method.
#
# The combination implemented below -- the client's own update driven by BOTH
# the real server gradient AND a simultaneous local auxiliary-classifier loss
# -- is not literally present in either reference function taken in
# isolation; it is reconstructed from the paper's own description (real
# activations/gradients cross the cut every step, plus an auxiliary loss term
# that lets the client start updating before/without waiting on the full
# end-to-end gradient) per explicit instruction to favor the paper's actual
# mechanism over the reference code alone. This has NOT been validated
# against the paper's own reported CIFAR-10/CIFAR-100/ImageNet numbers --
# that reproduction is an explicit follow-up, out of scope for the MNIST
# smoke test this was written for.
#
# Non-federated: the paper's method is a single client/server split with no
# multi-client weight aggregation, so `aggregate()` below does no cross-client
# averaging of either the client model or its auxiliary classifier --
# `comm_load_weights` stays 0 for the whole run, matching CLAUDE.md's own
# metric-definition note ("Communication -- WEIGHTS: ... ~=0 for
# non-federated methods"). This harness's client population/loop still runs
# (each client trains fully independently, nothing ever shared between them)
# but `num_clients=1` is the paper-faithful configuration.
#
# The local auxiliary classifier head reuses this harness's existing
# `clients[i].auxiliary_model` (already instantiated for every algorithm
# regardless of use, see `src/main.py`'s unconditional `init_auxiliary()`
# call) rather than a bespoke architecture matching the reference's own
# `auxillary_classifier2` (conv blocks + adaptive-pool-to-(2,2) + 3-layer
# MLP) -- sufficient for the "runs + logs correctly, non-chance accuracy on
# MNIST" bar; a bespoke head is only worth adding if/when paper-faithful
# CIFAR-100/ResNet-110 reproduction is pursued.
# ------------------------------------------------------------------------------
import time
import torch

from algos import register_algorithm, FLAlgorithm

# ------------------------------------------------------------------------------
@register_algorithm("dsl_aux")
class DSLAux(FLAlgorithm):

    def __init__(self, *args, **kwargs):
        super(DSLAux, self).__init__(*args, **kwargs)
        self.aux_loss_weight = self.cfg.aux_loss_weight
        # non-federated: no aggregation ever happens, so `aggregated_client`
        # just needs to exist for full_model()/eval_mode() from round 0.
        self.aggregated_client = self.clients[0].model

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        for c in self.clients:
            c.auxiliary_model.train()

    def special_models_eval_mode(self):
        for c in self.clients:
            c.auxiliary_model.eval()

    # the local auxiliary classifier head is resident on the client device
    def client_side_modules(self, i):
        return [self.clients[i].model, self.clients[i].auxiliary_model]

    def client_side_optimizers(self, i):
        return [self.clients[i].optimizer, self.clients[i].auxiliary_model.optimizer]

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it

        # NOTE this method interleaves one client graph across the cut: the
        # client's graph is built, held while the server runs, and then
        # backwarded TWICE (once for the local auxiliary loss with
        # retain_graph=True, once for the injected server gradient). Phases are
        # therefore bracketed by which MODULE owns the work, not by wall-clock
        # ordering -- `smashed_data` is a fresh detached leaf, so none of the
        # client's graph is mis-attributed to the server.
        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            self.clients[i].auxiliary_model.optimizer.zero_grad()

            # real client forward -- graph stays attached (not detached) so both
            # the real server gradient and the local auxiliary loss's gradient
            # can later accumulate into this client model's own parameters.
            splitting_output = self.clients[i].model(x)

        # separate leaf sent across the cut to the server (real activation
        # upload, matching the reference's `train_standard_split`).
        smashed_data = splitting_output.clone().detach().requires_grad_(True)
        self.charge_cut_activation(smashed_data)
        self.charge_cut_labels(y)     # loss is computed server-side

        with self.phase('server', i):
            self.server.optimizer.zero_grad()
            # real server forward + backward (first-order, unlike ho_sfl/mu_splitfed)
            out = self.server.model(smashed_data)
            loss = self.criterion(out, y)
            loss.backward()
            self.server.optimizer.step()

        grad_at_cut = smashed_data.grad.clone().detach()
        self.charge_cut_gradient(grad_at_cut)
        self.hold('client', grad_at_cut, i=i)

        with self.phase('client', i):
            # client-side auxiliary local loss on the same (still-attached)
            # output -- its backward adds gradient into the client's own conv
            # params too, on top of the real server gradient injected below.
            aux_out = self.clients[i].auxiliary_model.forward_inner(splitting_output)
            aux_loss = self.criterion(aux_out, y)
            (self.aux_loss_weight * aux_loss).backward(retain_graph=True)

            # inject the real server gradient at the cut into the same client graph.
            splitting_output.backward(grad_at_cut)

            self.clients[i].optimizer.step()
            self.clients[i].auxiliary_model.optimizer.step()

        with torch.no_grad():
            _, predicted = torch.max(out.data, 1)
            train_correct = predicted.eq(y.view_as(predicted)).sum().item()

        return {
            'acc': train_correct / y.size(dim=0),
            'loss': loss.item(),
            'aux_loss': aux_loss.item(),
        }

    def aggregate(self):
        # non-federated: no cross-client averaging of model or auxiliary
        # weights -- each client keeps training its own independent copy.
        t0 = time.time()
        self.aggregated_client = self.clients[0].model
        return {'client_agg_compute_time': time.time() - t0}

# ------------------------------------------------------------------------------
