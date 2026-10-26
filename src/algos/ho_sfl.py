# ------------------------------------------------------------------------------
# HO-SFL (Hybrid-Order Split Federated Learning), arXiv:2603.14773. Ported from
# the authors' HKU-WILL-Lab/HO-SFL, HO_SFL_CV/src/runner/ho_sfl_runner.py, and
# run against the paper's pseudocode (Alg. 1 main loop, Alg. 2 ZOScalar,
# Alg. 3 ClientSync).
#
# One HO-SFL ITERATION per client batch (paper Alg.1 with the batch's client as
# the participating set; the authors' runner does the same per `next(loader)`):
#   1. ClientSync (Alg.3 L3): the client catches up on iterations it missed
#      since its last turn by fetching their P seeds + P averaged scalars.
#   2. Server broadcasts P seeds (Alg.1 L4).
#   3. Client forward, NO backward; uploads activation z + labels (L10).
#   4. Server forward + backward + optimizer step on its ONE model (L14-18,
#      Eq.14), sends the activation gradient lambda back (L19).
#   5. Client runs P perturbed forwards (Alg.2), projects each onto lambda:
#      v_p = lambda . (z~_p - z) (Eq.17), uploads the P scalars (L23).
#   6. Server averages them and broadcasts v_bar (L26); with one participating
#      client v_bar = v.
#   7. Client forms g = (1/(P mu)) sum_p v_bar_p u_p (Eq.19) and steps AdamW (L28).
# No model weights ever cross the network: clients stay identical by replaying
# the same seeds and scalars, which is why one shared client model stands for
# all of them here. Byte sizes follow the authors' communicator.py: tensors at
# numel x element_size, 4 B per seed and per scalar.
#
# Differences from the original port (changed after review):
#   - The port buffered all of a round's batches and made ONE server step and
#     ONE client step per round (11 updates in an 11-round run). Now every batch
#     is a full iteration: one server step and one client step per batch.
#   - Aggregation traffic was one lump `weights.scalars` charge per round; it is
#     now charged per item: zo_seeds_down, zo_scalars_up, zo_scalars_down,
#     zo_sync_down.
#   - Benchmark settings (same for every method, not the authors' defaults):
#     batch 256 (authors 32), 10 clients (authors 100), and a round is one
#     local epoch of every client (authors: one batch per sampled client, run
#     to a 120k-sample budget). Methods are thus compared at equal samples
#     processed, as the authors themselves compare them.
#   - Client sampling. Paper Alg.1 L4 samples a random subset S_t of size K at
#     the start of each round (authors' code: 10 of 100 clients, one batch
#     each). Here S_t = {the client whose batch it is}: clients take turns, and
#     every client trains on all its data every round, as in every other method
#     of this benchmark (the shared loop has no client sampling). |S_t| = 1 is a
#     valid case of Alg.1. What sampling implies for communication is kept:
#     clients that missed iterations catch up via ClientSync (Alg.3, row Z4).
#   - Benchmark fairness: RANDOM init for every method (use_pretrained=false,
#     freeze_bn=false), unlike the paper's ImageNet-pretrained ResNet-18. With
#     random BatchNorm the client trains in train() mode like every other
#     method (the reference pins it to eval(), which only works with frozen
#     pretrained BN: from random init the running stats would never move).
#     Clean and perturbed forwards then normalise with the same batch's
#     statistics, so v_p stays a true perturbation difference.
# ------------------------------------------------------------------------------
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
@register_algorithm("ho_sfl")
class HO_SFL(FLAlgorithm):

    def __init__(self, *args, **kwargs):
        super(HO_SFL, self).__init__(*args, **kwargs)

        # Initialisation. The paper and the authors' conf/base.yaml start from
        # an ImageNet-pretrained ResNet-18 with frozen BatchNorm
        # (use_pretrained / freeze_bn = True). This benchmark starts EVERY
        # method from the same random init, so both default to False here
        # (algorithm/ho_sfl.yaml); set them True to reproduce the paper's own
        # setting. Applied BEFORE the model is shared out below, so every
        # client sees the same weights either way.
        self.use_pretrained = self.cfg.get('use_pretrained', False)
        self.freeze_bn = self.cfg.get('freeze_bn', False)
        if self.use_pretrained:
            full_state = pretrained_resnet18_state_dict()
            load_pretrained_client(self.clients[0].model, full_state)
            load_pretrained_server(self.server.model, full_state)
        if self.freeze_bn:
            freeze_batchnorm_affine(self.clients[0].model)
            freeze_batchnorm_affine(self.server.model)

        # single shared client model across all clients -- no per-client
        # weight divergence, nothing to aggregate as a weight vector.
        shared_model = self.clients[0].model
        for c in self.clients[1:]:
            c.model = shared_model

        shared_optimizer = self._get_optimizer(shared_model.parameters())
        for c in self.clients:
            c.optimizer = shared_optimizer
            c.lr_scheduler = None

        self.server.optimizer = self._get_optimizer(self.server.model.parameters())

        self.P = self.cfg.zo_p
        self.mu = self.cfg.zo_mu
        # iteration bookkeeping for ClientSync (Alg.3): global iteration count
        # and the iteration each client last took part in
        self._iter = 0
        self._last_seen = [0] * len(self.clients)
        # Alg.1 L2: the server keeps every iteration's P seeds + P averaged
        # scalars so lagging clients can catch up (Alg.3). Only its size
        # matters here (the shared model never needs replaying), so it is
        # tracked as a count and reported in server_side_extra_memory().
        self._history_iters = 0

        # The single-shared-client-model premise above IS the paper's
        # dimension-free aggregation, and aggregate() below relies on it by
        # only ever stepping clients[0]. Assert it so a future change that
        # gives a client its own model/optimizer fails loudly instead of
        # silently training one client and reporting it as all of them.
        assert all(c.model is self.clients[0].model for c in self.clients), \
            "HO-SFL requires one shared client model across all clients"
        assert all(c.optimizer is self.clients[0].optimizer for c in self.clients), \
            "HO-SFL requires one shared client optimizer across all clients"

    def _get_optimizer(self, params):
        if self.cfg.optimizer == 'adamw':
            return torch.optim.AdamW(
                params, lr=self.cfg.lr, betas=tuple(self.cfg.betas),
                weight_decay=self.cfg.weight_decay
            )
        elif self.cfg.optimizer == 'sgd':
            return torch.optim.SGD(
                params, lr=self.cfg.lr, momentum=self.cfg.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.cfg.optimizer}")

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        # Pretrained + frozen BN (the paper's setting): keep the client in
        # eval() so the perturbed forwards never touch BN statistics, exactly
        # as the reference does. Random init (this benchmark's default): leave
        # the client in train() like every other method -- see the header.
        if self.freeze_bn:
            self.clients[0].model.eval()

    def client_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it
        client = self.clients[i].model
        s = 8 if self.use_64bit else 4          # bytes per scalar
        SEED = 4                                # bytes per seed (int32)

        # 1. ClientSync (Alg.3 L3): seeds + averaged scalars of every missed
        #    iteration. One shared model stands for all clients here, so the
        #    state is already current; only the traffic is charged.
        missed = self._iter - self._last_seen[i]
        if missed > 0:
            self.charge_weights_zo('zo_sync_down', missed * self.P * (SEED + s))

        # 2. Seeds for this iteration (Alg.1 L4)
        seeds = [int(np.random.randint(0, 1_000_000)) for _ in range(self.P)]
        self.charge_weights_zo('zo_seeds_down', self.P * SEED)

        # 3. Client forward, no backward (Alg.1 L9-10)
        with self.phase('client', i):
            with torch.no_grad():
                a_m = client(x)
        # The client keeps its own a_m (step 5 needs it); what reaches the
        # server is a copy, so the upload is charged -- and held as server
        # memory -- on that copy, and a_m stays client memory.
        server_input = a_m.clone().detach().requires_grad_(True)
        self.charge_cut_activation(server_input)
        self.charge_cut_labels(y)     # loss is computed server-side

        # 4. Server first-order update on its single model (Alg.1 L14-19)
        with self.phase('server', i):
            self.server.optimizer.zero_grad()
            out = self.server.model(server_input)
            loss = self.criterion(out, y)
            with torch.no_grad():
                train_loss = loss.item()
                _, predicted = torch.max(out.data, 1)
                train_correct = predicted.eq(y.view_as(predicted)).sum().item()
            loss.backward()
            self.server.optimizer.step()
        g_a_m = server_input.grad.clone().detach()
        self.charge_cut_gradient(g_a_m)
        self.hold('client', x, a_m, g_a_m, i=i)

        # 5. ZOScalar (Alg.2): P perturbed forwards projected onto lambda
        v = torch.zeros(self.P, device=a_m.device, dtype=a_m.dtype)
        with self.phase('client', i):
            for p_idx, seed in enumerate(seeds):
                _perturb(client, seed, self.mu)
                with torch.no_grad():
                    a_tilde = client(x)
                self.hold('client', a_tilde, i=i)
                v[p_idx] = torch.sum((a_tilde - a_m) * g_a_m)
                _perturb(client, seed, -self.mu)
                del a_tilde
        self.charge_weights_zo('zo_scalars_up', self.P * s)        # Alg.1 L23

        # 6. Server averages and broadcasts (Alg.1 L26); one participant
        bar_v = v
        self.charge_weights_zo('zo_scalars_down', self.P * s)

        # 7. Client update (Alg.1 L27-28, Eq.19): g = (1/(P mu)) sum v_bar_p u_p
        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad(set_to_none=False)
            scale = 1.0 / (self.P * self.mu)
            for p_idx, seed in enumerate(seeds):
                _perturb_accumulate_grad(client, seed, bar_v[p_idx].item() * scale)
            self.clients[i].optimizer.step()

        self._iter += 1
        self._last_seen[i] = self._iter
        self._history_iters += 1            # Alg.1 L2 history grows by one entry
        return {
            'acc': train_correct / y.size(dim=0),
            'loss': train_loss,
        }

    def server_side_extra_memory(self):
        '''Alg.1 L2: history of seeds + averaged scalars for ClientSync, one
        entry per iteration, kept on the server for the whole run.'''
        s = 8 if self.use_64bit else 4
        nbytes = self._history_iters * self.P * (4 + s)
        return [{
            'item': f'catch-up history: {self._history_iters} iterations x P '
                    f'x (4 B seed + {s} B scalar)',
            'mb': nbytes / 1024 ** 2,
            'note': 'paper Alg.1 L2 "history buffers for seeds and scalars"',
        }]

    def aggregate(self):
        # Nothing to aggregate: every update already happened per iteration in
        # client_step, and no model weights are exchanged (dimension-free).
        self.aggregated_client = self.clients[0].model
        return {}

# ------------------------------------------------------------------------------
