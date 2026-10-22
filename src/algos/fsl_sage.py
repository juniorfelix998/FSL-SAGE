# ------------------------------------------------------------------------------
# FSL-SAGE (Nair et al., arXiv:2505.23182, ICML 2025), run against the paper's
# pseudocode (Alg. 1 client, Alg. 2 F-server, Alg. 3 S-server).
#
# Differences from the authors' implementation (changed after review):
#   A. WHICH batches are uploaded. Alg.1 l.11 / Alg.3 l.3: upload when
#      k = 0 mod K/Q with k in [K] = 1..K, i.e. paper batch k = local_iter + 1.
#      With K/Q = h = 5 and K = 24 that is batches 5,10,15,20 -> 4 per round.
#      The original code uploaded at local_iter = 0,5,...,20 -> 5 per round.
#      (Alg.1 writes its loop as 1..K-1, but the text says K local iterations
#      and Alg.3 uses [K]; the two readings agree whenever h does not divide K,
#      e.g. K = 24 and K = 118.)
#   B. WHEN the aux model is aligned. Alg.1 l.3-4: the client receives the
#      aligned x_a at the START of round t (t = 0 mod l); Eq.(2) aligns on the
#      Qt samples uploaded in the t rounds BEFORE it; Alg.3 l.9 aligns after the
#      uploads. So alignment now runs in `begin_round(t)`, before any client
#      trains. The original code aligned in the middle of iteration 0 (on that
#      batch too), while the client's autograd graph was alive -- which is what
#      reported Held-across-cut = 94.01 MiB. Under the paper it is 0.
#      At t = 0 the buffer is empty (Qt = 0): nothing is aligned, but the
#      initial x_a^0 is still sent (Alg.1 l.4), so rounds 0..l-1 train with an
#      unaligned aux model.
#   C. Initial broadcast. Alg.2 l.4 broadcasts x_c^0 to all clients before
#      round 0, plus l.8 after every round -> N(R+1) downloads. The original
#      code charged only the N*R post-aggregation broadcasts.
#   E. (memory only) The client computes its gradient estimate with
#      create_graph=False. Alg.1 l.14 needs only the VALUE of z_b_hat; the
#      original code built the second-order graph (needed by align() alone) on
#      every client step, which kept the aux forward graph alive after the step
#      and inflated client working memory / Held-across-cut. Numbers identical.
# ------------------------------------------------------------------------------
import inspect
import torch
from algos import register_algorithm, FLAlgorithm

# ------------------------------------------------------------------------------
@register_algorithm("fsl_sage")
class FSLSAGE(FLAlgorithm):

    def __init__(self, *args, **kwargs):
        super(FSLSAGE, self).__init__(*args, **kwargs)

        self.server_update_interval = self.cfg.server_update_interval
        self.align_interval = self.cfg.align_interval
        self.iters_per_epoch = [len(c.train_loader) for c in self.clients]

    def full_model(self, x):
        return self.server.model(self.aggregated_client(x))

    def special_models_train_mode(self, t):
        for c in self.clients: c.auxiliary_model.train()

    def special_models_eval_mode(self):
        for c in self.clients: c.auxiliary_model.eval()

    # Where the auxiliary (surrogate) model lives -- paper Alg. 1 vs Alg. 3:
    #   * CLIENT keeps x_a and RUNS it every batch to synthesise the cut-layer
    #     gradient (`__client_backward`), but never trains it: the aux optimizer
    #     only steps inside `align()`. So the client holds x_a's params + buffers
    #     and nothing else.
    #   * SERVER trains (aligns) x_a every l rounds -- that is where its grads
    #     and Adam state are created. The Server total (table value) carries ONE
    #     aux copy being aligned (params + grads + Adam); the other N-1 clients'
    #     copies and the alignment buffers are listed in server_side_extra_memory.
    # (Earlier the aux grads + Adam were charged to the client, overstating the
    # client by ~24 MiB for the 2.1M-param aux head.)
    def client_side_modules(self, i):
        return [self.clients[i].model]

    def client_side_optimizers(self, i):
        return [self.clients[i].optimizer]

    def client_side_inference_modules(self, i):
        return [self.clients[i].auxiliary_model]

    def server_side_modules(self):
        # all clients' aux models are the same shape; client 0's stands for
        # "the copy currently being aligned"
        return [self.server.model, self.clients[0].auxiliary_model]

    def server_side_optimizers(self):
        return [self.server.optimizer, self.clients[0].auxiliary_model.optimizer]

    def server_side_extra_memory(self):
        '''What Alg. 3 keeps on the S-server beyond the Server total (table
        value, `peak_server_mem_mb`), for the OTHER N-1 clients -- the Server
        total already carries the client being aligned (its aux model + Adam,
        and its alignment buffer, held during `begin_round`), so nothing is
        counted twice.

        Buffers are reported AT CAPACITY (steady state: after an alignment on a
        full buffer, z_f and the true gradients both hold max_dataset_size
        samples). The end-of-run size of every buffer is given in the note,
        since in a short run the true-gradient buffer can still be a smaller
        snapshot from the last alignment.'''
        MB = 1024 ** 2
        others = self.clients[1:]
        sample = next((c.auxiliary_model.data_x[0].numel()
                       * c.auxiliary_model.data_x.element_size()
                       for c in self.clients
                       if c.auxiliary_model.data_x.numel()), 0)
        label = 8                                      # int64 label
        aux0 = self.clients[0].auxiliary_model
        # buffer cap (max_dataset_size); aux models without one (test stubs)
        # fall back to the largest buffer actually seen
        cap = getattr(aux0, 'max_data_size', None) or max(
            c.auxiliary_model.data_x.shape[0] if c.auxiliary_model.data_x.numel()
            else 0 for c in self.clients)
        per_client_cap = cap * (2 * sample + label)   # z_f + true grad + y
        end_of_run = sum(
            t.numel() * t.element_size()
            for c in self.clients
            for t in (c.auxiliary_model.data_x, c.auxiliary_model.data_y,
                      c.auxiliary_model.data_labels)
        )
        aux = self._module_components(
            [c.auxiliary_model for c in others],
            [c.auxiliary_model.optimizer for c in others],
        )
        return [
            {'item': f'alignment buffers of the other {len(others)} clients, '
                     f'at capacity ({cap} x (z_f + true grad + y))',
             'mb': len(others) * per_client_cap / MB,
             'note': f'paper p.5 / Alg.3 "Store for client i"; steady state. '
                     f'Measured end of run, all {len(self.clients)} clients: '
                     f'{end_of_run / MB:.2f} MiB'},
            {'item': f'aux models of the other {len(others)} clients (params+grads+Adam)',
             'mb': sum(r['mb'] for r in aux),
             'note': 'paper Alg.3 l.9-13: server aligns every client\'s x_a; '
                     'the Server total (table value) holds one'},
        ]

    def should_probe_memory(self, t, i, j, k):
        # also sample the first upload (local_iter = h-1), or the server step
        # would never be measured
        return super().should_probe_memory(t, i, j, k) or (
            j == 0 and k == self.server_update_interval - 1
        )

    def begin_round(self, t):
        '''Round boundary, before any client trains (fixes B and C).'''
        if t == 0:
            # fix C -- Alg.2 l.4: broadcast the initial client model x_c^0
            for c in self.clients:
                self.charge_weights_model(c.model, 'client', 'down')

        if t % self.align_interval != 0:
            return

        # fix B -- Alg.3 l.9-13 then Alg.1 l.3-4: align every client's aux model
        # on the buffer collected in rounds < t, then send it down. Runs on the
        # SERVER; the true-gradient buffer never crosses the cut.
        self.meter.enabled = bool(self.cfg.get('measure_memory', True))
        self._phase_times = {}
        for i, c in enumerate(self.clients):
            self._cur_client = i
            aux = c.auxiliary_model
            with self.phase('server', i):
                if aux.data_x.numel():         # empty at t = 0 (Eq.2: Qt = 0)
                    # the buffer being aligned is resident on the server for
                    # the whole pass: z_f before the refresh, true grads after
                    self.hold('server', aux.data_x, i=i)
                    aux.refresh_data()
                    self.hold('server', aux.data_y, i=i)
                    aux.align()
            self.charge_weights_model(aux, 'aux', 'down')
            self.meter.end_step()
        # alignment compute time is reported with the round's first step
        self._align_times = self._phase_times
        self._phase_times = {}
        self.meter.enabled = False

    def end_step(self):
        out = super().end_step()
        pending = getattr(self, '_align_times', None)
        if pending:
            for key, v in pending.items():
                out[key] = out.get(key, 0.0) + v
            self._align_times = None
        return out

    def client_step(self, rd_cl_ep_it, x, y):

        t, i, j, k = rd_cl_ep_it       # (round, client, epoch, iter)

        ret_dict = dict()
        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            # client feedforward
            splitting_output = self.clients[i].model(x)
            local_smashed_data = \
                splitting_output.clone().detach().requires_grad_(True)

        # Alg.1 l.14-15: the client backprops with the aux model's gradient
        # estimate right away. Alignment now happens at the round boundary
        # (fix B), so the client never waits on the server and its graph is
        # freed before any server work: Held-across-cut = 0.
        self.__client_backward(i, splitting_output, local_smashed_data, y)

        # What crosses the cut is a COPY of the smashed data; the client keeps
        # no reference to its own activations once its step is done (otherwise
        # their storage stays alive while the server works and shows up as a
        # spurious Held-across-cut).
        upload = splitting_output.detach().clone()
        del splitting_output, local_smashed_data

        local_iter = j * self.iters_per_epoch[i] + k
        # fix A -- Alg.1 l.11 / Alg.3 l.3: paper batch k = local_iter + 1,
        # uploaded when k = 0 mod K/Q
        if (local_iter + 1) % self.server_update_interval == 0:
            smashed_data = upload.requires_grad_(True)
            self.charge_cut_activation(smashed_data)
            self.charge_cut_labels(y)     # loss is computed server-side

            with self.phase('server', i):
                self.server.optimizer.zero_grad()
                out = self.server.model(smashed_data)
                s_loss = self.criterion(out, y)

                with torch.no_grad():
                    train_loss = s_loss.item()
                    _, predicted = torch.max(out.data, 1)
                    train_correct = predicted.eq(y.view_as(predicted)).sum().item()
                    ret_dict['g_loss'] = train_loss
                    ret_dict['g_acc'] = train_correct / y.size(dim=0)

                s_loss.backward()
                self.server.optimizer.step()

                # Alg.3 l.7 -- the alignment buffer is held SERVER-side: these
                # are the very activations just uploaded, so no extra cut bytes.
                self.clients[i].auxiliary_model.add_datapoint(
                    smashed_data.detach(), y
                )
                self.hold('server', self.clients[i].auxiliary_model.data_x, i=i)

        return ret_dict

    def __client_backward(self, i, splitting_output, local_smashed_data, y):
        '''Client backpropagation using the surrogate's synthesised gradient --
        no gradient is requested from the server, so nothing is charged here.'''
        aux = self.clients[i].auxiliary_model
        with self.phase('client', i):
            # fix E: first-order estimate only (see header)
            if 'create_graph' in inspect.signature(aux.forward).parameters:
                client_grad_approx = aux(local_smashed_data, y, create_graph=False)
            else:
                client_grad_approx = aux(local_smashed_data, y)
            splitting_output.backward(client_grad_approx)
            self.clients[i].optimizer.step()

# ------------------------------------------------------------------------------