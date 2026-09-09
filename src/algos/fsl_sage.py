# ------------------------------------------------------------------------------
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

    # the surrogate model is a permanent resident of the client device: the
    # client evaluates it every batch to synthesise a gradient at the cut
    def client_side_modules(self, i):
        return [self.clients[i].model, self.clients[i].auxiliary_model]

    def client_side_optimizers(self, i):
        return [self.clients[i].optimizer, self.clients[i].auxiliary_model.optimizer]

    def client_step(self, rd_cl_ep_it, x, y):

        t, i, j, k = rd_cl_ep_it       # (round, client, epoch, iter)

        ret_dict = dict()
        with self.phase('client', i):
            self.clients[i].optimizer.zero_grad()
            # client feedforward
            splitting_output = self.clients[i].model(x)
            local_smashed_data = \
                splitting_output.clone().detach().requires_grad_(True)

        # server model update
        local_iter = j * self.iters_per_epoch[i] + k
        if local_iter % self.server_update_interval == 0:
            smashed_data = splitting_output.clone().detach().requires_grad_(True)
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

                # The alignment buffer is held SERVER-side: these are the very
                # activations just uploaded above, so no extra cut bytes are due.
                self.clients[i].auxiliary_model.add_datapoint(
                    splitting_output.clone().detach(), y
                )
                self.hold('server', self.clients[i].auxiliary_model.data_x, i=i)

        # Alignment runs on the SERVER: refresh_data() needs the server model to
        # compute true cut-layer gradients over the stored buffer, and only the
        # aligned surrogate is then downloaded to the client. Hence the one-way
        # (not round-trip) weight charge -- the true-gradient buffer never
        # crosses the cut, it is consumed where it was produced.
        if t % self.align_interval == 0 and local_iter == 0:
            with self.phase('server', i):
                self.clients[i].auxiliary_model.refresh_data()
                self.clients[i].auxiliary_model.align()
            self.charge_weights_model(
                self.clients[i].auxiliary_model, 'aux', 'down'
            )

        # client backpropagation using the surrogate's synthesised gradient --
        # no gradient is requested from the server, so nothing is charged here
        with self.phase('client', i):
            client_grad_approx = \
                self.clients[i].auxiliary_model(local_smashed_data, y)
            splitting_output.backward(client_grad_approx)
            self.clients[i].optimizer.step()

        return ret_dict

# ------------------------------------------------------------------------------