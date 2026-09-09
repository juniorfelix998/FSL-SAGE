# ------------------------------------------------------------------------------
from abc import ABC, abstractmethod
from typing import Dict, List, Callable
import time
import os, importlib
import logging, copy
from dataclasses import dataclass
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import torch
import torch.nn as nn

from contextlib import contextmanager

from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader
from models import Server, Client
from utils.utils import calculate_load, Checkpointer
from utils.comm import CommLedger, tensor_bytes
from utils.memory import MemoryMeter, module_static_bytes, optimizer_state_bytes

# ------------------------------------------------------------------------------
def aggregate_models(model_list, weights, device='cpu'):
    '''Aggregate Models
    
    Aggregate a given list of models using weights supplied in `weights`.

    Params
    ------
    model_list - List of pytorch models
    weights - List of weights used in averaging
    device - Device to use for computations

    Returns
    -------
    aggregated - Torch model containing the aggregated weights.
    '''

    assert len(weights) == len(model_list),\
        "Length of model_list and weights is different."

    assert sum(weights) == 1, "Sum of weights should be 1"

    aggregated = copy.deepcopy(model_list[0])
    aggregated_weights = aggregated.state_dict()

    for key in aggregated_weights:
        aggregated_weights[key] = model_list[0].state_dict()[key] * weights[0]

    for i in range(1, len(model_list)):
        for key in aggregated_weights:
            aggregated_weights[key] += model_list[i].state_dict()[key] * weights[i]

    aggregated.to(device)
    aggregated.load_state_dict(aggregated_weights)

    # Won't add to the comm load here because I'll add it for when algs load
    # in the agg model
    return aggregated

# ------------------------------------------------------------------------------
class FLAlgorithm(ABC):
    aggregated_client   : nn.Module
    loss                : float = np.inf
    acc                 : float = 0.0

    # Communication is priced through `self.ledger` (see utils/comm.py), which
    # keeps a per-category breakdown. These three stay byte-valued and cumulative
    # exactly as before -- they are now derived sums, so every existing consumer
    # of results.json / plot_results.py keeps working unchanged.
    @property
    def comm_load_cut(self) -> float:
        return self.ledger.cut_total

    @property
    def comm_load_weights(self) -> float:
        return self.ledger.weights_total

    @property
    def comm_load(self) -> float:
        return self.ledger.total

    def __init__(self,
        cfg: DictConfig, server: Server, clients: List[Client],
        test_loader: DataLoader, agg_factor: List[float],
        device: str = 'cpu', use_64bit: bool = False
    ):
        self.cfg = cfg
        self.server = server
        self.clients = clients
        self.test_loader = test_loader
        self.criterion = server.criterion

        self.agg_factor = agg_factor
        self.device = device
        self.use_64bit = use_64bit

        # --- measurement state ------------------------------------------
        self.ledger = CommLedger()
        self.meter = MemoryMeter(
            enabled=bool(cfg.get('measure_memory', True))
        )
        # Probing only the first few batches of each round keeps the
        # saved-tensor hooks' ~4% overhead amortized to well under 1%, and is
        # sufficient because every method's heaviest step is at or near the
        # round's first iteration: cse_fsl's server update fires at
        # `local_iter % q == 0`, fsl_sage's alignment at `local_iter == 0`, and
        # ho_sfl/mu_splitfed only do real work at (j, k) == (0, 0) at all.
        self.mem_probe_batches = int(cfg.get('mem_probe_batches', 2))
        self._cur_client = 0
        self._phase_times = {}
        self.held_across_cut = 0.0

    # -- communication charging ------------------------------------------
    # One helper per kind of transfer, so a method cannot silently disagree with
    # another about what a transfer costs. See utils/comm.py for the categories.

    def charge_cut_activation(self, t):
        self.ledger.charge_cut('act_up', tensor_bytes(t))

    def charge_cut_gradient(self, t):
        self.ledger.charge_cut('grad_down', tensor_bytes(t))

    def charge_cut_labels(self, y):
        '''Labels must cross the cut whenever the LOSS is computed server-side.
        Deliberately not charged for client-local-loss branches (cse_fsl's
        auxiliary head, han_locloss) -- that asymmetry is part of what those
        methods buy.'''
        self.ledger.charge_cut('labels_up', tensor_bytes(y))

    def charge_cut_scalar(self, n=1, direction='down'):
        '''n scalars crossing the cut. Sized from the run's actual dtype rather
        than a hard-coded 4 bytes, so `use_64bit: true` stays self-consistent.'''
        key = 'scalar_down' if direction == 'down' else 'scalar_up'
        self.ledger.charge_cut(key, n * self._scalar_size())

    def _scalar_size(self):
        return 8 if self.use_64bit else 4

    def charge_weights_model(self, model, kind, direction):
        '''kind in {client, aux, server}; direction in {up, down}.'''
        self.ledger.charge_weights(f'{kind}_{direction}', calculate_load(model))

    def charge_weights_roundtrip(self, model, kind):
        '''Upload + download of a full model, the FedAvg-style exchange.'''
        self.charge_weights_model(model, kind, 'up')
        self.charge_weights_model(model, kind, 'down')

    def charge_weights_scalars(self, nbytes):
        self.ledger.charge_weights('scalars', nbytes)

    # -- time + memory phases --------------------------------------------
    # A single bracket records BOTH elapsed time and peak retained-activation
    # bytes, replacing the ad-hoc t0/t1/t2 = time.time() markers each method used
    # to hand-roll. Keys and semantics of the emitted *_model_compute_time are
    # unchanged, so __add_aggregated_compute_times and the plots keep working.

    @contextmanager
    def phase(self, side, i=None):
        i = self._cur_client if i is None else i
        owner = ('client', i) if side == 'client' else ('server',)
        if side == 'server':
            # bytes this client must keep resident while the server works --
            # the real cost of crossing the cut with a gradient
            # saved_only: the input batch and any downloaded buffer are
            # resident regardless -- what the cut costs is the autograd graph
            # the client is forced to keep alive while it waits.
            self.held_across_cut = max(
                self.held_across_cut,
                self.meter.live_bytes(('client', i), saved_only=True)
            )
        key = f'{side}_model_compute_time'
        t0 = time.time()
        try:
            with self.meter.phase(owner):
                yield
        finally:
            self._phase_times[key] = \
                self._phase_times.get(key, 0.0) + (time.time() - t0)

    def hold(self, side, *tensors, i=None):
        '''Declare long-lived non-autograd tensors resident on one side.'''
        i = self._cur_client if i is None else i
        self.meter.hold(('client', i) if side == 'client' else ('server',), *tensors)

    def begin_step(self, rd_cl_ep_it, x, y):
        t, i, j, k = rd_cl_ep_it
        self._cur_client = i
        self._phase_times = {}
        self.meter.enabled = (
            bool(self.cfg.get('measure_memory', True))
            and j == 0 and k < self.mem_probe_batches
        )
        if self.meter.enabled:
            self.hold('client', x, y, i=i)     # the client owns its own batch

    def end_step(self):
        self.meter.end_step()
        out, self._phase_times = self._phase_times, {}
        return out

    # -- static memory partition -----------------------------------------
    # Which modules/optimizers live on ONE client device, and which on the server
    # host. Overridden by methods that put an auxiliary model on the client or
    # keep per-client server replicas.

    def client_side_modules(self, i):
        return [self.clients[i].model]

    def client_side_optimizers(self, i):
        return [self.clients[i].optimizer]

    def server_side_modules(self):
        return [self.server.model]

    def server_side_optimizers(self):
        return [self.server.optimizer]

    def register_meter_params(self):
        '''Refresh the parameter-storage exclusion set. Must run once per round:
        FedAvg rebuilds `aggregated_client` every round, and several algorithms
        deep-copy or rebind models.'''
        mods = [self.server.model, getattr(self, 'aggregated_client', None)]
        for i in range(len(self.clients)):
            mods.extend(self.client_side_modules(i))
        mods.extend(self.server_side_modules())
        self.meter.register_params(*mods)

    def memory_report(self):
        '''Per-device peak memory, in MiB. Reduction is max (never mean) all the
        way up -- across batches, epochs, rounds, and finally across clients --
        because the metric is what a SINGLE device must provide, not what the
        simulated population sums to.'''
        MB = 1024 ** 2
        per_client = []
        for i in range(len(self.clients)):
            params, grads = module_static_bytes(*self.client_side_modules(i))
            optim = optimizer_state_bytes(*self.client_side_optimizers(i))
            act = self.meter.peak_saved.get(('client', i), 0)
            resident = self.meter.peak.get(('client', i), 0)
            per_client.append({
                'params': params, 'grads': grads, 'optim': optim, 'act': act,
                # `resident` is saved + explicitly-held (input batch, downloaded
                # cut-gradient), i.e. what the device must actually provide
                'total': params + grads + optim + resident,
            })
        sp, sg = module_static_bytes(*self.server_side_modules())
        so = optimizer_state_bytes(*self.server_side_optimizers())
        sact = self.meter.peak_saved.get(('server',), 0)
        sresident = self.meter.peak.get(('server',), 0)

        worst = max(per_client, key=lambda d: d['total']) if per_client else {
            'params': 0, 'grads': 0, 'optim': 0, 'act': 0, 'total': 0
        }
        return {
            'peak_client_mem_mb'           : worst['total'] / MB,
            'client_param_mem_mb'          : worst['params'] / MB,
            'client_grad_mem_mb'           : worst['grads'] / MB,
            'client_optim_mem_mb'          : worst['optim'] / MB,
            'client_act_peak_mem_mb'       : worst['act'] / MB,
            'client_mem_held_across_cut_mb': self.held_across_cut / MB,
            'peak_server_mem_mb'           : (sp + sg + so + sresident) / MB,
            'server_param_mem_mb'          : sp / MB,
            'server_grad_mem_mb'           : sg / MB,
            'server_optim_mem_mb'          : so / MB,
            'server_act_peak_mem_mb'       : sact / MB,
            'peak_client_mem_mb_per_client': [d['total'] / MB for d in per_client],
        }

    @abstractmethod
    def full_model(self):
        pass
    
    @abstractmethod
    def client_step(self, x, y):
        pass

    def special_models_train_mode(self, t):
        pass

    def special_models_eval_mode(self):
        pass

    def train_mode(self, t):
        for c in self.clients: c.model.train()
        self.server.model.train()

        # the condition is required because the aggregate models are not defined
        # at the zero'th round
        if t > 0: self.aggregated_client.train()
        self.special_models_train_mode(t)

    def eval_mode(self):
        for c in self.clients: c.model.eval()
        self.server.model.eval()
        self.aggregated_client.eval()
        self.special_models_eval_mode()

    def aggregate_clients(self):
        ret_dict = dict()
        t0 = time.time()
        self.aggregated_client = aggregate_models(
            [c.model for c in self.clients], self.agg_factor, self.device
        )
        agg_weights = self.aggregated_client.state_dict()

        for c in self.clients:
            c.model.load_state_dict(agg_weights)
            self.charge_weights_roundtrip(self.aggregated_client, 'client')

        ret_dict['client_agg_compute_time'] = time.time() - t0
        return ret_dict

    def aggregate(self):
        return self.aggregate_clients()

    def evaluate(self):
        test_correct = 0
        test_loss = []
        self.eval_mode()
        with torch.no_grad():
            for x, y in tqdm(
                self.test_loader, desc="Test Batch", unit='batch', leave=False
            ):
                x = x.to(self.device).double() if self.use_64bit \
                    else x.to(self.device).float()
                y = y.to(self.device).long()
                out = self.full_model(x)
                batch_loss = self.criterion(out, y)
                test_loss.append(batch_loss.item())
                _, predicted = torch.max(out.data, 1)
                test_correct += predicted.eq(y.view_as(predicted)).sum().item()
            loss = sum(test_loss) / len(test_loss)
            acc =  test_correct / len(self.test_loader.dataset)

        return acc, loss

# ------------------------------------------------------------------------------
# maps alg_name: str -> instance of FLAlgorithm
ALGORITHM_REGISTRY: Dict[str, FLAlgorithm] = dict()

# ------------------------------------------------------------------------------
def register_algorithm(name):
    """Decorator to register a new alogrithm"""
    def register_alg_cls(cls):
        if name in ALGORITHM_REGISTRY:
            raise ValueError(
                'Cannot register duplicate algorithm {}'.format(name)
            )
        if not issubclass(cls, FLAlgorithm):
            raise ValueError(
                'Model {} must extend {}'.format(name, cls.__name__)
            )
        ALGORITHM_REGISTRY[name] = cls
        return cls
    return register_alg_cls

# -----------------------------------------------------------------------------
# Automatically import any Python files in the models/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and file[0].isalpha():
        module = file[:file.find('.py')]
        importlib.import_module('algos.' + module)

# ------------------------------------------------------------------------------
@dataclass
class FLResults():
    server              : Server
    client_list         : List[Client]
    loss                : List[float]
    accuracy            : List[float]
    train_metrics       : List[Dict[str, List]]
    aggregation_metrics : Dict[str, List]
    comm_load_cut       : List[float]
    comm_load_weights   : List[float]
    comm_load           : List[float]
    avg_compute_times   : Dict[str, float]
    comm_breakdown      : List[Dict[str, float]]
    memory_metrics      : Dict[str, float]
    comm_to_target      : Dict[str, float]

# ------------------------------------------------------------------------------
def __aggregate_metrics_dict(
    client_metrics_dict, reduction=np.mean, keys='', excl_keys=None
):
    '''Given a list of dicts, each dict comprising of metric key-value pairs,
    where each value is a list of metrics values, one for each round of FL,
    compute the reduction of the values specified by `reduction` for keys
    containing the substring `keys` and not containing the substring
    `excl_keys`, and create a new dictionary that transforms a list of key ->
    value mappings to a dictionary of key -> value-list.
    '''
    def key_condition(k, inc_k, exc_k):
        if exc_k:
            return (inc_k in k and exc_k not in k)
        else:
            return (inc_k in k)

    agg_metrics_dict = {
        k: [reduction(v)] for k, v in client_metrics_dict[0].items()
        if key_condition(k, keys, excl_keys)
    }
    for met_dict in client_metrics_dict[1:]:
        [agg_metrics_dict[k].append(reduction(v)) for k, v in met_dict.items()
         if key_condition(k, keys, excl_keys)]

    return agg_metrics_dict

# ------------------------------------------------------------------------------
def __print_aggregate_metrics(
    client_metrics_dict, aggregation_metrics, prefix='tr.'
):
    agg_metrics_dict = __aggregate_metrics_dict(
        client_metrics_dict, reduction=lambda x: x[-1]
    )
    print_str = ''
    for i, (k, v) in enumerate(agg_metrics_dict.items()):
        print_str += f'avg {prefix} {k}: {np.mean(v):.2f}' if i == 0 \
            else f', avg {prefix} {k}: {np.mean(v):.2f}'

    for k, v in aggregation_metrics.items():
        print_str += f', {prefix} {k}: {v[-1]:.2f}'

    return print_str

# ------------------------------------------------------------------------------
def __add_aggregated_compute_times(client_metrics_dict, aggregation_metrics):
    # reduce and pack model_compute_times
    agg_metrics_dict = __aggregate_metrics_dict(
        client_metrics_dict, reduction=np.sum, keys='model_compute_time'
    )
    # update keys with agg_compute_times
    agg_metrics_dict.update({k: np.sum(v) for k, v in aggregation_metrics.items()})
    new_compute_times = {k: np.mean(v) for k, v in agg_metrics_dict.items()}
    return new_compute_times

# ------------------------------------------------------------------------------
def take_lr_step(obj):

    if obj.lr_scheduler:
        obj.lr_scheduler.step()
        lr = obj.lr_scheduler.get_last_lr()[0]
    else:
        lr = obj.optimizer.param_groups[0]['lr']
    return lr

# ------------------------------------------------------------------------------
def log_and_step_lr_per_round(alg, alg_name):
    lr_dict = {}
    log_dict = {}
    if alg_name not in ['fed_avg', 'sl_multi_server']:
        lr_dict['server_lr'] = take_lr_step(alg.server)
        log_dict[f'Server/server_lr'] = lr_dict['server_lr']

    return lr_dict, log_dict

# ------------------------------------------------------------------------------
def log_and_step_lr_per_client(i, alg, alg_name):

    # for client model
    lr_dict = {f"client_{i}_lr" : take_lr_step(alg.clients[i])}
    log_dict = {}
    log_dict[f'Clients/client_{i}/cl_model_lr'] = lr_dict[f'client_{i}_lr']

    # for server model(s) is multi_server
    if alg_name != 'fed_avg':
        if alg_name == 'sl_multi_server':
            lr_dict[f'server_{i}_lr'] = \
                take_lr_step(alg.servers[i])
            log_dict[f'Server/server_{i}/server_lr'] = \
                lr_dict[f'server_{i}_lr']

    # for auxiliary models. For fsl_sage, the optimization happens within the
    # align() method.
    aux_algorithms = (
        'cse_fsl', 'fsl_sage', 'dsl_aux', 'han_locloss', 'fedsplitx',
        'locfedmix_sl',
    )
    if alg_name in aux_algorithms:
        lr_dict['aux_lr'] = take_lr_step(alg.clients[i].auxiliary_model)

    # log auxiliary model learning rate for fsl algorithms
    if alg_name in aux_algorithms:
        log_dict.update({
            f'Clients/client_{i}/aux_model/aux_model_lr': \
                alg.clients[i].auxiliary_model.optimizer.param_groups[0]['lr']
        })

    return lr_dict, log_dict

# ------------------------------------------------------------------------------
def compute_comm_to_target(test_acc, comm_load, comm_load_cut,
                            comm_load_weights, target_acc):
    '''Cumulative bytes at the first round whose test accuracy reaches
    `target_acc`.

    This is the round-count-independent number to rank on. Reading the final
    round's cumulative byte count instead makes a method that was run for more
    rounds look worse, and rewards a method that converges slowly -- exactly the
    trap the existing table fell into when a 5-round dsl_aux run was compared
    against a 3-round SplitFedv1 run.

    Returns None-valued fields when the target was never reached, so a
    non-converged run is visibly incomparable rather than silently ranked.
    '''
    for idx, acc in enumerate(test_acc):
        if acc >= target_acc:
            return {
                'target_acc'              : target_acc,
                'rounds_to_target'        : idx + 1,
                'comm_to_target'          : comm_load[idx],
                'comm_cut_to_target'      : comm_load_cut[idx],
                'comm_weights_to_target'  : comm_load_weights[idx],
            }
    return {
        'target_acc'              : target_acc,
        'rounds_to_target'        : None,
        'comm_to_target'          : None,
        'comm_cut_to_target'      : None,
        'comm_weights_to_target'  : None,
    }

# ------------------------------------------------------------------------------
def _run_fl_algorithm(
    cfg:DictConfig,
    server: Server,
    clients: List[Client],
    test_loader: DataLoader,
    checkpointer: Checkpointer,
    torch_device,
    logger_fn: Callable,
    test_loss=None,
    test_acc=None,
    train_metrics=None,
    aggregation_metrics=None,
    comm_load=None,
    comm_load_cut=None,
    comm_load_weights=None,
    comm_breakdown=None,
) -> FLResults:

    # get algorithm
    # Measurement knobs live in the GLOBAL config but algorithms only ever see
    # `cfg.algorithm`, so inject them there rather than duplicating the defaults
    # across all 14 algorithm yamls.
    with open_dict(cfg.algorithm):
        cfg.algorithm.measure_memory = cfg.get('measure_memory', True)
        cfg.algorithm.mem_probe_batches = cfg.get('mem_probe_batches', 2)

    alg = ALGORITHM_REGISTRY[cfg.algorithm.name](
        cfg.algorithm, server, clients, test_loader,
        cfg.agg_factor, 
        device=torch_device, use_64bit=cfg.use_64bit
    )

    with open_dict(cfg):
        if cfg.comm_threshold_mb is None:
            cfg.comm_threshold_mb = np.inf

    if comm_load is None: comm_load = []
    if comm_load_cut is None: comm_load_cut = []
    if comm_load_weights is None: comm_load_weights = []
    if comm_breakdown is None: comm_breakdown = []
    if test_loss is None: test_loss = []
    if test_acc is None: test_acc = []
    if train_metrics is None:
        train_metrics = [{} for _ in range(cfg.num_clients)]
    if aggregation_metrics is None:
        aggregation_metrics = {}
    
    # main loop
    with logging_redirect_tqdm():
        for t in tqdm(
            range(cfg.rounds), unit="rd", desc="Round", leave=False,
            colour='green'
        ):

            log_dict = {}
            # set all models to train mode and train
            alg.train_mode(t)
            # refresh the meter's parameter-storage exclusion set: FedAvg
            # rebuilds `aggregated_client` every round
            alg.register_meter_params()
            with tqdm(
                range(cfg.num_clients), unit="cl", desc="Client", leave=False,
                colour='blue'
            ) as pbar:
                for i in pbar:
                    tr_mets = {}
                    for j in tqdm(
                        range(clients[i].epochs), unit="ep", desc="Local epoch",
                        leave=False
                    ):
                        with tqdm(
                            clients[i].train_loader, unit="batch",
                            desc="Local batch", leave=False
                        ) as pbar_local:
                            for k, (x, y) in enumerate(pbar_local):
                                x = x.to(torch_device).double() \
                                    if cfg.use_64bit else x.to(torch_device).float()
                                y = y.to(torch_device).long()

                                alg.begin_step((t, i, j, k), x, y)
                                tr_metrics = alg.client_step(
                                    (t, i, j, k), x, y
                                )
                                tr_metrics.update(alg.end_step())
                                pbar_local.set_postfix(**tr_metrics)
                                if j == 0 and k == 0:
                                    tr_mets = {
                                        k: [v] for k, v in tr_metrics.items()
                                    }
                                else:
                                    [tr_mets[k].append(v) for k, v in
                                    tr_metrics.items()]

                    # compute mean of metrics
                    tr_mets = {k: np.mean(v) if 'model_compute_time' not in k
                               else np.sum(v) for k, v in tr_mets.items()}
                    if t == 0:
                        train_metrics[i] = {
                            k: [v] for k, v in tr_mets.items()
                        }
                    else:
                        [train_metrics[i][k].append(v) for k, v in
                        tr_mets.items()]
                    log_dict.update({
                        f'Clients/client_{i}/{k}': v for k, v in tr_mets.items()
                    })

                    # adjust learning rate based on algorithm
                    lr_dict, log_dict_ = log_and_step_lr_per_client(
                        i, alg, cfg.algorithm.name
                    )
                    log_dict.update(log_dict_)
                    pbar.set_postfix(**tr_mets, **lr_dict)

            # aggregate required models
            agg_metrics = alg.aggregate()
            comm_load.append(alg.comm_load)
            comm_load_cut.append(alg.comm_load_cut)
            comm_load_weights.append(alg.comm_load_weights)
            comm_breakdown.append(alg.ledger.snapshot())
            if t == 0:
                for k, v in agg_metrics.items():
                    aggregation_metrics[k] = [v]
            else:
                [aggregation_metrics[k].append(v) for k, v in
                agg_metrics.items()]

            tr_str = __print_aggregate_metrics(
                train_metrics, aggregation_metrics, prefix='tr'
            )

            # set models to eval mode and evaluate
            alg.eval_mode()
            acc_, loss_ = alg.evaluate()
            test_acc.append(acc_)
            test_loss.append(loss_)
            log_dict.update({
                'Test/accuracy': acc_,
                'Test/loss': loss_,
                # scalars, not the whole growing list -- every other entry
                # in log_dict is a scalar and wandb needs a time series
                'Test/load': comm_load[-1],
                'Test/load_cut': comm_load_cut[-1],
                'Test/load_weights': comm_load_weights[-1]
            })

            # adjust learning rates for server models in single server runs
            # i.e., sl_single_server, cse_fsl and fsl_sage
            _, log_dict_ = log_and_step_lr_per_round(
                alg, cfg.algorithm.name
            )
            log_dict.update(log_dict_)

            logging.info(
                f' > Round {t}, ' + tr_str +
                f', ts. loss: {loss_:.2f}, ts. acc: {100. * acc_:.2f}%' +
                f', comm cut: {(alg.comm_load_cut / (1024**3)):.3f} GiB' +
                f', comm weights: {(alg.comm_load_weights / (1024**3)):.3f} GiB' +
                f', comm total: {(alg.comm_load / (1024**3)):.3f} GiB.',
            )
            logger_fn(log_dict, step=t)

            # save checkpoints
            if cfg.save and t % cfg.checkpoint_interval == 0:
                checkpointer.save(
                    t, alg.server, alg.clients, {'accuracy': acc_}
                )

            # stop if communication load exceeds threshold
            if alg.comm_load / (1024**2) >= cfg.comm_threshold_mb:
                logging.info(f"Communication budget reached/exceeded @ {t:d} rounds!")
                break

    avg_compute_times = __add_aggregated_compute_times(
        train_metrics, aggregation_metrics
    )
    memory_metrics = alg.memory_report()
    comm_to_target = compute_comm_to_target(
        test_acc, comm_load, comm_load_cut, comm_load_weights,
        float(cfg.get('target_acc', 0.97))
    )
    logging.info(
        f" > Peak client mem: {memory_metrics['peak_client_mem_mb']:.2f} MiB "
        f"(act {memory_metrics['client_act_peak_mem_mb']:.2f}, "
        f"held across cut {memory_metrics['client_mem_held_across_cut_mb']:.2f}), "
        f"peak server mem: {memory_metrics['peak_server_mem_mb']:.2f} MiB."
    )

    return FLResults(
        alg.server, alg.clients, test_loss, test_acc, train_metrics,
        aggregation_metrics, comm_load_cut, comm_load_weights, comm_load,
        avg_compute_times, comm_breakdown, memory_metrics, comm_to_target
    )

# ------------------------------------------------------------------------------
def run_fl_algorithm(
    cfg:DictConfig,
    server: Server,
    clients: List[Client],
    test_loader: DataLoader,
    checkpointer: Checkpointer,
    torch_device,
    logger_fn: Callable,
    warm_start=False
):

    if cfg.algorithm.name == 'fsl_sage' and warm_start:
        ws_cfg = copy.deepcopy(cfg)
        with open_dict(ws_cfg):
            ws_cfg.rounds = 1
            ws_cfg.algorithm.name = 'cse_fsl'
            if 'server_update_interval' not in ws_cfg.algorithm.keys():
                ws_cfg.algorithm.server_update_interval = 5

        logging.info("Warm-starting auxiliary model with CSE-FSL")
        results = _run_fl_algorithm(
            ws_cfg, server, clients, test_loader, checkpointer, torch_device,
            logger_fn
        )
        server = results.server
        clients = results.client_list
        test_loss = results.loss
        test_acc = results.accuracy
        train_metrics = results.train_metrics
        aggregation_metrics = results.aggregation_metrics
        comm_load = results.comm_load
        comm_load_cut = results.comm_load_cut
        comm_load_weights = results.comm_load_weights
        comm_breakdown = results.comm_breakdown
    else:
        test_loss = None
        test_acc = None
        train_metrics = None
        aggregation_metrics = None
        comm_load = None
        comm_load_cut = None
        comm_load_weights = None
        comm_breakdown = None

    return _run_fl_algorithm(
        cfg, server, clients, test_loader, checkpointer, torch_device,
        logger_fn, test_loss, test_acc, train_metrics, aggregation_metrics,
        comm_load, comm_load_cut, comm_load_weights, comm_breakdown
    )

# ------------------------------------------------------------------------------
