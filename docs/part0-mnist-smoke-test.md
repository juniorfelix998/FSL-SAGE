# Part 0 — MNIST Smoke Test: Process, Results, and What It Proves

This document records the smoke-test phase of the "What Crosses the Cut?" benchmark
(see `CLAUDE.md`), completed on 2026-08-24. Its purpose is to confirm the FSL-SAGE
harness's shared training loop, data pipeline, and metric logger genuinely
generalize across methods and datasets *before* the harness is trusted for the
main CIFAR-10/CIFAR-100 benchmark table.

## 1. Environment setup

The repo ships `conda_env.yaml`, but it was exported from a **Linux + CUDA**
machine: exact conda build hashes (e.g. `h5eee18b_0`) and the `nvidia-*-cu12`/
`triton` pip packages don't exist for macOS at all, so `conda env create -f
conda_env.yaml` cannot succeed on a Mac regardless of network conditions.

Resolution (macOS arm64, no prior conda installation on the machine):
```bash
brew install --cask miniconda
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n sage python=3.12.7 -y
conda activate sage
pip install torch==2.5.1 torchvision==0.20.1 hydra-core==1.3.2 hydra-joblib-launcher==1.2.0 \
  omegaconf==2.3.0 wandb==0.19.3 numpy==2.1.3 pandas==2.2.3 scipy==1.14.1 matplotlib==3.9.2 \
  h5py==3.12.1 pyyaml==6.0.2 tqdm==4.67.0 requests==2.32.3 pillow==11.0.0 prettytable==3.12.0 \
  joblib==1.4.2 antlr4-python3-runtime==4.9.3 gitpython==3.1.43
```
Same pinned versions as `conda_env.yaml`'s pip section, minus the CUDA-only packages
that don't apply without an NVIDIA GPU. **On the Linux/CUDA machines used for the
actual benchmark runs, `conda env create -f conda_env.yaml` as documented in the
README should work unmodified** — this workaround is specific to development on
non-Linux/non-CUDA machines.

## 2. Step 1 — run the repo unmodified on its default example

Default Hydra config (`src/hydra_config/config.yaml`): `dataset=cifar10`,
`model=simple_conv`, `algorithm=fsl_sage`, `rounds=200`, `num_clients=10`,
`save=True` (which calls `wandb.login()`).

**Bug found in the repo's own defaults** (not introduced by this work):
`src/hydra_config/model/simple_conv.yaml` never defines `client.optimizer`/
`server.optimizer` blocks (unlike every other model config, e.g. `resnet18.yaml`),
so `src/models/__init__.py:124` crashes with
`ConfigAttributeError: Key 'optimizer' is not in struct` the moment the literal
default config is run. This needs a fix (add the missing optimizer blocks) before
`simple_conv` can be used in any real benchmark run.

Workaround used for the smoke test — CLI overrides only, no source edits:
```bash
cd src
python main.py rounds=3 save=False device=cpu model=resnet18
```
(`model=resnet18` is also what the README's own example command uses;
`device=cpu`/`save=False` are dev-machine-only conveniences — no CUDA on this
Mac, and `save=False` avoids requiring a wandb login for a throwaway run.)

**Result:** 3 rounds of FSL-SAGE / CIFAR-10 / ResNet-18 / 10 clients / IID
completed cleanly. Test accuracy reached 44.69%; cumulative communication was
logged as a single combined `comm: 0.47 GiB` figure (see Step 2 — this is the
gap that motivated the CUT/WEIGHTS split).

## 3. Step 2 — the communication-counting hook, and the CUT/WEIGHTS split

### What was there before

- **Core primitive** — `src/utils/utils.py:17-27`, `calculate_load(model)`:
  sums `param.nelement() * param.element_size()` across a model's parameters
  and buffers. Returns raw **bytes** despite the variable name `size_all_mb`
  (the `/1024**2` conversion is commented out).
- **Accumulator** — `FLAlgorithm.comm_load` (`src/algos/__init__.py`), a single
  running float per algorithm instance, never reset across a run.
- **Per-batch cut traffic** — SplitFedv1/v2 (`src/algos/baselines.py`) add both
  the activation upload and the gradient-of-activation download every batch
  (`smashed_data.numel()*element_size()` and `smashed_data.grad.numel()*
  element_size()`). CSE-FSL and FSL-SAGE only add the activation upload (no
  gradient download — both use a client-side auxiliary/surrogate model instead
  of waiting on a true server gradient).
- **Weight-transfer traffic** — the shared `aggregate_clients()` base method
  adds `2 * calculate_load(...)` (upload + download) once per client per round
  for FedAvg-style weight averaging; FSL-SAGE additionally charges the
  aligned-auxiliary-model transfer back to the client, and CSE-FSL additionally
  charges its own auxiliary-model aggregation.
- **The gap:** all of the above — true per-batch cut traffic *and* full-model
  weight transfers — landed in the **same single scalar**. There was no way to
  tell, from the logged number alone, how much of a method's communication cost
  was inherent to the split (cut) versus incidental to federated aggregation
  (weights) — exactly the distinction CLAUDE.md's four-metrics spec requires.

### What was changed

- `FLAlgorithm` gained two real accumulators, `comm_load_cut` and
  `comm_load_weights`; `comm_load` became a read-only `@property` returning
  their sum, preserving every existing call site that reads `alg.comm_load`.
- Every per-batch activation/gradient line (the true cut traffic) was
  redirected to `comm_load_cut`; every full-model aggregation/alignment
  transfer was redirected to `comm_load_weights`.
- Both new accumulators were threaded through the round loop, `FLResults`,
  the wandb `log_dict` (`Test/load_cut`, `Test/load_weights`), the warm-start
  continuation path (CSE-FSL warm-starting FSL-SAGE), and `main.py`'s saved
  `results.json`/`metrics.pt` (the original `comm_load` key was kept for
  backward compatibility with `inference/plot_results.py`).
- Per-round log lines now read:
  `comm cut: X GiB, comm weights: Y GiB, comm total: Z GiB` (previously just
  `comm: Z GiB`).

**Validation:** a 2-round SplitFedv2/CIFAR-10/ResNet-18 run logged
`comm cut: 1.526 GiB, comm weights: 0.020 GiB, comm total: 1.546 GiB` —
cut + weights sums exactly to total.

## 4. Step 3 — adding MNIST as a dataset

- New `src/hydra_config/dataset/mnist.yaml` (`name: mnist`, `num_classes: 10`,
  mirrors `cifar10.yaml`).
- New branch in `src/datasets/__init__.py::get_dataset()`: resizes to 32×32
  and repeats the single grayscale channel to 3 via
  `transforms.Grayscale(num_output_channels=3)`, so the exact same ResNet
  backbone used for CIFAR is reused with **zero model-code changes**. Downloads
  to `../datas/mnist`.
- No new `dataset_model` override file was needed (unlike FEMNIST's 1-channel
  case): `resnet18_sl_client` already defaults to `n_channels=3` and
  `resnet18.yaml`'s server already defaults to `num_classes=10`, both of which
  already match channel-repeated MNIST.

## 5. Step 4 — one method run to completion on MNIST

```bash
python main.py rounds=2 num_clients=2 save=False device=cpu model=resnet18 \
  algorithm=sl_single_server dataset=mnist
```
Result: exit code 0. Test accuracy reached **98.80%** (vs. 44–58% on CIFAR-10
under the same setup in Step 1) — the large accuracy jump is itself evidence
the pipeline is genuinely training on MNIST data, not silently falling back to
a cached CIFAR loader. `comm cut: 1.831 GiB, comm weights: 0.020 GiB,
comm total: 1.851 GiB`.

## 6. Step 5 — confirming all metrics log to a file

Added to `src/main.py`:
- **Latency** — a module-level `_APP_START_TIME` timestamp (set at import
  time); `latency_s = time.time() - _APP_START_TIME` computed and logged at
  the end of the run.
- **Memory** — `get_peak_memory_mb()` reads
  `resource.getrusage(resource.RUSAGE_SELF).ru_maxrss` (platform-aware: this
  field is bytes on macOS, KB on Linux). When a CUDA device is in use,
  `torch.cuda.max_memory_allocated()` is also tracked separately
  (`peak_cuda_memory_mb`), reset via `torch.cuda.reset_peak_memory_stats()` at
  run start — relevant for the eventual GPU-based main benchmark runs, even
  though this smoke test ran CPU-only.
- Both fields added to `results.json` and `metrics.pt`.

**Validation run** (`save=True`, `WANDB_MODE=offline` to avoid requiring live
wandb credentials for a throwaway run):
```bash
WANDB_MODE=offline python main.py rounds=2 num_clients=2 save=True device=cpu \
  model=resnet18 algorithm=sl_single_server dataset=mnist
```
Saved to `saves/sl_single_server/resnet18/mnist-iid/R2m2E1B256-seed200/260824-194622/`.

`output.log` (human-readable):
```
Round 0, ... ts. acc: 97.98%, comm cut: 0.916 GiB, comm weights: 0.010 GiB, comm total: 0.926 GiB.
Round 1, ... ts. acc: 98.80%, comm cut: 1.831 GiB, comm weights: 0.020 GiB, comm total: 1.851 GiB.
Total latency: 538.49s
Peak process memory: 1060.78 MiB
[NOTICE] Saved results to '.../results.json'.
```

`results.json` (machine-readable, all four required metrics present):
```json
"test_acc": [0.9798, 0.988],
"comm_load": [994000192.0, 1988000384.0],
"comm_load_cut": [983040000.0, 1966080000.0],
"comm_load_weights": [10960192.0, 21920384.0],
"latency_s": 538.49,
"peak_memory_mb": 1060.78
```
Verified `comm_load_cut + comm_load_weights == comm_load` exactly at every round.

## 7. Cross-method validation: CSE-FSL vs. SplitFedv2 on identical settings

To sanity-check that the CUT/WEIGHTS split produces *meaningfully different*,
not just structurally-present, numbers across methods, CSE-FSL was run with
every setting identical to the Step 5 SplitFedv2 baseline (MNIST, ResNet-18,
IID, 2 rounds, 2 clients, batch 256, seed 200):
```bash
WANDB_MODE=offline python main.py rounds=2 num_clients=2 save=True device=cpu \
  model=resnet18 algorithm=cse_fsl dataset=mnist seed=200
```
Saved to `saves/cse_fsl/resnet18/mnist-iid/R2m2E1B256q5-seed200/260824-201250/`.

| Metric (round 2, cumulative) | SplitFedv2 | CSE-FSL |
|---|---|---|
| `comm_load_cut` | 1,966,080,000 B | 201,326,592 B |
| `comm_load_weights` | 21,920,384 B | 89,275,648 B |
| `comm_load` (total) | 1,988,000,384 B | 290,602,240 B |
| cut : weights ratio | ~99 : 1 | ~69 : 31 |
| `test_acc` | 98.80% | 98.41% |
| `latency_s` | 538.5 | 350.6 |
| `peak_memory_mb` | 1060.8 | 1113.4 |

CSE-FSL used **~7x less total communication** than SplitFedv2 — expected,
since it only exchanges true gradients every 5th iteration instead of every
batch — and its cut:weights ratio is visibly different (weights are a much
larger relative share, because of its per-round auxiliary-model aggregation).
Wall-clock latency dropped accordingly. This is direct evidence the split
instrumentation is capturing a real, method-dependent signal rather than a
constant offset.

## 8. What this smoke test proves for the benchmark paper

1. **The shared harness design holds up.** One training loop
   (`_run_fl_algorithm` in `src/algos/__init__.py`), one data pipeline
   (`src/datasets/__init__.py`), and one metric logger now run FSL-SAGE,
   SplitFedv2, and CSE-FSL against two different datasets (CIFAR-10 and
   MNIST) without any per-method or per-dataset special-casing in the loop
   itself — exactly the "fair, common measurement layer" the paper's
   contribution rests on.
2. **All four required metrics are now uniformly instrumented and file-logged**
   for every method: communication is split into CUT and WEIGHTS (previously
   a single undifferentiated scalar — this was the key missing piece), plus
   accuracy, latency, and peak memory. Both a human-readable (`output.log`)
   and machine-readable (`results.json`/`metrics.pt`) record are produced per
   run, ready for the plotting/ranking scripts in `inference/`.
3. **The CUT/WEIGHTS split is not cosmetic — it changes what's comparable.**
   The CSE-FSL vs. SplitFedv2 comparison shows the two methods trade off cut
   traffic against weight traffic very differently; ranking methods on a
   single combined `comm_load` number (the pre-existing behavior) would have
   obscured *why* one method communicates less, which matters for the paper's
   analysis, not just its headline ranking.
4. **A real bug in the base repo was found and documented** (broken
   `simple_conv` optimizer config) before it could silently break a benchmark
   run — worth fixing (or deprioritizing `simple_conv` as a model choice)
   before the main CIFAR/CIFAR-100 runs.
5. **New datasets plug in cheaply.** Adding MNIST took one new Hydra config
   file and one new `elif` branch in the dataset loader, with zero changes to
   any model/backbone code — a good sign for onboarding CIFAR-100 and the
   later regression datasets without destabilizing the shared loop.

## 9. Known limitations / follow-ups for later phases

- **Peak memory is whole-process, not split by role.** This harness runs
  client(s), server, and auxiliary models in a single simulated process, so
  `peak_memory_mb` (via `resource.getrusage`) captures the combined footprint,
  not a true separate "client memory" vs. "server memory" figure. If the main
  benchmark needs that granularity, it will require bracketing memory
  snapshots around `client_step` vs. the server-side forward/backward calls
  specifically — a larger change than this smoke test warranted.
- **`simple_conv` model config is broken** (see Step 1) — needs the missing
  `optimizer` blocks added before it can be used as a benchmark model choice.
- **`src/algos/fed_rolex.py` is an empty stub** — not implemented, not
  currently part of the benchmark method list, no action needed unless it
  becomes relevant later.
- **All timings here are CPU-only, single dev machine, 2-client toy runs** —
  useful only to confirm the instrumentation itself works end-to-end, not as
  representative absolute numbers for the GPU-based main benchmark.
- **CIFAR-10's first auto-download took ~28 minutes** on this machine's
  network — worth pre-caching datasets on the actual benchmark hardware to
  avoid repeated slow downloads across seeds/methods.

## 10. Files changed

| File | Change |
|---|---|
| `src/algos/__init__.py` | `comm_load` split into `comm_load_cut`/`comm_load_weights` (+ backward-compatible property); threaded through round loop, `FLResults`, wandb logging, warm-start path |
| `src/algos/baselines.py` | Per-batch activation/gradient lines → `comm_load_cut` |
| `src/algos/cse_fsl.py` | Activation upload → `comm_load_cut`; auxiliary-model aggregation → `comm_load_weights` |
| `src/algos/fsl_sage.py` | Activation upload → `comm_load_cut`; aligned auxiliary-model transfer → `comm_load_weights` |
| `src/hydra_config/dataset/mnist.yaml` | New — MNIST dataset config |
| `src/datasets/__init__.py` | New `mnist` branch in `get_dataset()` |
| `src/main.py` | Added `latency_s`, `peak_memory_mb` (+ `peak_cuda_memory_mb` when on GPU) tracking; extended `results.json`/`metrics.pt` |
