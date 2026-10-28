# HO-SFL: paper vs original port (changed after review)

HO-SFL (arXiv:2603.14773) is ported from the authors'
`HKU-WILL-Lab/HO-SFL`, `HO_SFL_CV/src/runner/ho_sfl_runner.py`. After review,
the code follows the paper's pseudocode (Alg.1 main loop, Alg.2 ZOScalar,
Alg.3 ClientSync). Every change is also listed at the top of
`src/algos/ho_sfl.py`.

## What the paper does, step by step, and where it is in the code

| Paper | What happens | Code (`client_step`) | Charged as |
|---|---|---|---|
| Alg.3 L3 | client catches up on missed iterations (seeds + averaged scalars) | step 1 | `weights.zo_sync_down` |
| Alg.1 L4 | server sends P seeds | step 2 | `weights.zo_seeds_down` |
| Alg.1 L9-10 | client forward (no backward), sends activation + labels | step 3 | `cut.act_up`, `cut.labels_up` |
| Alg.1 L14-19, Eq.14 | server forward, backward and step on its one model; sends the activation gradient | step 4 | `cut.grad_down` |
| Alg.2, Eq.15-17 | P perturbed forwards; v_p = gradient · (perturbed − clean activation) | step 5 | `weights.zo_scalars_up` |
| Alg.1 L26, Eq.18 | server averages the scalars and sends them back | step 6 | `weights.zo_scalars_down` |
| Alg.1 L27-28, Eq.19 | client update g = (1/(Pμ)) Σ v̄_p u_p, AdamW step | step 7 | – |
| Alg.1 L2 | server keeps a history of seeds + scalars for catch-ups | `server_side_extra_memory` | server memory line |

- **Sizes:** tensors at their actual bytes; 4 B per seed and per scalar, as
  in the authors' `communicator.py`.
- **Model weights:** never sent (paper p.5–6, Table 4).

## Changes from the original port

| | Original port | Now |
|---|---|---|
| Iteration | buffered all 240 batches of a round, then **one** server step and **one** client step per round (11 of each in 11 rounds) | **one full iteration per batch**: 240 server and client steps per round, as in Alg.1 and the authors' runner |
| Aggregation traffic | one lump `weights.scalars` charge per round | seeds, scalars up, scalars down and catch-up charged separately, per iteration |
| Server memory | catch-up history not counted | listed (Alg.1 L2) |

## Deliberate differences from the paper (benchmark fairness)

- **Random init**, like every method in the benchmark. The paper starts from
  ImageNet-pretrained ResNet-18 with frozen BatchNorm. With random weights the
  client's BatchNorm trains normally. Set `use_pretrained` / `freeze_bn` to
  true in `src/hydra_config/algorithm/ho_sfl.yaml` to reproduce the paper's
  setting.
- **No random client sampling.** The paper picks a random S_t (authors: 10 of
  100 clients). Here S_t is the client whose batch it is: clients take turns,
  and each uses all its data every round. |S_t| = 1 is a valid case of Alg.1,
  and the catch-up traffic that sampling implies is still charged (row Z4).
- **Benchmark settings.** Batch 256, 10 clients, and one round = one pass over
  each client's data (the authors use batch 32, 100 clients, and one batch per
  round to a 120k-sample budget).

## Checked by
- **`test/check_accounting.py`:** exact bytes for every HO-SFL category, and
  every breakdown row's count against the paper's formula.
- **The breakdown printed at the end of each run:** rows C1–C3, W1/W2, Z1–Z4
  and W5, each with ✓.
