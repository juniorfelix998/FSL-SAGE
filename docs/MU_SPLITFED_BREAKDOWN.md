# MU-SplitFed: paper vs port (changed after review)

**Sources:**
- **Paper:** Liang et al., NeurIPS 2025, arXiv:2510.21155.
- **Authors' code** (Johnny-Zip/MU-SplitFed):
  - LLM/OPT-only and single-client, with no aggregation;
  - does not run as published: it imports a `cezo_fl` package that isn't in
    the repository.
- **The vision baseline here** is the third-party ResNet-18 reimplementation
  in HKU-WILL-Lab/HO-SFL (`mu_splitfed_runner.py`), checked step by step
  against the paper.
- **Label:** report it as a reimplementation. Every change is also listed at
  the top of `src/algos/mu_splitfed.py`.

## Adapting the LLM paper to vision
MU-SplitFed uses no LoRA and no LLM-specific loss: it is full-parameter
zeroth-order training (MeZO-style). So it maps directly onto the benchmark's
ResNet-18 split:
- client = conv1 to layer2, server = layer3 to fc (the same middle cut as every
  method);
- cross-entropy loss;
- the cut activation takes the place of the LLM hidden state.

## The paper, step by step, and where it is in the code

| Paper | What happens | Code (`group_step`) | Charged as |
|---|---|---|---|
| p.4 "a subset of clients", p.8 "10 clients … 50% partial" | 5 of 10 clients per step, fair sampling | `_run_grouped_round` in `src/algos/__init__.py` | – |
| Alg.1 L3 | each sampled client pulls the global x_c; the server starts its copy from x_s | pull | `weights.client_down` |
| Eq.4, Alg.1 L5 | client: clean h, h⁺, h⁻ (forward only), sent with the labels | forwards | `cut.act_up` × 3, `cut.labels_up` |
| Eq.5, Alg.1 L6-8 | τ = 2 server steps on that client's copy, on the clean h | server steps | – |
| Eq.6, Alg.1 L9 | one number δ from the post-τ copy, sent back | δ | `cut.scalar_down` |
| Alg.1 L10 | client step along its random direction | client update | – |
| Alg.1 L11, Eq.7 | each client uploads; x ← x + η_g·mean(x_m − x) for client models and server copies | averaging | `weights.client_up`; server copies not charged (inside the Split Server) |

**Eq.7's printed sign:** it would move the model the wrong way. We use the form
from the paper's proof, as both codebases do.

## Changes from the original port

| | Original port | Now |
|---|---|---|
| Aggregation | once per round (24 client steps between aggregations) | after **every step**, as in Alg.1 |
| Client sampling | all 10 clients, one after another | 5 of 10 per step (fair sampling) |
| Client model traffic | one round trip per client per round | one pull + one upload per client batch (L3, L11) |
| Server-copy averaging | charged as Server-model FedAvg | not charged: the paper averages them inside the Split Server |
| Init | ImageNet-pretrained, frozen BatchNorm | random, like every method |
| Memory | all 10 server copies in the server value | global + the copy being trained; other sampled copies listed separately |

## Deliberate differences from the paper
- **Random directions:** Gaussian (both codebases); the paper's theory uses a
  sphere.
- **Fair sampling:** clients are picked at random among those with batches
  left, so every client uses all its data every round. The paper doesn't say
  how its 5 are chosen.
- **Batch 256** for the reported run (the batch every method uses); the paper
  uses 32.
- **Batches of size 1 are skipped** (as in the reference); none occur here.

## Checked by
- **`test/check_accounting.py`:** exact bytes for every category with 3
  clients and K = 2, and every breakdown row against the paper's formula.
- **The breakdown printed at the end of each run:** rows C1–C4, W1, W2 and W5,
  each with ✓.
