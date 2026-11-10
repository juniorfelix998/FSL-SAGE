# FedSplitX: paper vs our code

**Source:** Shin, Ahn, Kang, Kang, *FedSplitX: Federated Split Learning for
Computationally-Constrained Heterogeneous Clients*, arXiv:2310.14579.

**No public code.** This is a **reimplementation** written from the paper alone:
- Algorithm 1 (p.3);
- Sec. 2.1–2.3 (pp.3–4);
- Sec. 3 (p.5);
- Appendix A.1.

Report it as a reimplementation. Code: `src/algos/fedsplitx.py`; the header maps every line of Algorithm 1 to the code.

## The model at the benchmark's middle cut

Appendix A.1 gives three ResNet-18 partition points (depth levels). The paper puts an auxiliary network at each one.

| Depth level | Paper (App. A.1) | Our ResNet-18 | Aux network |
|---|---|---|---|
| 1 | after the MaxPool (conv → BN → MaxPool) | output of `maxpool`, 64 ch, 8×8 | a_1: 64 → 10 |
| 2 | between ResBlock 1 and ResBlock 2 | output of `layer1`, 64 ch, 8×8 | a_2: 64 → 10 |
| 3 | between ResBlock 2 and ResBlock 3 | output of `layer2`, 128 ch, 4×4 | a_3: 128 → 10 |

The benchmark's **middle cut** (client = stem, layer1, layer2) is exactly **depth level 3**:
- **Client:** the stem, layer1, layer2 and all **3** aux networks.
- **Server:** layer3, layer4 and fc. It has **no** aux network; its loss is the model output only (Sec. 2.2, i = d_k+1 … M+1 = only M+1).

Client size: Table 3 gives 689.4K parameters at level 3; ours has 683,072. The paper doesn't describe its stem exactly, so the small gap is not explained.

## Algorithm 1, line by line

| Paper | What happens | Code | Charged as |
|---|---|---|---|
| L3, p.5 "10% of the clients are randomly selected" | P_t: C = 10% of the clients, at random | `begin_round` | – |
| L5 | the client downloads the global client-side model (printed `w^s[:d_k]`, a typo for `w^c`) | `client_step`, first batch of the round | `weights.client_down` (+ `aux_down`) |
| L6 | the server takes its copy for this client from the global server-side model | same | `weights.server_down` |
| L7, L17–18 GetSmashedData | the client forwards **all of D_k** through the **downloaded** model and sends the smashed data and labels | one no-grad pass, sent as **one** upload | `cut.act_up` ×1, `cut.labels_up` ×1 |
| L8, L13–16 Client_Update | E epochs; loss = Σ_{i=1..3} CE(a_i(f[:i](x)), y), no weights; updates the client model and a_[1..3] | every batch | – |
| L9–11 | the server trains its copy for E epochs on the upload; loss = output CE | every batch, on the matching slice of the upload | – |
| L12 heteroavg | average the client-side models (fed server) and server-side models (main server) | `aggregate()` | `weights.client_up` (+ `aux_up`), `weights.server_up` |
| p.4, Fig.1 | no gradient goes back to the client | – | `cut.grad_down` = 0 |
| Sec. 3 (p.5) | inference = ensemble of all aux networks + output; report client-side and full-model accuracy | `evaluate()` | – |

**heteroavg with one depth level.** The paper's heteroavg averages each part of the model over the clients that hold it (Sec. 2.3). Here every client is at level 3, so every client holds every part, and heteroavg becomes a **plain average** over the sampled clients.

## Not specified in the paper: our choice

| Item | Our choice |
|---|---|
| Aux network architecture | global average pool + one Linear layer (650 + 650 + 1,290 = 2,590 params) |
| How the aux networks are averaged and sent | averaged with the client model; their transfers are on separate rows (W3, W4) |
| Loss | cross-entropy |
| Ensemble rule | mean of the heads' probabilities |
| Learning rate, momentum, weight decay | SGD lr 0.01, no momentum, no weight decay, no schedule (the paper says only "SGD") |
| BatchNorm in GetSmashedData | batch statistics (as in training); running statistics restored after, so training starts from the downloaded model |
| Which clients are averaged | the sampled clients P_t |
| How the server walks through the upload | one step per client batch, on the matching slice (B = 256), E = 1 |

## Benchmark settings (differences from the paper's experiments)

| | Paper (p.5) | Here |
|---|---|---|
| Clients, depth levels | 50 clients at 3 depth levels (16 / 16 / 18) | 10 clients, all at level 3 (the middle cut, the same as every method) |
| Clients per round | 10% | 10%, i.e. **1 client per round** |
| Batch | 64 | 256 (every method's batch) |
| Rounds | 1000, 1 local epoch | 11, 1 local epoch |
| Dataset | CIFAR-10/100 | MNIST |
| Server-copy averaging | not discussed as traffic | **charged** as Server-model FedAvg: one download (L6) and one upload (L12) of the server model per sampled client per round |

## Changes from the earlier port

| | Earlier port | Now |
|---|---|---|
| Aux positions | after layer1, layer2, layer3 (split client / server) | after the stem, layer1, layer2 (App. A.1); all on the client at the middle cut |
| Smashed data | one upload per batch, from the model being trained | **one upload of all D_k per round**, from the downloaded model (L7, L18) |
| Clients per round | all 10 | 10% (L3) |
| Downloads | client FedAvg round trip for every client | L5 / L6 downloads and L12 uploads, sampled clients only |
| Optimizer | Adam (harness default) | SGD (p.5) |
| Accuracy | full ensemble only | full ensemble + client-side ensemble logged |

## Checked by
- **`test/check_accounting.py`:** exact bytes for every category with 2 of 3 clients sampled; every breakdown row against the paper's formula; the 3 partition points on the real ResNet-18.
- **The breakdown printed at the end of each run:** rows C1–C3 and W1–W5, each with ✓.
