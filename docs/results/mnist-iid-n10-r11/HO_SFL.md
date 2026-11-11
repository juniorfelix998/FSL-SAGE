# HO-SFL: MNIST IID, 10 clients, 11 rounds

**Paper:** HO-SFL, arXiv:2603.14773.

**Code:** ported from the authors' `HO_SFL_CV/src/runner/ho_sfl_runner.py`, checked step by step against the paper's Algorithm 1 (see [`docs/HO_SFL_BREAKDOWN.md`](../../HO_SFL_BREAKDOWN.md)).

**Run:**
- Colab T4, ResNet-18, middle cut, seed 200;
- **batch 256** (the batch every method uses), random init;
- P = 5, μ = 1e-3;
- **3 of the 10 clients sampled per iteration** (`sampled_clients: 3`).

```
python main.py rounds=11 save=False device=cuda algorithm=ho_sfl \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true \
    model.client.batch_size=256
```

| Result | Value |
|---|---:|
| Test accuracy (after the 11th round) | **97.80%** |
| Communication, total | **10317.93 MiB** (Cut 10317.54 + Weights 0.39) |
| Client memory | **49.44 MiB** |
| Server memory, per device (table value) | 180.27 MiB |
| Server memory, total for all 10 clients | **180.31 MiB** |
| Latency | **673.42 s** |

Units: MiB = 1024² bytes (the benchmark table's "MB").

## How HO-SFL works here

Each **iteration** takes one batch from each of 3 clients (paper Alg.1):

1. **Sample.** 3 clients are picked at random among those that still have batches left this round, so every client uses all its data every round, like the other methods.
2. **Catch-up.** Each picked client catches up on iterations it missed: seeds + averaged scalars (Alg.3).
3. **Forward and upload.** The server sends 5 seeds. Each client runs a forward pass only (**no backward**) and sends its activations and labels up.
4. **Server step.** The server runs forward + backward for each of the 3 batches and sends each client **its own** activation gradient λ (Alg.1 L16). It then takes **one** step with the gradient averaged over the 3 (L18).
5. **Scalars.** Each client runs 5 more forwards, with its weights nudged along 5 random directions from the seeds. It sends 5 numbers: how much each nudge agrees with λ (Eq.17).
6. **Average and update.** The server averages the 3 clients' numbers and sends 5 back. The client model takes one AdamW update from them (Eq.19).

**No model weights are ever sent.** The clients stay identical by replaying the same seeds and numbers.

## Inputs

- **Batches per client per round:** 24 (23 × 256 + 1 of 112).
- **Client batches in the run:** 10 × 11 × 24 = **2,640**.
- **Iterations (updates) in the run:** **900**. Most have 3 clients. Near the end of a round fewer than 3 clients have batches left, so a few have 1 or 2.
- **One activation of one image:** 128 × 4 × 4 × 4 B = 8,192 B, so one full batch is 2.00 MiB.
- **One seed or scalar:** 4 B.

## Communication: 10317.93 MiB

| What is sent | Count | MiB |
|---|---:|---:|
| Activations, client → server (L10) | 2,640 | 5156.25 |
| Labels (L10) | 2,640 | 5.04 |
| **Activation gradient λ, server → client** (L19) | 2,640 | **5156.25** |
| Client model up / down | 0 | 0.00 |
| 5 scalars up, 20 B per client batch (L23) | 2,640 | 0.05 |
| 5 seeds down (L4) | 2,640 | 0.05 |
| 5 averaged scalars down (L26) | 2,640 | 0.05 |
| Catch-up for missed iterations (Alg.3) | 1,748 | 0.24 |
| **Total** | Cut 10317.54 + Weights 0.39 | **10317.93** |

**Where the bytes come from:**
- **Activations and gradients:** 660,000 images (10 clients × 6,000 × 11 rounds) × 8,192 B = 5,406,720,000 B = 5156.25 MiB each way.
- **Labels:** 660,000 × 8 B = 5.04 MiB.
- **Scalars:** 2,640 × 5 × 4 B = 52,800 B = 0.05 MiB per row.
- **Catch-up:** each missed iteration costs 5 × (4 B seed + 4 B scalar) = 40 B. The count is random, because it depends on which clients are picked; it is checked against a replay of the actual sampling sequence (✓).

**Model sharing costs 0.39 MiB in total**, against 574.88 MiB for one FedAvg round trip per client per round.

### Compared with the paper's vision run

The paper's vision experiment (Appendix C.5, Table 4; Table 3) uses the same model and the same split as ours. The differences are the dataset, the number of clients and how long it trains.

| Setting | Paper | Here |
|---|---|---|
| Dataset | CIFAR-10 IID (32 × 32 × 3) | MNIST IID, resized to 32 × 32 × 3 |
| Model and split | ResNet-18, split between the 2nd and 3rd residual blocks | ResNet-18, middle cut (the same split) |
| Clients | 100, 10 sampled per round | 10, 3 sampled per iteration |
| Images per client | 500 | 6,000 |
| Batch | 32 | 256 |
| P, μ | 5, 1e-3 | 5, 1e-3 |
| Initialization | ImageNet-pretrained, frozen BatchNorm | random |
| Training length | 160,000 images processed (p.7) | 660,000 images processed (11 rounds) |

| Result | Paper | Here |
|---|---:|---:|
| Activations up | 1.22 GB | 5.04 GiB |
| Gradients down | 1.22 GB | 5.04 GiB |
| Seeds + scalars | 0.0042 GB | 0.0004 GiB |
| **Communication, total** | **2.44 GB** | **10.08 GiB** |
| **Communication per image** | **≈ 16,400 B** | **≈ 16,390 B** |
| Test accuracy | 75.0% (CIFAR-10) | 97.80% (MNIST) |

- **Per image, the communication is the same.**
  - **Why:** both send one activation up and one gradient down for every image, and with 32 × 32 inputs and the same split one activation is 8,192 B in both.
  - **Paper's units:** its "GB" is 1024³ bytes; 160,000 × 8,192 B = 1.2207 × 1024³, its 1.22.
- **The total is 4.125× the paper's**, because we process 4.125× as many images (660,000 / 160,000); 2.44 × 4.125 = 10.07.
- **Labels: we charge them, the paper's table has no row for them (5.04 MiB here).**
  - **Why they cross the cut:** the loss is computed on the server. The server needs the true answer for every image, so the client sends the labels with the activation (Alg.1 L10: "Client sends (z<sub>m</sub>, y<sub>m</sub>) to Server"; the loss in L14 uses y<sub>m</sub>).
  - **Why 8 B per label:** each label is a 64-bit integer, PyTorch's standard type for class labels (`y.long()`). One batch of 256 labels is 2,048 B; 660,000 labels are 5,280,000 B = 5.04 MiB.
  - **Same rule for every method:** labels are charged whenever the server computes the loss (`charge_cut_labels`, `src/algos/__init__.py:142-152`; called in `ho_sfl.py:342`).
  - **Why the paper may leave them out:** at its scale they are 160,000 × 8 B = 1.28 MB, about 0.001 GB, too small for its table. Here they are 0.05% of the total.
- **Seeds and scalars: the paper sends more (≈ 0.004 GB) than we do (0.39 MiB ≈ 0.0004 GiB), for two reasons.**
  - **1. More updates per image.** Seeds and scalars are sent once per client batch: 5 seeds down, 5 scalars up and 5 averages down, 20 B each.
    - **Paper, batch 32:** 160,000 images / 32 = **5,000 client batches**, one per 32 images.
    - **Here, batch 256:** **2,640 client batches** for 660,000 images, one per 250 images.
    - Per image, the paper makes 256 / 32 = **8× as many** updates, so it sends 8× as many seeds and scalars per image. In total, 5,000 vs 2,640 client batches.
  - **2. More catch-up.** A client that sat out must download the 5 seeds and 5 averaged scalars of every iteration it missed (40 B per missed iteration, Alg.3).
    - **Paper:** 10 of 100 clients per iteration, so a client takes part about 1 iteration in 10 and has missed about 9 each time it returns: ≈ 360 B per return.
    - **Here:** 3 of 10 per iteration, so a client misses about 2 between turns (1,748 catch-ups in total, 0.24 MiB).
  - **Rough count for the paper:**
    - seeds, scalars up and averages down: 3 × 5,000 × 20 B ≈ 0.3 MB;
    - catch-up: ≈ 5,000 × 9 × 40 B ≈ 1.8 MB;
    - ≈ 2 MB in total, the same order as its reported 0.0002 + 0.004 GB. The paper does not give its exact counts or byte sizes, so the figure cannot be matched exactly.
  - In both, this traffic is under 0.2% of the total.
- **Accuracy is not comparable across the two.** MNIST is a much easier task than CIFAR-10. In the paper, HO-SFL reaches 75.0% against SFL's 77.5% and FSL-SAGE's 62.1% (Table 3). Here it reaches 97.80% against FSL-SAGE's 98.96% and CSE-FSL's 99.13%.
- **The ranking on communication differs.**
  - **In the paper,** HO-SFL (2.44 GB) is below SFL (4.20) and FSL-SAGE (3.71). Their cost there is mostly model transfers: each client holds only 500 images, so a model goes up and down after every ~500 images (0.84 GB of SFL model uploads = 329 uploads for 160,000 images).
  - **Here,** each client trains on 6,000 images between transfers (110 uploads for 660,000 images), so model traffic per image is 12× smaller.
  - HO-SFL's cost per image is fixed, so in this setting it sends more than FSL-SAGE (1642.46 MiB) and CSE-FSL (3222.21 MiB).

Code:
- iteration: `src/algos/ho_sfl.py` `group_step` (steps 1–7 numbered in the code);
- fair sampling: `src/algos/__init__.py` `_run_grouped_round`.

## Memory

| Client: 49.44 MiB | MiB |
|---|---:|
| Client model: params 2.61 + grads 2.61 + AdamW 5.21 | 10.43 |
| Working memory at its peak (during a nudged forward; see below) | 39.01 |
| **Total** | **49.44** |

The client never runs a backward pass, so it keeps no activations for backprop (0.00). Its working memory peaks inside a nudged forward pass (step 5):

| Alive at that moment | Arithmetic | MiB |
|---|---|---:|
| Input batch | 256 × 3 × 32 × 32 × 4 B | 3.00 |
| Its clean activation a (kept for the scalars) | 256 × 8,192 B | 2.00 |
| The downloaded gradient λ | 256 × 8,192 B | 2.00 |
| conv1's output and bn1's output, alive together | 2 × 256 × 64 × 16 × 16 × 4 B | 32.00 |
| Labels + saved BatchNorm statistics | 2,048 B + 7,760 B | 0.01 |
| **Working peak** | | **39.01** |

| Server | MiB |
|---|---:|
| Server model: params 40.08 + grads 40.05 + AdamW 80.10 | 160.22 |
| Working memory: activations for backprop 14.04 + the 3 sampled clients' uploads, held until the one server step (3 × 2.00) + small outputs 0.01 | 20.05 |
| **Per device (table value)** | **180.27** |
| + history of seeds and scalars for catch-ups (900 iterations × 40 B) | 0.03 |
| **Total for all 10 clients** | **180.31** |

- **Held-across-cut = 0:** the client keeps no activations while the server works.
- **System peak:** 59.01 MiB live (both sides at one moment), 229.67 with model and optimizer bytes.

## Time

| | Value |
|---|---:|
| Latency | 673.42 s |
| Client compute (mean per client over 11 rounds) | 18.13 s |
| Server compute (mean per client over 11 rounds) | 17.93 s |

Each client batch needs 6 client forwards (1 clean + 5 nudged) and a share of one server step.

## Accuracy

**97.80%**, close to CSE-FSL (99.13%) and FSL-SAGE (98.96%). The client learns from 5 random directions per update instead of an exact gradient, so it is slightly lower.

## All 10 clients in every iteration (K = 10)

The same run with every client taking part in every iteration (the paper's full-participation case). The reported HO-SFL result stays K = 3; this run shows what the number of sampled clients changes.

```
python main.py rounds=11 save=False device=cuda algorithm=ho_sfl \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true \
    model.client.batch_size=256 algorithm.sampled_clients=10
```

| | K = 3 (reported) | K = 10 |
|---|---:|---:|
| Iterations in the run | 900 | **264** (24 per round) |
| Images per update | 3 × 256 = 768 | 10 × 256 = 2,560 |
| Activations up + λ down + labels | 10317.54 | **10317.54** |
| Seeds + scalars up + averages down | 0.15 | **0.15** |
| Catch-up for missed iterations | 0.24 (1,748 transfers) | **0** |
| **Communication, total** | **10317.93** | **10317.69** |
| **Test accuracy** | **97.80%** | **97.93%** |
| Client memory | 49.44 | **49.44** |
| Server working memory | 20.05 = 14.04 + 3 × 2.00 uploads | **34.05** = 14.04 + 10 × 2.00 uploads |
| Server, per device (table value) | 180.27 | **194.27** |
| Seed/scalar history | 0.03 (900 × 40 B) | 0.01 (264 × 40 B) |
| System peak, live / with static | 59.01 / 229.67 | 122.03 / 292.68 |
| **Latency** | **673.42 s** | **670.71 s** |
| Compute, client / server (mean per client) | 18.13 / 17.93 s | 13.86 / 23.42 s |

Both runs: every breakdown row ✓.

- **Communication is the same.** HO-SFL sends an activation up and a λ down for every image, whatever K is. The only saving is the catch-up traffic: with every client in every iteration, no client ever falls behind, so ClientSync (Alg.3) never runs.
- **Accuracy is about the same.** K = 10 makes 3.4× fewer updates (264 vs 900), but each one averages the scalars of 10 clients and the server gradient of 10 batches, so it is less noisy. On MNIST the two effects roughly cancel. With one seed, the 0.13-point difference is within run-to-run variation.
- **The server needs more memory.** It holds every sampled client's upload until its one step: 10 × 2.00 MiB instead of 3 × 2.00, so 194.27 vs 180.27 MiB. At the busiest moment the whole system holds about twice as much (122.03 vs 59.01 MiB live), because all 10 clients hold their batch, activation and λ at once.
- **Client memory is unchanged.** Each client still holds one batch, its activation and its λ.
- **Latency is the same.** The forward and backward work per image is identical; only the number of optimizer steps changes.
  - The client and server compute times shift: the client rebuilds and applies its update 264 times instead of 900.
  - These timers measure CPU time without waiting for the GPU, so time moves between the two sides. Compare latency, not the compute times.

## Differences from the paper (deliberate, for a fair benchmark)

- **Random init**, like every other method. The paper uses ImageNet-pretrained weights with frozen BatchNorm; here the client's BatchNorm trains normally.
- **Fair sampling.** 3 of 10 clients per iteration, as in the paper's language experiments. They are picked at random among clients with batches left, so every client uses all its data every round (the paper samples freely).
- **Batch 256**, the batch every method uses (the paper: 32).
- **10 clients**, and one round is one pass over each client's data (the paper's vision runs: 100 clients, 10 sampled, a 160k-image budget).
