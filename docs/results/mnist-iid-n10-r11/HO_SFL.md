# HO-SFL: MNIST IID, 10 clients, 11 rounds

**Paper:** HO-SFL, arXiv:2603.14773.

**Code:** ported from the authors' `HO_SFL_CV/src/runner/ho_sfl_runner.py`, checked step by step against the paper's Algorithm 1 (see [`docs/HO_SFL_BREAKDOWN.md`](../../HO_SFL_BREAKDOWN.md)).

**Both runs:**
- Colab T4, ResNet-18, middle cut, seed 200;
- **batch 256** (the batch every method uses), random init;
- P = 5, μ = 1e-3.

They differ only in how many clients take part in each iteration (K):

| Run | Clients per iteration | Command |
|---|---|---|
| **Main (reported)** | **all 10 (K = 10)**, the paper's full-participation case | `... model.client.batch_size=256 algorithm.sampled_clients=10` |
| Reference | 3 of the 10 (K = 3, `sampled_clients: 3` in the yaml) | `... model.client.batch_size=256` |

```
python main.py rounds=11 save=False device=cuda algorithm=ho_sfl \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true \
    model.client.batch_size=256 algorithm.sampled_clients=10      # drop the last option for K = 3
```

| Result | **K = 10 (main)** | K = 3 (reference) |
|---|---:|---:|
| Test accuracy (after the 11th round) | **97.93%** | 97.80% |
| Communication, total | **10317.69 MiB** (Cut 10317.54 + Weights 0.15) | 10317.93 MiB (Cut 10317.54 + Weights 0.39) |
| Client memory | **49.44 MiB** | 49.44 MiB |
| Server memory, per device (table value) | **194.27 MiB** | 180.27 MiB |
| Server memory, total for all 10 clients | **194.28 MiB** | 180.31 MiB |
| Latency | **670.71 s** | 673.42 s |

Both runs: every breakdown row ✓. Units: MiB = 1024² bytes (the benchmark table's "MB").

## How HO-SFL works here

Each **iteration** takes one batch from each of K clients (paper Alg.1):

1. **Sample.** K clients take part. With K = 10 that is every client, every iteration. With K = 3 they are picked at random among those that still have batches left this round, so every client still uses all its data every round, like the other methods.
2. **Catch-up.** A client that missed iterations catches up first: seeds + averaged scalars of each missed iteration (Alg.3). With K = 10 no client ever misses one.
3. **Forward and upload.** The server sends 5 seeds. Each client runs a forward pass only (**no backward**) and sends its activations and labels up.
4. **Server step.** The server runs forward + backward for each of the K batches and sends each client **its own** activation gradient λ (Alg.1 L16). It then takes **one** step with the gradient averaged over the K batches (L18).
5. **Scalars.** Each client runs 5 more forwards, with its weights nudged along 5 random directions from the seeds. It sends 5 numbers: how much each nudge agrees with λ (Eq.17).
6. **Average and update.** The server averages the K clients' numbers and sends 5 back. The client model takes one AdamW update from them (Eq.19).

**No model weights are ever sent.** The clients stay identical by replaying the same seeds and numbers.

## Inputs

| | K = 10 (main) | K = 3 (reference) |
|---|---:|---:|
| Batches per client per round | 24 (23 × 256 + 1 of 112) | 24 |
| Client batches in the run | 10 × 11 × 24 = **2,640** | 2,640 |
| Iterations (updates) in the run | **264** (24 per round) | 900 |
| Images per update | 10 × 256 = 2,560 | 3 × 256 = 768 |

- **Why 264:** with K = 10 each iteration takes one batch from every client, so a round of 24 batches each is 24 iterations; 11 × 24 = 264.
- **Why 900 for K = 3:** most iterations have 3 clients. Near the end of a round fewer than 3 clients have batches left, so a few have 1 or 2.
- **One activation of one image:** 128 × 4 × 4 × 4 B = 8,192 B, so one full batch is 2.00 MiB.
- **One seed or scalar:** 4 B.

## Communication

| What is sent | Count | K = 10 (main) MiB | K = 3 (reference) MiB |
|---|---:|---:|---:|
| Activations, client → server (L10) | 2,640 | 5156.25 | 5156.25 |
| Labels (L10) | 2,640 | 5.04 | 5.04 |
| **Activation gradient λ, server → client** (L19) | 2,640 | **5156.25** | **5156.25** |
| Client model up / down | 0 | 0.00 | 0.00 |
| 5 scalars up, 20 B per client batch (L23) | 2,640 | 0.05 | 0.05 |
| 5 seeds down (L4) | 2,640 | 0.05 | 0.05 |
| 5 averaged scalars down (L26) | 2,640 | 0.05 | 0.05 |
| Catch-up for missed iterations (Alg.3) | K = 10: **0** · K = 3: 1,748 | **0.00** | 0.24 |
| **Total** | | **10317.69** (Cut 10317.54 + Weights 0.15) | 10317.93 (Cut 10317.54 + Weights 0.39) |

**Where the bytes come from:**
- **Activations and gradients:** 660,000 images (10 clients × 6,000 × 11 rounds) × 8,192 B = 5,406,720,000 B = 5156.25 MiB each way.
- **Labels:** 660,000 × 8 B = 5.04 MiB.
- **Scalars:** 2,640 × 5 × 4 B = 52,800 B = 0.05 MiB per row.
- **Catch-up:**
  - each missed iteration costs 5 × (4 B seed + 4 B scalar) = 40 B;
  - **with K = 10 no client ever misses an iteration, so there is none;**
  - with K = 3 the count is random, because it depends on which clients are picked. It is checked against a replay of the actual sampling sequence (✓).

**Model sharing costs 0.15 MiB in total (K = 10)**, against 574.88 MiB for one FedAvg round trip per client per round.

### Compared with the paper's vision run

The paper's vision experiment (Appendix C.5, Table 4; Table 3) uses the same model and the same split as ours. The differences are the dataset, the number of clients and how long it trains.

| Setting | Paper | Here (main) |
|---|---|---|
| Dataset | CIFAR-10 IID (32 × 32 × 3) | MNIST IID, resized to 32 × 32 × 3 |
| Model and split | ResNet-18, split between the 2nd and 3rd residual blocks | ResNet-18, middle cut (the same split) |
| Clients | 100, 10 sampled per round | 10, all 10 in every iteration |
| Images per client | 500 | 6,000 |
| Batch | 32 | 256 |
| P, μ | 5, 1e-3 | 5, 1e-3 |
| Initialization | ImageNet-pretrained, frozen BatchNorm | random |
| Training length | 160,000 images processed (p.7) | 660,000 images processed (11 rounds) |

| Result | Paper | Here (main) |
|---|---:|---:|
| Activations up | 1.22 GB | 5.04 GiB |
| Gradients down | 1.22 GB | 5.04 GiB |
| Seeds + scalars | 0.0042 GB | 0.0001 GiB |
| **Communication, total** | **2.44 GB** | **10.08 GiB** |
| **Communication per image** | **≈ 16,400 B** | **≈ 16,390 B** |
| Test accuracy | 75.0% (CIFAR-10) | 97.93% (MNIST) |

- **Per image, the communication is the same.**
  - **Why:** both send one activation up and one gradient down for every image, and with 32 × 32 inputs and the same split one activation is 8,192 B in both.
  - **Paper's units:** its "GB" is 1024³ bytes; 160,000 × 8,192 B = 1.2207 × 1024³, its 1.22.
- **The total is 4.125× the paper's**, because we process 4.125× as many images (660,000 / 160,000); 2.44 × 4.125 = 10.07.
- **Labels: we charge them, the paper's table has no row for them (5.04 MiB here).**
  - **Why they cross the cut:** the loss is computed on the server. The server needs the true answer for every image, so the client sends the labels with the activation (Alg.1 L10: "Client sends (z<sub>m</sub>, y<sub>m</sub>) to Server"; the loss in L14 uses y<sub>m</sub>).
  - **Why 8 B per label:** each label is a 64-bit integer, PyTorch's standard type for class labels (`y.long()`). One batch of 256 labels is 2,048 B; 660,000 labels are 5,280,000 B = 5.04 MiB.
  - **Same rule for every method:** labels are charged whenever the server computes the loss (`charge_cut_labels`, `src/algos/__init__.py:142-152`; called in `ho_sfl.py:342`).
  - **Why the paper may leave them out:** at its scale they are 160,000 × 8 B = 1.28 MB, about 0.001 GB, too small for its table. Here they are 0.05% of the total.
- **Seeds and scalars: the paper sends more (≈ 0.004 GB) than we do (0.15 MiB with K = 10, 0.39 MiB with K = 3), for two reasons.**
  - **1. More updates per image.** Seeds and scalars are sent once per client batch: 5 seeds down, 5 scalars up and 5 averages down, 20 B each.
    - **Paper, batch 32:** 160,000 images / 32 = **5,000 client batches**, one per 32 images.
    - **Here, batch 256:** **2,640 client batches** for 660,000 images, one per 250 images.
    - Per image, the paper makes 256 / 32 = **8× as many** updates, so it sends 8× as many seeds and scalars per image. In total, 5,000 vs 2,640 client batches.
  - **2. Catch-up.** A client that sat out must download the 5 seeds and 5 averaged scalars of every iteration it missed (40 B per missed iteration, Alg.3).
    - **Paper:** 10 of 100 clients per iteration, so a client takes part about 1 iteration in 10 and has missed about 9 each time it returns: ≈ 360 B per return.
    - **Here, K = 10:** no client ever misses an iteration, so there is no catch-up. (K = 3: about 2 missed between turns; 1,748 catch-ups, 0.24 MiB.)
  - **Rough count for the paper:**
    - seeds, scalars up and averages down: 3 × 5,000 × 20 B ≈ 0.3 MB;
    - catch-up: ≈ 5,000 × 9 × 40 B ≈ 1.8 MB;
    - ≈ 2 MB in total, the same order as its reported 0.0002 + 0.004 GB. The paper does not give its exact counts or byte sizes, so the figure cannot be matched exactly.
  - In both, this traffic is under 0.2% of the total.
- **Accuracy is not comparable across the two.** MNIST is a much easier task than CIFAR-10. In the paper, HO-SFL reaches 75.0% against SFL's 77.5% and FSL-SAGE's 62.1% (Table 3). Here it reaches 97.93% against FSL-SAGE's 98.96% and CSE-FSL's 99.13%.
- **The ranking on communication differs.**
  - **In the paper,** HO-SFL (2.44 GB) is below SFL (4.20) and FSL-SAGE (3.71). Their cost there is mostly model transfers: each client holds only 500 images, so a model goes up and down after every ~500 images (0.84 GB of SFL model uploads = 329 uploads for 160,000 images).
  - **Here,** each client trains on 6,000 images between transfers (110 uploads for 660,000 images), so model traffic per image is 12× smaller.
  - HO-SFL's cost per image is fixed, so in this setting it sends more than FSL-SAGE (1642.46 MiB) and CSE-FSL (3222.21 MiB).

Code:
- iteration: `src/algos/ho_sfl.py` `group_step` (steps 1–7 numbered in the code);
- client sampling: `src/algos/__init__.py` `_run_grouped_round`.

## Memory

**Client: 49.44 MiB in both runs.**

| Client | MiB |
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

**Server:**

| Server | K = 10 (main) | K = 3 (reference) |
|---|---:|---:|
| Server model: params 40.08 + grads 40.05 + AdamW 80.10 | 160.22 | 160.22 |
| Activations kept for its backward (one batch) | 14.04 | 14.04 |
| The sampled clients' uploads, held until the one server step | **10 × 2.00 = 20.00** | 3 × 2.00 = 6.00 |
| Small outputs (predictions, loss) | 0.01 | 0.01 |
| **Per device (table value)** | **194.27** | 180.27 |
| + history of seeds and scalars (Alg.1 L2), 40 B per iteration | 264 × 40 B = 0.01 | 900 × 40 B = 0.03 |
| **Total for all 10 clients** | **194.28** | 180.31 |

- **Held-across-cut = 0:** the client keeps no activations while the server works.
- **System peak:** K = 10: 122.03 MiB live (both sides at one moment), 292.68 with model and optimizer bytes. K = 3: 59.01 / 229.67. With K = 10 all 10 clients hold their batch, activation and λ at the same moment.

## Time

| | K = 10 (main) | K = 3 (reference) |
|---|---:|---:|
| Latency | **670.71 s** | 673.42 s |
| Client compute (mean per client over 11 rounds) | 13.86 s | 18.13 s |
| Server compute (mean per client over 11 rounds) | 23.42 s | 17.93 s |

- **Latency is about the same in both runs.** Each client batch needs 6 client forwards (1 clean + 5 nudged) and a share of a server forward and backward; that work per image is identical.
- **The compute times shift between client and server.** With K = 10 the client rebuilds and applies its update 264 times instead of 900. These timers measure CPU time without waiting for the GPU, so time moves between the two sides. Compare latency, not the compute times.

## Accuracy

**97.93%** (K = 10), close to CSE-FSL (99.13%) and FSL-SAGE (98.96%). The client learns from 5 random directions per update instead of an exact gradient, so it is slightly lower. K = 3 reaches 97.80%.

## What changes between K = 10 and K = 3

- **Communication is the same.** HO-SFL sends an activation up and a λ down for every image, whatever K is. The only difference is the catch-up traffic: with every client in every iteration, no client ever falls behind, so ClientSync (Alg.3) never runs (0 vs 0.24 MiB).
- **Accuracy is about the same (97.93% vs 97.80%).** K = 10 makes 3.4× fewer updates (264 vs 900), but each one averages the scalars of 10 clients and the server gradient of 10 batches, so it is less noisy. On MNIST the two effects roughly cancel. With one seed, the 0.13-point difference is within run-to-run variation.
- **The server needs more memory with K = 10.** It holds every client's upload until its one step: 10 × 2.00 MiB instead of 3 × 2.00, so 194.27 vs 180.27 MiB. At the busiest moment the whole system holds about twice as much (122.03 vs 59.01 MiB live).
- **Client memory is unchanged.** Each client still holds one batch, its activation and its λ.
- **Latency is the same** (670.71 vs 673.42 s); only the number of optimizer steps changes.

## Differences from the paper (deliberate, for a fair benchmark)

- **Random init**, like every other method. The paper uses ImageNet-pretrained weights with frozen BatchNorm; here the client's BatchNorm trains normally.
- **Client participation.** The main run uses all 10 clients in every iteration. The paper's vision runs sample 10 of 100 clients; its language runs sample 3 of 10, as in our reference run.
- **Batch 256**, the batch every method uses (the paper: 32).
- **10 clients**, and one round is one pass over each client's data (the paper's vision runs: 100 clients, 10 sampled, a 160k-image budget).
