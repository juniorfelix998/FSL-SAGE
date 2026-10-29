# HO-SFL: MNIST IID, 10 clients, 11 rounds

Paper: HO-SFL, arXiv:2603.14773. Code: ported from the authors'
`HO_SFL_CV/src/runner/ho_sfl_runner.py`. Run: Colab T4, ResNet-18, middle cut,
seed 200, P = 5, μ = 1e-3, random init, **3 of the 10 clients sampled per
iteration** (`sampled_clients: 3`).

There are two runs, differing only in batch size:

| Run | Batch | Why |
|---|---:|---|
| **Main** | 107 | One update sees 3 × 107 ≈ 321 images, the paper's 10 clients × 32 = 320. Used for accuracy, communication and latency. |
| **Memory** | 256 | The batch every other method uses, so the memory numbers compare directly with CSE-FSL and FSL-SAGE. |

```
python main.py rounds=11 save=False device=cuda algorithm=ho_sfl \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true \
    model.client.batch_size=107        # or 256 for the memory run
```

| Result | Main (batch 107) | Memory (batch 256) |
|---|---:|---:|
| Test accuracy (after the 11th round) | **96.32%** | 97.83% |
| Communication, total | **10318.46 MiB** | 10317.93 MiB |
| Client memory | 14.19 MiB | **19.43 MiB** |
| Server memory, total for all 10 clients | 168.70 MiB | **180.30 MiB** |
| Latency | **839.90 s** | 612.28 s |

Units: MiB = 1024² bytes (the benchmark table's "MB").

## How HO-SFL works here

Each **iteration** takes one batch from each of 3 clients (paper Alg.1):

1. **Sample.** 3 clients are picked at random among those that still have
   batches left this round, so every client uses all its data every round,
   like the other methods.
2. **Catch-up.** Each picked client catches up on iterations it missed
   (seeds + averaged scalars, Alg.3).
3. **Forward and upload.** The server sends 5 seeds. Each client runs a
   forward pass only (**no backward**) and sends its activations and labels up.
4. **Server step.** The server runs forward + backward for the 3 batches and
   takes **one** step with the averaged gradient. It sends each client its
   activation gradient back.
5. **Scalars.** Each client runs 5 more forwards, with its weights nudged
   along 5 random directions from the seeds. It sends 5 numbers: how much each
   change agrees with the gradient.
6. **Average and update.** The server **averages the 3 clients' numbers**
   and sends 5 back. The client model takes one update from them (AdamW).

**No model weights are ever sent.** The clients stay identical by replaying
the same seeds and numbers.

## Inputs

| | Main (batch 107) | Memory (batch 256) |
|---|---:|---:|
| Batches per client per round | 57 (56 × 107 + 1 of 8) | 24 (23 × 256 + 1 of 112) |
| Client batches in the run | 10 × 11 × 57 = 6,270 | 10 × 11 × 24 = 2,640 |
| Iterations (updates) in the run | 2,119 | 900 |

- **Why 2,119 and 900, not exactly ÷ 3:** most iterations have 3 clients. Near
  the end of a round fewer than 3 clients have batches left, so a few have 1
  or 2.
- **Per client per round:** all 6,000 images go up (6,000 × 8,192 B =
  46.88 MiB), and the same amount of gradient comes back.
- **One seed or scalar** is 4 B.

## Communication

| What is sent | Main: count | Main: MiB | Memory: count | Memory: MiB |
|---|---:|---:|---:|---:|
| Activations, client → server | 6,270 | 5156.25 | 2,640 | 5156.25 |
| Labels | 6,270 | 5.04 | 2,640 | 5.04 |
| **Activation gradient, server → client** | 6,270 | **5156.25** | 2,640 | **5156.25** |
| Client model up / down | 0 | 0.00 | 0 | 0.00 |
| 5 scalars up (20 B per client batch) | 6,270 | 0.12 | 2,640 | 0.05 |
| 5 seeds down | 6,270 | 0.12 | 2,640 | 0.05 |
| 5 averaged scalars down | 6,270 | 0.12 | 2,640 | 0.05 |
| Catch-up for missed iterations | 4,171 | 0.56 | 1,748 | 0.24 |
| **Total** | | **10318.46** | | **10317.93** |

- **Activations and gradients are the same in both runs.** They depend on the
  number of images, not the batch size.
- **Most of the total:** activations up and the full gradient back for every
  image.
- **Model sharing costs under 1 MiB.** Only seeds and numbers are sent, and
  there are more of them at batch 107 because there are more batches.
- **The catch-up count is random,** because it depends on which clients are
  picked. It is checked against a replay of the actual sampling sequence (✓).

Code:
- iteration: `src/algos/ho_sfl.py` `group_step` (steps 1–7 numbered in the code)
- fair sampling: `src/algos/__init__.py` `_run_grouped_round`

## Memory

| Client | Main (batch 107) | Memory (batch 256) |
|---|---:|---:|
| Client model: params 2.61 + grads 2.61 + AdamW 5.21 | 10.43 | 10.43 |
| Working memory: input batch + its activation + downloaded gradient + one nudged activation (no backward, so no activations kept) | 1.25 + 0.84 + 0.84 + 0.84 = 3.76 | 3.00 + 2.00 + 2.00 + 2.00 = 9.00 |
| **Total** | **14.19** | **19.43** |

| Server | Main (batch 107) | Memory (batch 256) |
|---|---:|---:|
| Server model: params 40.08 + grads 40.05 + AdamW 80.10 | 160.22 | 160.22 |
| Working memory: activations + the 3 sampled clients' uploads, held until the one server step | 5.88 + 3 × 0.84 = 8.39 | 14.04 + 3 × 2.00 = 20.04 |
| **Per device (table value)** | **168.62** | **180.26** |
| + history of seeds and scalars for catch-ups (2,119 / 900 iterations × 40 B) | 0.08 | 0.03 |
| **Total for all 10 clients** | **168.70** | **180.30** |

- **Held-across-cut = 0:** the client keeps no activations.
- **System peak, live** (busiest moment for both sides together): the server's
  backward pass on the 3rd client's batch, while all 3 clients hold their
  inputs and the first two hold their activation and gradient.
  - **Main:** 5.88 + 2.51 + 3.76 + 3.34 = **15.50**, or 186.16 with model and
    optimizer bytes.
  - **Memory run:** 14.04 + 6.00 + 9.01 + 8.00 = **37.04**, or 207.70.

## Time

| | Main (batch 107) | Memory (batch 256) |
|---|---:|---:|
| Latency | 839.90 s | 612.28 s |
| Client compute (mean per client over 11 rounds) | 31.80 s | 16.35 s |
| Server compute (mean per client over 11 rounds) | 25.14 s | 18.00 s |

Batch 107 is slower: about 2.4× as many batches and iterations, each with 6
client forwards and a server step.

## Differences from the paper (deliberate, for a fair benchmark)

- **Random init**, like every other method. The paper uses ImageNet-pretrained
  weights with frozen BatchNorm; here the client's BatchNorm trains normally.
- **Fair sampling.** 3 of 10 clients per iteration, as in the paper's language
  experiments. They are picked at random among clients with batches left, so
  every client uses all its data every round (the paper samples freely).
- **Benchmark settings.** 10 clients, and one round is one pass over each
  client's data. The main run's batch (107) is chosen so one update sees about
  the paper's 320 images.
