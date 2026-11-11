# CSE-FSL, FSL-SAGE, HO-SFL, MU-SplitFed and FedSplitX: MNIST IID, 10 clients, 11 rounds

Same setup for all five:
- Colab T4, ResNet-18, middle cut;
- MNIST (32×32, 3 channels), IID, seed 200;
- batch 256, random init.

Each follows its paper's pseudocode, and every transfer count matches the paper.
- **What changed from the original code:**
  [CSE-FSL / FSL-SAGE](../../CSE_FSL_FSL_SAGE_BREAKDOWN.md#paper-vs-original-code-changed-after-review),
  [HO-SFL](../../HO_SFL_BREAKDOWN.md),
  [MU-SplitFed](../../MU_SPLITFED_BREAKDOWN.md) (a reimplementation: the authors'
  code is LLM-only),
  [FedSplitX](../../FEDSPLITX_BREAKDOWN.md) (a reimplementation from the paper:
  no public code).
- **Details:** [CSE_FSL.md](CSE_FSL.md), [FSL_SAGE.md](FSL_SAGE.md),
  [HO_SFL.md](HO_SFL.md), [MU_SPLITFED.md](MU_SPLITFED.md),
  [FEDSPLITX.md](FEDSPLITX.md).

Units: MiB.

## Summary

HO-SFL is reported with all 10 clients in every iteration (K = 10), at batch 256.
The 3-of-10 run is in [HO_SFL.md](HO_SFL.md).

MU-SplitFed samples 5 of the 10 clients per step and averages the models after
every step; it is reported at batch 256.

FedSplitX trains 10% of the clients per round (1 of 10), as in its paper, at
batch 256.

| | CSE-FSL | FSL-SAGE | HO-SFL | MU-SplitFed | FedSplitX |
|---|---:|---:|---:|---:|---:|
| **Test accuracy** | **99.13%** | **98.96%** | **97.93%** | **20.08%** | **95.85%** |
| Communication: Cut | 880.86 | 880.86 | 10317.54 | 15473.80 | 516.13 |
| Communication: Weights | 2341.35 | 761.60 | 0.15 | 13797.24 | 939.43 |
| **Communication: Total** | **3222.21** | **1642.46** | **10317.69** | **29271.03** | **1455.56** |
| Client memory | 148.80 | 124.74 | **49.44** | **5.62** | **102.53** |
| Held-across-cut | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Server memory, per device (table value) | 176.26 | 296.98 | 194.27 | 86.16 | 181.17 |
| **Server memory, total for all 10 clients** | **176.26** | **726.47** | **194.28** | **246.47** | **181.17** |
| System peak (live) | 106.28 | 106.28 | 122.03 | 21.01 | 144.16 |
| **Latency** | **338.87 s** | **422.37 s** | **670.71 s** | **548.80 s** | **144.27 s** |

HO-SFL's memory was measured with the updated meter, which also counts a
forward pass's working set (the layer outputs alive at the same time).

## Why they differ

**CSE-FSL vs FSL-SAGE**

| Difference | Reason |
|---|---|
| **Communication −1579.75** | **The aux model.** CSE-FSL averages it every round, sending it up and down 220 times (1766.46 MiB). FSL-SAGE only sends it down 20 times (160.59) and adds 10 initial client-model copies (26.13): 1766.46 − 160.59 − 26.13 = 1579.74. |
| **Client −24.06** | **FSL-SAGE's client never trains the aux model**, so it doesn't store the aux model's gradients (8.02) or Adam state (16.04). |
| **Server per device +120.72** | **FSL-SAGE's server trains the aux model:** +32.09 for its model, gradients and Adam state, and +88.63 working memory for the alignment pass over 1,000 stored samples. |
| **Server total +550.21** | **FSL-SAGE's server also keeps every other client's aux model and stored uploads:** 288.79 + 140.69. |
| **Latency +83.50 s** | **The round-10 alignment**, about 106 s of server work in total. |

**HO-SFL vs CSE-FSL and FSL-SAGE**

| Difference | Reason |
|---|---|
| **Communication 3–6× higher** | HO-SFL sends activations up for **every** image (5156.25 MiB), where the others upload 1 batch in 5 (880.00). It also sends the full **gradient back** for every image (5156.25); CSE-FSL and FSL-SAGE never send gradients. |
| **Same per image as the paper** | The paper's Table 4 (CIFAR-10, same ResNet-18 split) gives 2.44 GB for 160,000 images: activation + gradient, 16,400 B per image. We process 660,000 images, so 4.125× that: 10.08 GiB. The paper's HO-SFL beats FSL-SAGE because its clients hold only 500 images each, so model transfers are frequent; here each client trains 6,000 images between transfers. |
| **Weights under 1 MiB** | HO-SFL never sends model weights: clients stay identical by sharing 5 seeds and 5 numbers per client batch (20 B each way): 0.15 in total. With all 10 clients in every iteration no client falls behind, so there is no catch-up traffic. |
| **Client memory 49.44, about a third of theirs** | No aux model, and no backward pass on the client, so it keeps no activations for backprop. Its model + AdamW (10.43) and the forward pass's working set (39.01): conv1 and bn1 outputs alive together (32.00), the input batch (3.00), its activation and the downloaded gradient (4.00). |
| **Server 194.27** | One server model like CSE-FSL (160.22). The working memory is higher (34.05 vs 16.04) because the server holds all 10 clients' uploads (10 × 2.00) until its one averaged step. |
| **System peak 122.03, above theirs (106.28)** | No client keeps activations for backprop, but all 10 clients hold their batch, activation and gradient at the same moment, while the server holds 10 uploads. |
| **Latency 670.71 s, the slowest of the three** | Each client batch needs 6 client forward passes (1 clean + 5 nudged). |
| **Accuracy 1.0–1.2 pt lower** | The client learns from 5 random directions per update instead of an exact gradient. |

**MU-SplitFed vs the others** (batch 256)

| Difference | Reason |
|---|---|
| **Accuracy 20.08%, by far the lowest** | **Both** sides learn only from random directions, the server's 10.5 million weights included, from random init. The HO-SFL paper reports 14.0% for this reimplementation on CIFAR-10 (Table 3, p.23). |
| **Communication 29271.03** | **3 activations** per client batch (clean + 2 nudged): 15468.75 MiB. Models are averaged after every step, so each client sends its model up and pulls it down **for every batch**: 2 × 6898.62 MiB. |
| **No gradient back** | Like CSE-FSL and FSL-SAGE (unlike HO-SFL): only one number per batch comes back (0.01 MiB in total). |
| **Client memory 5.62, the lowest** | The client only runs forward passes and updates its weights in place: its model's parameters (2.61) plus its input batch (3.00). No gradients, no optimizer state, no activations. |
| **Server 86.16 per device, 246.47 for all copies** | The global server model + the copy being trained (40.08 each, no gradients or optimizer state) + the 3 uploads held (6.00). The other 4 sampled clients' copies add 160.31. |

**FedSplitX vs the others** (batch 256)

| Difference | Reason |
|---|---|
| **Communication 1455.56, the lowest** | **Only 1 client per round** (the paper's 10%), and it sends its activations **once per round** for all its 6,000 images (515.63 MiB in total). No gradient comes back. Most of the rest is the server-copy averaging (881.72), which the benchmark charges; without it, 573.84. |
| **Latency 144.27 s, the fastest** | 11 client-rounds of training in total, against 110 for the methods that train every client every round. |
| **Accuracy 95.85%, below CSE-FSL / FSL-SAGE** | It sees about 1.1 passes over MNIST in 11 rounds, against 11. The client-side heads alone reach 91.61%. |
| **Client memory 102.53** | Its model with gradients (5.22, plain SGD: no optimizer state), 3 tiny aux heads (0.02), and the backward pass through the stem, layer1 and layer2 with 3 heads (94.29) plus the batch (3.00). |
| **Server 181.17** | The global server model (40.08) + the copy being trained with gradients (80.13) + working memory 60.96, of which **46.88 is the whole round's upload**, held until the copy has trained on it. |

## Communication, side by side

| What is sent | CSE-FSL | FSL-SAGE | HO-SFL | MU-SplitFed | FedSplitX |
|---|---:|---:|---:|---:|---:|
| Smashed data + labels | 880.86 (440) | 880.86 (440) | 5161.29 (2,640) | 15473.79 (7,920 + 2,640 labels) | 516.13 (11 + 11 labels, all 6,000 images each) |
| Gradient back to client | 0 | 0 | 5156.25 (2,640) | 0 | 0 |
| Client model up | 287.44 (110) | 287.44 (110) | 0 | 6898.62 (2,640) | 28.74 (11) |
| Client model down | 287.44 (110) | 313.57 (120, incl. initial copy) | 0 | 6898.62 (2,640) | 28.74 (11) |
| Aux model up | 883.23 (110) | 0 | – | – | 0.11 (11) |
| Aux model down | 883.23 (110) | 160.59 (20) | – | – | 0.11 (11) |
| Seeds + scalars + catch-up | – | – | 0.15 (no catch-up) | 0.01 (loss-difference numbers) | – |
| Server copy down + up (Server-model FedAvg) | 0 | 0 | 0 | 0 | 881.72 (22) |
| **Total** | **3222.21** | **1642.46** | **10317.69** | **29271.03** | **1455.56** |

## Memory, side by side

| | CSE-FSL | FSL-SAGE | HO-SFL | MU-SplitFed | FedSplitX |
|---|---:|---:|---:|---:|---:|
| Client model (params + grads + Adam) | 10.43 | 10.43 | 10.43 | 2.61 (params only) | 5.22 (params + grads, SGD) |
| Aux model on the client | 32.09 | 8.03 (params only) | – | – | 0.02 (3 small heads) |
| Client working memory | 106.28 | 106.28 | 39.01 (forward working set, no backprop) | 3.00 | 97.29 |
| **Client total** | **148.80** | **124.74** | **49.44** | **5.62** | **102.53** |
| Server model (params + grads + Adam) | 160.22 | 160.22 | 160.22 | 80.16 (global + copy being trained, params only) | 120.21 (global 40.08 + copy being trained 80.13, SGD) |
| Aux model being aligned | – | 32.09 | – | – | – |
| Server working memory | 16.04 | 104.67 | 34.05 (incl. 10 uploads) | 6.00 | 60.96 (incl. the round's upload 46.88) |
| **Server, per device (table value)** | **176.26** | **296.98** | **194.27** | **86.16** | **181.17** |
| + other clients' state / catch-up history | – | 288.79 + 140.69 | 0.01 (seed/scalar history) | 160.31 (4 other server copies) | – (1 client per round) |
| **Server, total for all 10 clients** | **176.26** | **726.47** | **194.28** | **246.47** | **181.17** |

These results replace the earlier 3-round ones in
[`../mnist-iid-n10-r3/`](../mnist-iid-n10-r3/), which used the original code
(5 uploads per round, FSL-SAGE Held = 94.01) and are kept for the record.
