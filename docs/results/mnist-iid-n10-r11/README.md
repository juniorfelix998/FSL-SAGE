# CSE-FSL, FSL-SAGE and HO-SFL: MNIST IID, 10 clients, 11 rounds

Same setup for all three:
- Colab T4, ResNet-18, middle cut;
- MNIST (32×32, 3 channels), IID, seed 200;
- batch 256 (HO-SFL main run: 107), random init.

Each follows its paper's pseudocode, and every transfer count matches the paper.
- **What changed from the original code:**
  [CSE-FSL / FSL-SAGE](../../CSE_FSL_FSL_SAGE_BREAKDOWN.md#paper-vs-original-code-changed-after-review),
  [HO-SFL](../../HO_SFL_BREAKDOWN.md).
- **Details:** [CSE_FSL.md](CSE_FSL.md), [FSL_SAGE.md](FSL_SAGE.md),
  [HO_SFL.md](HO_SFL.md).

Units: MiB.

## Summary

HO-SFL samples 3 of the 10 clients per iteration. It has two columns:
- **HO-SFL (main):** batch 107, so one update sees ~321 images, as in the
  paper.
- **HO-SFL (b256):** the same method at batch 256, used to compare memory with
  the other two.

| | CSE-FSL | FSL-SAGE | HO-SFL (main) | HO-SFL (b256) |
|---|---:|---:|---:|---:|
| **Test accuracy** | **99.13%** | **98.96%** | **96.32%** | 97.83% |
| Communication: Cut | 880.86 | 880.86 | 10317.54 | 10317.54 |
| Communication: Weights | 2341.35 | 761.60 | 0.92 | 0.39 |
| **Communication: Total** | **3222.21** | **1642.46** | **10318.46** | 10317.93 |
| Client memory | 148.80 | 124.74 | 14.19 | **19.43** |
| Held-across-cut | 0.00 | 0.00 | 0.00 | 0.00 |
| Server memory, per device (table value) | 176.26 | 296.98 | 168.62 | **180.26** |
| **Server memory, total for all 10 clients** | **176.26** | **726.47** | 168.70 | **180.30** |
| System peak (live) | 106.28 | 106.28 | 15.50 | **37.04** |
| **Latency** | **338.87 s** | **422.37 s** | **839.90 s** | 612.28 s |

## Why they differ

**CSE-FSL vs FSL-SAGE**

| Difference | Reason |
|---|---|
| **Communication −1579.75** | **The aux model.** CSE-FSL averages it every round, sending it up and down 220 times (1766.46 MiB). FSL-SAGE only sends it down 20 times (160.59) and adds 10 initial client-model copies (26.13): 1766.46 − 160.59 − 26.13 = 1579.74. |
| **Client −24.06** | **FSL-SAGE's client never trains the aux model**, so it doesn't store the aux model's gradients (8.02) or Adam state (16.04). |
| **Server per device +120.72** | **FSL-SAGE's server trains the aux model:** +32.09 for its model, gradients and Adam state, and +88.63 working memory for the alignment pass over 1,000 stored samples. |
| **Server total +550.21** | **FSL-SAGE's server also keeps every other client's aux model and stored uploads:** 288.79 + 140.69. |
| **Latency +83.50 s** | **The round-10 alignment**, about 106 s of server work in total. |

**HO-SFL vs the other two** (memory compared at batch 256)

| Difference | Reason |
|---|---|
| **Communication 3–6× higher** | HO-SFL sends activations up for **every** image (5156.25 MiB), where the others upload 1 batch in 5 (880.00). It also sends the full **gradient back** for every image (5156.25); CSE-FSL and FSL-SAGE never send gradients. |
| **Weights under 1 MiB** | HO-SFL never sends model weights: clients stay identical by sharing 5 seeds and 5 numbers per client batch (20 B each way), plus catch-ups. |
| **Client memory 19.43, the lowest** | No aux model, and no backward pass on the client, so it keeps no activations: its model + AdamW (10.43) and the batch's tensors (9.00). |
| **Server 180.26** | One server model like CSE-FSL (160.22). The working memory is a little higher (20.04 vs 16.04) because the server holds the 3 sampled clients' uploads until its one averaged step. |
| **System peak 37.04** | The server's backward pass while 3 clients hold their batches; still far below 106.28, because no client keeps activations. |
| **Latency, slowest** | Each client batch needs 6 client forward passes (1 clean + 5 nudged); the main run also has 2.4× as many, smaller batches. |
| **Accuracy 1.3–2.8 pt lower** | The client learns from 5 random directions per update instead of an exact gradient. |

## Communication, side by side

| What is sent | CSE-FSL | FSL-SAGE | HO-SFL (main) | HO-SFL (b256) |
|---|---:|---:|---:|---:|
| Smashed data + labels | 880.86 (440) | 880.86 (440) | 5161.29 (6,270) | 5161.29 (2,640) |
| Gradient back to client | 0 | 0 | 5156.25 (6,270) | 5156.25 (2,640) |
| Client model up | 287.44 (110) | 287.44 (110) | 0 | 0 |
| Client model down | 287.44 (110) | 313.57 (120, incl. initial copy) | 0 | 0 |
| Aux model up | 883.23 (110) | 0 | – | – |
| Aux model down | 883.23 (110) | 160.59 (20) | – | – |
| Seeds + scalars + catch-up | – | – | 0.92 | 0.39 |
| **Total** | **3222.21** | **1642.46** | **10318.46** | **10317.93** |

## Memory, side by side

| | CSE-FSL | FSL-SAGE | HO-SFL (main) | HO-SFL (b256) |
|---|---:|---:|---:|---:|
| Client model (params + grads + Adam) | 10.43 | 10.43 | 10.43 | 10.43 |
| Aux model on the client | 32.09 | 8.03 (params only) | – | – |
| Client working memory | 106.28 | 106.28 | 3.76 | 9.00 |
| **Client total** | **148.80** | **124.74** | **14.19** | **19.43** |
| Server model (params + grads + Adam) | 160.22 | 160.22 | 160.22 | 160.22 |
| Aux model being aligned | – | 32.09 | – | – |
| Server working memory | 16.04 | 104.67 | 8.39 | 20.04 |
| **Server, per device (table value)** | **176.26** | **296.98** | **168.62** | **180.26** |
| + other clients' state / catch-up history | – | 288.79 + 140.69 | 0.08 | 0.03 |
| **Server, total for all 10 clients** | **176.26** | **726.47** | **168.70** | **180.30** |

These results replace the earlier 3-round ones in
[`../mnist-iid-n10-r3/`](../mnist-iid-n10-r3/), which used the original code
(5 uploads per round, FSL-SAGE Held = 94.01) and are kept for the record.
