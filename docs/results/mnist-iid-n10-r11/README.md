# CSE-FSL, FSL-SAGE and HO-SFL: MNIST IID, 10 clients, 11 rounds

Same setup for all three:
- Colab T4, ResNet-18, middle cut;
- MNIST (32×32, 3 channels), IID, seed 200;
- batch 256, random init.

Each follows its paper's pseudocode, and every transfer count matches the paper.
- **What changed from the original code:**
  [CSE-FSL / FSL-SAGE](../../CSE_FSL_FSL_SAGE_BREAKDOWN.md#paper-vs-original-code-changed-after-review),
  [HO-SFL](../../HO_SFL_BREAKDOWN.md).
- **Details:** [CSE_FSL.md](CSE_FSL.md), [FSL_SAGE.md](FSL_SAGE.md),
  [HO_SFL.md](HO_SFL.md).

Units: MiB.

## Summary

| | CSE-FSL | FSL-SAGE | HO-SFL |
|---|---:|---:|---:|
| **Test accuracy** | **99.13%** | **98.96%** | **97.59%** |
| Communication: Cut | 880.86 | 880.86 | 10317.54 |
| Communication: Weights | 2341.35 | 761.60 | 1.02 |
| **Communication: Total** | **3222.21** | **1642.46** | **10318.55** |
| Client memory | 148.80 | 124.74 | **19.43** |
| Held-across-cut | 0.00 | 0.00 | 0.00 |
| Server memory, per device (table value) | 176.26 | 296.98 | 176.26 |
| **Server memory, total for all 10 clients** | **176.26** | **726.47** | **176.36** |
| System peak (live) | 106.28 | 106.28 | 19.04 |
| **Latency** | **338.87 s** | **422.37 s** | **633.68 s** |

## Why they differ

**CSE-FSL vs FSL-SAGE**

| Difference | Reason |
|---|---|
| **Communication −1579.75** | **The aux model.** CSE-FSL averages it every round, sending it up and down 220 times (1766.46 MiB). FSL-SAGE only sends it down 20 times (160.59) and adds 10 initial client-model copies (26.13): 1766.46 − 160.59 − 26.13 = 1579.74. |
| **Client −24.06** | **FSL-SAGE's client never trains the aux model**, so it doesn't store the aux model's gradients (8.02) or Adam state (16.04). |
| **Server per device +120.72** | **FSL-SAGE's server trains the aux model:** +32.09 for its model, gradients and Adam state, and +88.63 working memory for the alignment pass over 1,000 stored samples. |
| **Server total +550.21** | **FSL-SAGE's server also keeps every other client's aux model and stored uploads:** 288.79 + 140.69. |
| **Latency +83.50 s** | **The round-10 alignment**, about 106 s of server work in total. |

**HO-SFL vs the other two**

| Difference | Reason |
|---|---|
| **Communication 3–6× higher** | HO-SFL sends activations up for **every** batch, 2,640 times (5156.25 MiB), where the others send 440 uploads (880.00). It also sends the full **gradient back** every time (5156.25); CSE-FSL and FSL-SAGE never send gradients. |
| **Weights ≈ 0 (1.02)** | HO-SFL never sends model weights: clients stay identical by sharing 5 seeds and 5 numbers per batch (20 B each way). |
| **Client memory 19.43, the lowest** | No aux model, and the client never runs a backward pass, so it keeps no activations. It holds only its model + AdamW state (10.43) and the batch's tensors (9.00). |
| **Server 176.26** | Same as CSE-FSL: one server model and one batch's working memory. Its catch-up history adds only 0.10. |
| **System peak 19.04** | The client's peak is tiny, so the peak moment is the server's backward pass. |
| **Latency, slowest** | The client runs 6 forward passes per batch (1 clean + 5 perturbed): 32.25 s of client compute, against 4.40 s (CSE-FSL) and 2.49 s (FSL-SAGE). |
| **Accuracy −1.5 pt** | The client learns only from 5 random directions per batch, instead of an exact gradient. |

## Communication, side by side

| What is sent | CSE-FSL | FSL-SAGE | HO-SFL |
|---|---:|---:|---:|
| Smashed data + labels | 880.86 (440) | 880.86 (440) | 5161.29 (2,640) |
| Gradient back to client | 0 | 0 | 5156.25 (2,640) |
| Client model up | 287.44 (110) | 287.44 (110) | 0 |
| Client model down | 287.44 (110) | 313.57 (120, incl. initial copy) | 0 |
| Aux model up | 883.23 (110) | 0 | – |
| Aux model down | 883.23 (110) | 160.59 (20) | – |
| Seeds + scalars + catch-up | – | – | 1.02 |
| **Total** | **3222.21** | **1642.46** | **10318.55** |

## Memory, side by side

| | CSE-FSL | FSL-SAGE | HO-SFL |
|---|---:|---:|---:|
| Client model (params + grads + Adam) | 10.43 | 10.43 | 10.43 |
| Aux model on the client | 32.09 | 8.03 (params only) | – |
| Client working memory | 106.28 | 106.28 | 9.00 |
| **Client total** | **148.80** | **124.74** | **19.43** |
| Server model (params + grads + Adam) | 160.22 | 160.22 | 160.22 |
| Aux model being aligned | – | 32.09 | – |
| Server working memory | 16.04 | 104.67 | 16.04 |
| **Server, per device (table value)** | **176.26** | **296.98** | **176.26** |
| + other clients' state / catch-up history | – | 288.79 + 140.69 | 0.10 |
| **Server, total for all 10 clients** | **176.26** | **726.47** | **176.36** |

These results replace the earlier 3-round ones in
[`../mnist-iid-n10-r3/`](../mnist-iid-n10-r3/), which used the original code
(5 uploads per round, FSL-SAGE Held = 94.01) and are kept for the record.
