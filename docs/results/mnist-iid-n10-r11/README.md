# CSE-FSL vs FSL-SAGE: MNIST IID, 10 clients, 11 rounds

Same setup for both:
- Colab T4, ResNet-18, middle cut;
- MNIST (32×32, 3 channels), IID, seed 200;
- batch 256, uploads every 5th batch.

Both follow their paper's pseudocode
([what changed from the original code](../../CSE_FSL_FSL_SAGE_BREAKDOWN.md#paper-vs-original-code-changed-after-review)).
Every transfer count matches the paper. Details: [CSE_FSL.md](CSE_FSL.md),
[FSL_SAGE.md](FSL_SAGE.md). Units: MiB.

## Summary

| | CSE-FSL | FSL-SAGE | Difference |
|---|---:|---:|---|
| **Test accuracy** | **99.13%** | **98.96%** | −0.17 pt |
| Communication: Cut | 880.86 | 880.86 | same |
| Communication: Weights | 2341.35 | 761.60 | −1579.75 |
| **Communication: Total** | **3222.21** | **1642.46** | **−49%** |
| Client memory | 148.80 | 124.74 | −24.06 |
| Held-across-cut | 0.00 | 0.00 | same |
| Server memory, per device (table value) | 176.26 | 296.98 | +120.72 |
| **Server memory, total for all 10 clients** | **176.26** | **726.47** | **+550.21** |
| System peak (live) | 106.28 | 106.28 | same |
| **Latency** | **338.87 s** | **422.37 s** | **+25%** |

## Why they differ

| Difference | Reason |
|---|---|
| **Communication −1579.75** | **The aux model.** CSE-FSL averages it every round, sending it up and down 220 times (1766.46 MiB). FSL-SAGE only sends it down 20 times (160.59) and adds 10 initial client-model copies (26.13): 1766.46 − 160.59 − 26.13 = 1579.74. |
| **Client −24.06** | **FSL-SAGE's client never trains the aux model**, so it doesn't store the aux model's gradients (8.02) or Adam state (16.04). |
| **Server per device +120.72** | **FSL-SAGE's server trains the aux model:** +32.09 for its model, gradients and Adam state, and +88.63 working memory for the alignment pass over 1,000 stored samples. |
| **Server total +550.21** | **FSL-SAGE's server also keeps every other client's aux model and stored uploads:** 288.79 + 140.69. |
| **Latency +83.50 s** | **The round-10 alignment**, about 106 s of server work in total. |

**What is the same:**
- **Cut traffic:** both upload batches 5, 10, 15, 20 of each round, 440 uploads in total.
- **No gradients sent back:** in either method.
- **Client working memory:** 106.28 MiB for both.

## Communication, side by side

| What is sent | CSE-FSL | FSL-SAGE |
|---|---:|---:|
| Smashed data + labels (440 uploads) | 880.86 | 880.86 |
| Client model up | 287.44 (110) | 287.44 (110) |
| Client model down | 287.44 (110) | 313.57 (120, incl. initial copy) |
| Aux model up | 883.23 (110) | 0 |
| Aux model down | 883.23 (110) | 160.59 (20) |
| **Total** | **3222.21** | **1642.46** |

## Memory, side by side

| | CSE-FSL | FSL-SAGE |
|---|---:|---:|
| Client model (params + grads + Adam) | 10.43 | 10.43 |
| Aux model on the client | 32.09 | 8.03 (params only) |
| Client working memory | 106.28 | 106.28 |
| **Client total** | **148.80** | **124.74** |
| Server model (params + grads + Adam) | 160.22 | 160.22 |
| Aux model being aligned | – | 32.09 |
| Server working memory | 16.04 | 104.67 |
| **Server, per device (table value)** | **176.26** | **296.98** |
| + other 9 clients' aux models | – | 288.79 |
| + other 9 clients' stored uploads | – | 140.69 |
| **Server, total for all 10 clients** | **176.26** | **726.47** |

These results replace the earlier 3-round ones in
[`../mnist-iid-n10-r3/`](../mnist-iid-n10-r3/), which used the original code
(5 uploads per round, FSL-SAGE Held = 94.01) and are kept for the record.
