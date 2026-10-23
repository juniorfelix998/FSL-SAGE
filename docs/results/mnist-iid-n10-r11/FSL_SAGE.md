# FSL-SAGE: MNIST IID, 10 clients, 11 rounds

Paper: Nair et al., arXiv:2505.23182. Run: Colab T4, ResNet-18, middle cut,
seed 200, batch 256, h = 5, l = 10.

```
python main.py rounds=11 save=False device=cuda algorithm=fsl_sage \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true
```

| Result | Value |
|---|---:|
| Test accuracy (after the 11th round) | **98.96%** |
| Communication, total | **1642.46 MiB** |
| Client memory | **124.74 MiB** |
| Server memory, total for all 10 clients | **726.47 MiB** |
| Server memory, per device (table value) | 296.98 MiB |
| Latency | **422.37 s** |

Units: MiB = 1024² bytes (the benchmark table's "MB").

## How FSL-SAGE works here

- **The client only runs its aux model.** Every batch, it uses it to estimate
  the gradient it would otherwise need from the server.
- **The server trains (aligns) each client's aux model** on that client's
  stored uploads, at the start of rounds 0 and 10 (every l = 10 rounds), and
  sends it down. At round 0 nothing is stored yet, so the first real
  alignment is at round 10. That is why the run uses 11 rounds.
- **No gradient is ever sent back**, and the aux model is never averaged.

## Inputs

- **Batches per round:** 6,000 images per client ÷ 256 = **24** batches.
- **Uploads per round:** each client uploads **4** of them: batches 5, 10,
  15, 20 (paper Alg.1 l.11, Alg.3 l.3).
- **Sizes:**
  - one upload = **2 MiB**
  - labels = **0.002 MiB**
  - client model = **2.6131 MiB**
  - aux model = **8.0294 MiB**

## Communication: 1642.46 MiB

| What is sent | Count | MiB |
|---|---|---:|
| Smashed data, client → server | 10 × 11 × 4 = 440 | 880.00 |
| Labels with each upload | 440 | 0.86 |
| Gradient back to client | 0 | 0.00 |
| Client model up (FedAvg) | 10 × 11 = 110 | 287.44 |
| Client model down | 10 × 12 = 120: one initial copy (Alg.2 l.4) + one per round | 313.57 |
| Aux model down | 10 × 2 = 20: rounds 0 and 10 | 160.59 |
| Aux model up | 0 | 0.00 |
| **Total** | Cut 880.86 + Weights 761.60 | **1642.46** |

Code:
- uploads: `src/algos/fsl_sage.py:210-213`
- initial broadcast and aux downloads: `src/algos/fsl_sage.py:142-166`
- client FedAvg: `src/algos/__init__.py:474`

## Memory

| Client: 124.74 MiB | MiB |
|---|---:|
| Client model: params 2.61 + grads 2.61 + Adam 5.21 | 10.43 |
| Aux model, params only (the client never trains it) | 8.03 |
| Working memory: activations 103.28 + input batch 3.00 | 106.28 |
| **Total** | **124.74** |

### Server memory: 726.47 MiB total for all 10 clients

The server aligns **one** client's aux model at a time but keeps state for
**all** 10.

| Part | MiB |
|---|---:|
| Server model: params 40.08 + grads 40.05 + Adam 80.10 | 160.22 |
| Aux model being aligned (1 client): 8.03 + 8.02 + 16.04 | 32.09 |
| Working memory during alignment: activations 87.05 + that client's stored uploads 9.81 + its true gradients 7.81 | 104.67 |
| **Per device (table value)**: what the server needs at its busiest moment | **296.98** |
| + aux models of the other 9 clients: 9 × 32.09 | 288.79 |
| + stored uploads of the other 9 clients: 9 × 15.63 (1,000 samples each) | 140.69 |
| **Total for all 10 clients** | **726.47** |

- **Held-across-cut = 0:** the aligned aux model arrives at the start of a
  round, so the client never waits for the server.
- **System peak:** 106.28 MiB live, or 317.05 with all model and optimizer
  bytes added.

Code: `src/algos/fsl_sage.py:73-82` (where the aux model is counted),
`:84-131` (other clients), `:140-171` (alignment).

## Time

- **Latency:** 422.37 s.
- **Compute (mean per client over 11 rounds):**
  - client 2.49 s
  - server 11.25 s
- **Client averaging (total over 11 rounds):** 1.70 s.
- **The server figure hides the real alignment time.** Alignment in round 10
  takes about 106 s for all 10 clients, but the code books it all to client 0,
  so the mean over clients shows about a tenth of it.
