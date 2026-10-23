# CSE-FSL: MNIST IID, 10 clients, 11 rounds

Paper: Mu & Shen, arXiv:2302.05599. Run: Colab T4, ResNet-18, middle cut,
seed 200, batch 256, h = 5.

```
python main.py rounds=11 save=False device=cuda algorithm=cse_fsl \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true
```

| Result | Value |
|---|---:|
| Test accuracy (after the 11th round) | **99.13%** |
| Communication, total | **3222.21 MiB** |
| Client memory | **148.80 MiB** |
| Server memory | **176.26 MiB** |
| Latency | **338.87 s** |

Units: MiB = 1024² bytes (the benchmark table's "MB").

## Inputs

- **Batches per round:** 6,000 images per client ÷ 256 = **24** batches
  (the last one has 112 images).
- **Uploads per round:** each client uploads **4** of them: batches 5, 10,
  15, 20, the last batch of every 5-batch window (paper Eq.8, Fig.2).
- **Sizes:**
  - one upload = 256 × 2,048 floats × 4 B = **2 MiB**
  - labels = 256 × 8 B = **0.002 MiB**
  - client model = **2.6131 MiB**
  - aux model = **8.0294 MiB**

## Communication: 3222.21 MiB

| What is sent | Count | MiB |
|---|---|---:|
| Smashed data, client → server | 10 clients × 11 rounds × 4 = 440 | 880.00 |
| Labels with each upload | 440 | 0.86 |
| Gradient back to client | 0 (never sent) | 0.00 |
| Client model up + down (FedAvg) | 2 × 110 | 574.88 |
| Aux model up + down (FedAvg) | 2 × 110 | 1766.46 |
| **Total** | Cut 880.86 + Weights 2341.35 | **3222.21** |

More than half of everything sent (1766.46 MiB, 55%) is the aux model. It's
averaged like the client model and is 3× bigger.

Code:
- uploads: `src/algos/cse_fsl.py:104-107`
- client FedAvg: `src/algos/__init__.py:474`
- aux FedAvg: `src/algos/cse_fsl.py:138`

## Memory

| Client: 148.80 MiB | MiB |
|---|---:|
| Client model: params 2.61 + grads 2.61 + Adam 5.21 | 10.43 |
| Aux model (trained on the client every batch): 8.03 + 8.02 + 16.04 | 32.09 |
| Working memory: activations 103.28 + input batch 3.00 | 106.28 |
| **Total** | **148.80** |

| Server: 176.26 MiB | MiB |
|---|---:|
| Server model: params 40.08 + grads 40.05 + Adam 80.10 | 160.22 |
| Working memory: activations 14.04 + uploaded batch 2.00 | 16.04 |
| **Total** | **176.26** |

- **Held-across-cut = 0:** the client finishes its own step before uploading,
  so nothing is kept waiting for the server.
- **System peak:** 106.28 MiB live (the client's working memory), or 309.02
  with all model and optimizer bytes added.

## Time

- **Latency:** 338.87 s, the whole run.
- **Compute (mean per client over 11 rounds):**
  - client 4.40 s
  - server 0.65 s
- **Averaging (total over 11 rounds):**
  - client models 1.63 s
  - aux models 0.80 s
