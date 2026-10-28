# HO-SFL: MNIST IID, 10 clients, 11 rounds

Paper: HO-SFL, arXiv:2603.14773. Code: ported from the authors'
`HO_SFL_CV/src/runner/ho_sfl_runner.py`. Run: Colab T4, ResNet-18, middle cut,
seed 200, batch 256, P = 5, μ = 1e-3, random init.

```
python main.py rounds=11 save=False device=cuda algorithm=ho_sfl \
    model=resnet18 dataset=mnist cut=middle seed=200 comm_threshold_mb=204800 measure_memory=true
```

| Result | Value |
|---|---:|
| Test accuracy (after the 11th round) | **97.59%** |
| Communication, total | **10318.55 MiB** |
| Client memory | **19.43 MiB** |
| Server memory | **176.26 MiB** |
| Latency | **633.68 s** |

Units: MiB = 1024² bytes (the benchmark table's "MB").

## How HO-SFL works here

Every client batch is one HO-SFL iteration (paper Alg.1):

1. **Catch-up:** the client catches up on iterations it missed (seeds +
   averaged scalars, Alg.3).
2. **Seeds:** the server sends P = 5 random seeds.
3. **Client forward only, no backward.** The client sends its activations and
   labels up.
4. **Server step:** the server runs forward + backward + an optimizer step on
   its one model, then sends the **activation gradient** back.
5. **Perturbed forwards:** the client runs 5 more forwards, each with its
   weights nudged along a random direction made from one seed. For each it
   sends one number back: how much the output change agrees with the gradient.
6. **Client update:** the server sends the averaged numbers back, and the
   client turns them into a weight update (AdamW).

**No model weights are ever sent.** All clients stay identical by replaying
the same seeds and numbers.

## Inputs

- **Iterations:** every batch is uploaded, so 24 per client per round,
  including the last one of 112 images.
- **Activation size:** one image's activation is 2,048 floats × 4 B = 8,192 B;
  its label is 8 B.
- **Per client per round:** all 6,000 images go up (6,000 × 8,192 B =
  46.88 MiB), and the same amount of gradient comes back.
- **One seed or scalar** is 4 B (as in the authors' `communicator.py`).

## Communication: 10318.55 MiB

| What is sent | Count | MiB |
|---|---|---:|
| Activations, client → server | 10 × 11 × 24 = 2,640 | 5156.25 |
| Labels with each activation | 2,640 | 5.04 |
| **Activation gradient, server → client** | 2,640 | **5156.25** |
| Client model up / down | 0 (never sent) | 0.00 |
| 5 scalars up | 2,640 × 20 B | 0.05 |
| 5 seeds down | 2,640 × 20 B | 0.05 |
| 5 averaged scalars down | 2,640 × 20 B | 0.05 |
| Catch-up for missed iterations | 109 catch-ups, 22,680 missed iterations × 40 B | 0.87 |
| **Total** | Cut 10317.54 + Weights 1.02 | **10318.55** |

- **Why the total is so large:** HO-SFL sends the full activations up and
  the full gradient back for **every** batch (2,640 times).
- **Model sharing is almost free:** 1.02 MiB in total, because only seeds
  and numbers are exchanged.

Code: `src/algos/ho_sfl.py` `client_step` (steps 1–7 are numbered in the code).

## Memory

| Client: 19.43 MiB | MiB |
|---|---:|
| Client model: params 2.61 + grads 2.61 + AdamW 5.21 | 10.43 |
| Working memory: input batch 3.00 + its activation 2.00 + downloaded gradient 2.00 + one perturbed activation 2.00 (no autograd activations: forward only) | 9.00 |
| **Total** | **19.43** |

| Server: 176.26 MiB | MiB |
|---|---:|
| Server model: params 40.08 + grads 40.05 + AdamW 80.10 | 160.22 |
| Working memory: activations 14.04 + uploaded batch 2.00 | 16.04 |
| **Total** | **176.26** |
| + history of seeds and scalars for catch-ups (Alg.1 L2): 2,640 × 40 B | 0.10 |
| **Total for all 10 clients** | **176.36** |

- **Held-across-cut = 0:** the client keeps no autograd activations at all.
- **System peak:** 19.04 MiB live (server activations 14.04 + upload 2.00
  + client input 3.00 at one moment), or 189.70 with all model and optimizer
  bytes added.

## Time

- **Latency:** 633.68 s.
- **Compute (mean per client over 11 rounds):**
  - client 32.25 s: 6 forwards per batch, one clean + 5 perturbed;
  - server 3.02 s.

## Differences from the paper (deliberate, for a fair benchmark)

- **Random init**, like every other method. The paper uses an
  ImageNet-pretrained ResNet-18 with frozen BatchNorm. Because of this, the
  client's BatchNorm trains normally.
- **No random client sampling.** The paper picks 10 of 100 clients per step;
  here the 10 clients take turns, and each uses all its data every round, like
  every other method.
- **Benchmark settings.** Batch 256 and 10 clients (the authors use 32 and
  100), and a round is one pass over each client's data. Methods are compared
  at equal samples processed, as the authors themselves compare them.
