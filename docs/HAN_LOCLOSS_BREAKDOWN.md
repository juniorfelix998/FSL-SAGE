# Han et al. (locally generated losses): paper vs our code

**Source:** Han, Bhatti, Lee, Moon, *Accelerating Federated Learning with Split
Learning on Locally Generated Losses*, ICML 2021 Workshop on Federated Learning
(FL-ICML'21).

**No public code.** This is a **reimplementation** written from the paper alone:
- Sec. 3.1 (local losses, Eq. 1–2, p.3);
- Sec. 3.2 (the protocol, Eq. 3–5, p.3 right column → p.4 top-left);
- Table 1 (p.4, communication per round);
- p.6 (models, auxiliary network, implementation details).

Report it as a reimplementation. It has **not yet reproduced** the paper's Table 2 (see the last section). Code:
`src/algos/han_locloss.py` (`algorithm=han_locloss`); its header maps every step to the code.

The paper has no algorithm box. The protocol is prose, so the step numbers below (0–5) are ours. Each one quotes the paper.

## The model at the benchmark's middle cut

The paper splits an AlexNet-like CNN (MNIST/FMNIST) and VGG-11 (CIFAR-10) after their 4th conv (p.6). Its auxiliary network is **"a single FC layer"**:
- CNN: 23,050 params = Linear(2304, 10) on the flattened 256×3×3 client output, 0.60% of |w|;
- VGG-11: 10,250 params, 0.11% of |w|.

| | Paper (CNN, p.6) | Our ResNet-18, middle cut |
|---|---|---|
| Client w_C | 4 convs, 387,840 params | stem + layer1 + layer2, 2,740,048 B (2.61 MiB) |
| Server w_S | conv5 + 3 FC, 3,480,330 params | layer3 + layer4 + fc, 42,025,080 B (40.08 MiB) |
| Cut activation | 256×3×3 = 2,304 per sample | 128×4×4 = 2,048 per sample (8,192 B) |
| Aux a_C | Linear(2304, 10), 0.60% of \|w\| | flatten + Linear(2048, 10): 20,490 params = 81,960 B, ≈0.18% of \|w\| |

## The protocol, step by step

| Step | Paper | What happens | Code | Charged as |
|---|---|---|---|---|
| 0 | Sec. 3.2 ¶1, p.3: "two different servers, the main server and the fed server"; the server "randomly selects the participating group A_t with K clients" | init w0 = [w_C, w_S], a_C; here all clients take part every round | `__init__`, `__install_paper_auxiliary_head` | – |
| 1 | p.3: "each client k ∈ A_t downloads w_C^t and a_C^t from the fed server" | client and aux download | `aggregate()` of the previous round (download half) | W2 `weights.client_down`, W4 `weights.aux_down` |
| 2 | p.3: "performs forward propagation … in a specific mini-batch … uploads g_{w_C,k}(x) to the main server"; Fig. 1(c), p.2: "(Activation, Label)" | forward one minibatch, upload activations and labels | `client_step`: `charge_cut_activation`, `charge_cut_labels` | C1 `cut.act_up`, C2 `cut.labels_up` |
| 3a | Eq. (3), (4), p.3 | CE of a_C(w_C(x)); SGD on w_C and a_C from this local loss alone | `client_step`, client phase | – |
| 3b | p.3: "the fed server aggregates the client-side models as w_C = 1/K Σ w_C,k. The auxiliary networks are also aggregated" | FedAvg of w_C and a_C | `aggregate()`: `aggregate_clients` + aux FedAvg | W1 `weights.client_up`, W3 `weights.aux_up` |
| 4a | Eq. (5), p.3: per client, in parallel with Step 3, w_S,k = w_S^t − η ∇̃F_S,k | the server copy for client k trains on that client's activations (CE at the model output) | `client_step`, server phase on `self.servers[i]` | – |
| – | p.6: "the downlink communication for transmitting the backpropagated signals is not required" | no gradient goes back | – | C3 `cut.grad_down` = 0 |
| 4b | p.4 top-left: "the server-side models are aggregated as w_S = 1/K Σ w_S,k" | FedAvg of the copies on the main server | `aggregate()`: server FedAvg | W5 = **0**: never leaves the main server |
| 5 | p.4: "we obtain w_T = [w_C^T, w_S^T]. This final model is utilized at inference" | test accuracy of w_C → w_S; aux unused | `full_model` | – |

## Communication (closed form)

Per round with N clients, each with b minibatches. Here the minibatches cover all 6,000 images: 23 × 256 + 112.

| Row | Count | Bytes per transfer |
|---|---|---|
| C1 activations | N × b | batch × 8,192 |
| C2 labels | N × b | batch × 8 |
| C3 gradient | 0 | – |
| W1, W2 client model | N each | 2,740,048 |
| W3, W4 aux | N each | 81,960 |
| W5 server copies | 0 | – |

For the benchmark (N = 10, R = 11):

| | Bytes |
|---|---|
| Cut = 8,200 × 6,000 × 10 × 11 | 5,412,000,000 |
| Weights = 2 × (2,740,048 + 81,960) × 10 × 11 | 620,841,760 |
| **Total** | **6,032,841,760 B = 5,753.37 MiB** |

**Table 1 (p.4)** gives the per-round communication as (q|D| + 2α|w|)K. It counts activations only and drops |a| ("|a| ≪ |w|"). We also charge:
- the labels, which the server needs for its CE;
- the aux transfers, which are real transfers.

Without the aux, Weights would be 602,810,560 B.

## Memory

**Client:** the client model and the aux head, each with params, grads and momentum, plus the forward working set. The client's graph is freed before the upload and nothing comes back, so **held-across-cut = 0**.

**Server, per device (table value):**
- the global w_S^t (params);
- the copy being trained for one client (params, grads and momentum);
- the working set for one minibatch.

**Not in the per-device total:** a parallel main server (Step 4a) holds the other N − 1 copies at the same time. That is (N − 1) × 120.18 MiB (params + grads + momentum), listed as a separate line by `server_side_extra_memory()`.

**Sequential = parallel:** our loop runs the clients one after another. The clients share nothing until `aggregate()`: each has its own server copy and optimizer. `test_han_locloss` runs the clients in reverse order and gets a bit-identical model, accuracy and loss.

## Not specified in the paper: our choice

| Item | Our choice |
|---|---|
| Forward schedule. Steps 2–3 read per minibatch; Table 1's latency model forwards the whole local epoch with w_C^t first. | Interleaved per batch: forward → upload → client step → server step on that batch. The bytes are the same under either reading. Accuracy could differ: Eq. (5) writes the server input as g_{w_C,k^t}(x), from the round-start model, while our server trains on the client model as it is after its earlier local steps in the round. |
| Aux architecture: Sec. 3.1 says "we adopt a MLP for a_C"; p.6 says "A single FC layer with 23,050 parameters" (= 2304 × 10 + 10, so we infer Linear(2304, 10) on the flattened client output). | flatten + Linear(2048, 10), no pooling. A pooled head (Linear(128, 10), 5,160 B) would be ≈0.01% of \|w\|, below the paper's range. |
| Weight decay, lr schedule. The theory uses η0/(1+t); the experiments use a fixed lr. | wd 0, fixed lr. |
| Optimizer state across rounds | Each client, aux head and server copy keeps its own momentum buffer; FedAvg replaces the weights only. |
| Aggregation weights: the paper writes 1/K. | The harness's data-size weights, equal on equal IID shards. |
| Labels: Fig. 1(c) only | Sent with every minibatch. |
| Byte precision: the paper counts elements | fp32 activations and weights, int64 labels. |
| Aux-head init | PyTorch default. |
| BatchNorm in FedAvg | every state_dict entry is averaged, running stats included. |
| First download (Step 1, round 0) | not billed, because clients start identical; a download is billed after the last round instead. The count (N × R) and the bytes are the same. |

## Benchmark settings (differences from the paper's experiments)

| | Paper (p.6) | Here |
|---|---|---|
| Model | AlexNet-like CNN / VGG-11, split after conv4 | ResNet-18, middle cut (the same for every method) |
| Clients, sampling | N = 1000, K = 300 per round | 10, all every round |
| Batch, local epochs | 10, 1 | 256, 1 |
| Optimizer | SGD lr 0.01, momentum 0.9 | **the same**: SGD lr 0.01, momentum 0.9, wd 0 |
| Rounds | a simulated time budget (Table 2) | 11 |
| Dataset | MNIST, FMNIST, CIFAR-10 (IID and non-IID: 5 label shards per client) | MNIST IID, 32×32, 3 channels (later non-IID: Dirichlet α = 0.5) |
| Aux FedAvg traffic | left out of Table 1 | charged (W3, W4) |

## Changes from the earlier port

| | Earlier port | Now |
|---|---|---|
| Optimizer | harness Adam lr 1e-3, wd 1e-4 | SGD lr 0.01, momentum 0.9, wd 0 (p.6) |
| Server-copy FedAvg | charged as a round trip per client per round (2 × 42,025,080 B × N × R): "the harness's own extrapolation, not in the paper" | the paper **does** specify it (Step 4b, p.4), on the main server, so it is **not charged**. The rule is the same as SplitFedv1, FedSplitX and MU-SplitFed. |
| Server memory | all N copies counted per device | per device: the global w_S + one copy being trained; the other N − 1 copies on a separate line |
| Breakdown | raw ledger rows | paper-labelled rows C1–C3, W1–W5, each with ✓ |

## Reproduction target (open)

The paper reports Table 2 (p.6) as test accuracy at a fixed **simulated training time**:
- PC = 1, PS = 100, R = 1, β = 0.2;
- N = 1000, K = 300, batch 10, SGD lr 0.01, momentum 0.9.

| Method | MNIST IID | MNIST non-IID | FMNIST IID | FMNIST non-IID | CIFAR-10 IID | CIFAR-10 non-IID |
|---|---|---|---|---|---|---|
| Proposed | 97.73 | 97.01 | 86.70 | 85.74 | 80.72 | 78.91 |

Converting the time budget with Table 1's latency formula gives about **190 rounds** for MNIST (1.5 × 10^11) and about 450–500 for CIFAR-10. This is our estimate.

Reproducing it needs:
- the paper's CNN: 4 client convs 1→32→64→128→256, then server conv 256→256 + FC 2304→1024→512→10; 3×3 convs with bias, no BN; pool positions not given;
- its 1000-client split.

Reproduction is a separate task, and it must pass before the method is ranked (CLAUDE.md).

## Checked by
- **`test/check_accounting.py`, `test_han_locloss`:**
  - exact bytes per category;
  - every breakdown row against the paper's count;
  - the optimizer (SGD lr 0.01, momentum 0.9);
  - the client-order check.
- **`test_all_methods_on_real_model`:** the real ResNet-18 split, the aux-size guard, and no gradient back.
- **The breakdown printed at the end of each run:** rows C1–C3 and W1–W5, each with ✓.
