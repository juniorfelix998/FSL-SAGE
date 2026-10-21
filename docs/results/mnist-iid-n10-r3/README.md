# CSE-FSL vs FSL-SAGE: MNIST IID, 10 clients, 3 rounds

Both methods ran under identical conditions: Colab Tesla T4, ResNet-18,
middle cut, MNIST (32×32, 3-channel), IID, seed 200, batch 256, h = 5 (FSL-SAGE
l = 10). Per-method details are in [CSE_FSL.md](CSE_FSL.md) and
[FSL_SAGE.md](FSL_SAGE.md). How each line is defined is in
[../../CSE_FSL_FSL_SAGE_BREAKDOWN.md](../../CSE_FSL_FSL_SAGE_BREAKDOWN.md).

Every transfer count matched the count each paper's algorithm predicts (all ✓).
Units are MiB, which is what the benchmark table calls "MB".

## Headline

| Metric | CSE-FSL | FSL-SAGE | Difference |
|---|---:|---:|---|
| Comm: Cut | 300.29 | 300.29 | identical |
| Comm: Weights | 638.55 | 237.08 | FSL-SAGE −401.47 (−63%) |
| ↳ Server-model FedAvg | 0.00 | 0.00 | both keep one server model |
| **Comm: Total** | **938.84** | **537.37** | **FSL-SAGE −401.47 (−43%)** |
| Cut share of total | 32% | 56% | |
| Client memory | 148.80 | 133.76 | FSL-SAGE −15.04 (client never trains the aux) |
| Held-across-cut | 0.00 | 94.01 | FSL-SAGE stalls on alignment |
| Server memory | 176.26 | 218.60 | FSL-SAGE +42.34 (server trains the aux) |
| System peak (live) | 106.28 | 123.30 | FSL-SAGE +17.02 |
| Latency (s) | 93.99 | 113.64 | FSL-SAGE +19.65 s (+21%) |

FSL-SAGE memory is re-split so the aux model's gradients and Adam state sit on
the server, which is where it is trained. Before the re-split the figures were
client 157.82 and server 186.52; see "Where the aux model lives" below. Those
FSL-SAGE values are recomputed from the same T4 run, so rerun on Colab to
confirm.

Accuracy is not in the pasted output. Add it from each run's
`Round 2 … ts. acc` log line.

## Communication, line by line

| # | What is sent | CSE-FSL count | CSE-FSL MiB | FSL-SAGE count | FSL-SAGE MiB |
|---|---|---:|---:|---:|---:|
| C1 | Smashed activations, client → server (every 5th batch) | 150 | 300.00 | 150 | 300.00 |
| C2 | Labels with those batches | 150 | 0.29 | 150 | 0.29 |
| C3 | Gradient back to client | 0 | 0.00 | 0 | 0.00 |
| W1 | Client model up (FedAvg) | 30 | 78.39 | 30 | 78.39 |
| W2 | Averaged client model down | 30 | 78.39 | 30 | 78.39 |
| W3 | Aux model up | **30** | **240.88** | **0** | **0.00** |
| W4 | Aux model down | **30** | **240.88** | **10** | **80.29** |
| W5 | Server-model FedAvg | 0 | 0.00 | 0 | 0.00 |
| | **Total** | | **938.84** | | **537.37** |

**The entire 401.47 MiB gap is aux-model traffic (W3 + W4).**
- **CSE-FSL** averages its aux head like a client model: every client sends it
  up and down **every round**. That is 10 × 3 × 2 × 8.03 = 481.76 MiB.
- **FSL-SAGE** never averages the aux model. The server trains it ("aligns"
  it) and sends it down **once every l = 10 rounds**. With 3 rounds that
  happens once: 10 × 8.03 = 80.29 MiB.

Everything else is identical: the cut traffic (same h = 5 schedule) and the
client-model FedAvg. Neither method ever sends a gradient back (C3 = 0), and
neither pays Server-model FedAvg (W5 = 0), because both keep a single server
model.

**How the gap grows with rounds:** CSE-FSL's aux cost rises by 160.59 MiB
every round. FSL-SAGE's rises by 80.29 MiB once every 10 rounds. At 200 rounds
that is 32,118 MiB vs 1,606 MiB of aux traffic.

## Memory, line by line

### Where the aux model lives (why the two methods differ)

| | CSE-FSL | FSL-SAGE |
|---|---|---|
| Who **trains** the aux model | the **client**, every batch, on its local loss (Step 3, Eq.5) | the **server**, every l rounds, by alignment (Alg.3 l.9–13) |
| Who **runs** it | client | client: it estimates the cut gradient every batch (Alg.1) |
| On the client | params + grads + Adam = 32.09 | params only = 8.03 |
| On the server | nothing between rounds | the copy being aligned with its grads + Adam = 32.09 (plus the other 9 clients' copies, listed separately) |

### Client (worst single client)

| Item | CSE-FSL | FSL-SAGE |
|---|---:|---:|
| Client model x_c (params + grads + Adam) | 10.43 | 10.43 |
| Aux model | 32.09 (params + grads + Adam) | 8.03 (params only) |
| Working-memory peak | 106.28 | 115.30 |
| ↳ of which activations | 103.28 | 112.30 |
| **Client total** | **148.80** | **133.76** |
| Held-across-cut | 0.00 | 94.01 |

- **FSL-SAGE's client is lighter by 15 MiB.** It keeps the aux model only to
  run it, so it doesn't need its gradients or Adam state (−24 MiB). Its working
  memory is 9 MiB higher, because the aux model produces the gradient through
  a second-order graph (`create_graph=True`).
- **Held-across-cut (the clearest behavioural difference):**
  - CSE-FSL never waits for the server.
  - FSL-SAGE waits on alignment iterations: its client keeps 94 MiB of
    activations alive until the newly aligned aux model arrives.
  - With R = 3 this happens once (round 0), but the column reports the maximum.

### Server

| Item | CSE-FSL | FSL-SAGE |
|---|---:|---:|
| Server model x_s (params + grads + Adam) | 160.22 | 160.22 |
| Aux model being aligned (params + grads + Adam) | – | 32.09 |
| Working-memory peak | 16.04 | 26.29 |
| ↳ of which activations | 14.04 | 22.29 |
| **Server total** | **176.26** | **218.60** |
| *Not in total:* alignment buffers × 10 clients | – | 98.20 |
| *Not in total:* aux models of the other 9 clients | – | 288.79 |

- Both keep **one** server model (160.22 MiB).
- FSL-SAGE's server also holds the aux model it is aligning (32.09), plus
  ~10 MiB more working memory for the stored z_f buffer and the alignment
  pass.
- FSL-SAGE's protocol also keeps, **per client**, an alignment buffer and the
  other clients' aux models with their Adam state. These scale with N: 387 MiB
  at 10 clients, so a real server would need ≈ 606 MiB. They are listed
  separately so the headline stays a per-device peak.

### System peak

| | CSE-FSL | FSL-SAGE |
|---|---:|---:|
| Live bytes, both sides at one instant | 106.28 | 123.30 |
| + static model/optimizer bytes | 309.02 | 334.07 |

## Time

| | CSE-FSL | FSL-SAGE |
|---|---:|---:|
| **Latency (s)** | **93.99** | **113.64** |
| Client compute (s) | 1.20 | 0.66 |
| Server compute (s) | 0.22 | 3.05 |
| Aggregation compute (s) | 0.42 + 0.21 aux | 0.41 |

FSL-SAGE trades server compute (alignment) for communication. The client does
less work: CSE-FSL trains its aux head every batch, while FSL-SAGE's is trained
by the server.

## Comparison with the 2-client table

Before the aux re-split, the memory numbers (148.80 / 157.82 client, 0 / 94.01
held, 176.26 / 186.52 server, 106.28 / 123.30 system) were **the same** as the
2-client benchmark table, as expected: memory is per device. With the re-split,
the table's FSL-SAGE Client becomes 133.76 and Server becomes 218.60. CSE-FSL
is unchanged.

Communication changes with N:

| | 2 clients (table) | 10 clients (this run) |
|---|---:|---:|
| Batches per client per round | 118 | 24 |
| Smashed uploads per round (all clients) | 2 × 24 = 48 | 10 × 5 = 50 |
| Cut (both methods) | 288.28 | 300.29 |
| CSE-FSL Weights | 127.71 | 638.55 |
| FSL-SAGE Weights | 47.42 | 237.08 |
| CSE-FSL Total | 415.99 | 938.84 |
| FSL-SAGE Total | 335.70 | 537.37 |

- **Cut stays about the same:** the same 60,000 images are uploaded every 5th
  batch, whoever holds them.
- **Weights grow 5×:** with 5× more clients, there are 5× more model
  exchanges.
- As a result, FSL-SAGE's advantage grows from 19% (2 clients) to 43%
  (10 clients).
