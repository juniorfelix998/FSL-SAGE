# BP-Free Split Learning Benchmark — Working Context

This repo is the base harness for a benchmark paper comparing backpropagation-free
split-learning methods under one protocol. Read this before starting.

## What this project is
A unified benchmark ("What Crosses the Cut?") of BP-free split / split-federated learning
methods. We do NOT invent new methods. We run existing published methods on identical
conditions (same dataset, backbone, cut, seed, metrics) and rank them. The contribution is
the fair, common measurement layer — not new algorithms.

## The methods

Code-available (USE their code, port into this harness, DO NOT reproduce their published
numbers — just run them in our setting):
- CSE-FSL, FSL-SAGE, SplitFedv1, SplitFedv2  — already in the FSL-SAGE harness (this repo)
- MU-SplitFed  — standalone repo: https://github.com/Johnny-Zip/MU-SplitFed  (needs porting in)
- HO-SFL       — standalone repo: https://github.com/HKU-WILL-Lab/HO-SFL  (needs porting in)

No-code (implement from paper, AI-assisted, MUST reproduce the paper's reported numbers
before entering the benchmark; label these clearly as reimplementations):
- DSL-Aux (non-federated, our closest prior, featured), Han et al. (locally generated
  losses), LocFedMix-SL, FedSplitX, HOSL (non-federated hybrid-order)

## The four metrics (log these uniformly for EVERY method)
1. Communication — CUT: activation + gradient bytes across the cut (a scalar-only method
   like HO-SFL is charged only its scalar bytes, not a full tensor)
2. Communication — WEIGHTS: model-weight / aggregation traffic (≈0 for non-federated methods)
3. Communication — TOTAL: sum of cut + weights (the value we rank on)
   [The existing repo likely logs ONE combined comm number — splitting it into cut vs
    weights is our first instrumentation task.]
4. Accuracy (test acc; RMSE/R^2 for regression datasets later)
5. Memory: peak client + server memory
6. Latency: wall-clock from application start to completion

## Datasets / models (scope)
- Smoke test: MNIST / half-MNIST (this task)
- Main benchmark: CIFAR-10, CIFAR-100 + (later) one more classification set + two regression sets
- Heterogeneity: Dirichlet alpha = infinity (IID) and alpha = 0.5
- Backbone: ResNet-18 main (ResNet-110 for DSL-Aux reproduction); later VGG-style CNN + ViT-Tiny
- Cuts: early / middle / late (main table = middle)
- Seeds: 3 default

## FIRST TASK — MNIST smoke test (Part 0). Do these in order:
1. Get this FSL-SAGE repo running UNMODIFIED on its default CIFAR example for a few rounds.
   Confirm the environment/deps work before changing anything. Report the exact run command.
2. Find and show me the exact place in the code where communication (bytes/MB) is counted,
   and where the training loop and dataset loaders live. This is the hook everything hangs off.
3. Add MNIST (or half-MNIST) as a dataset: resize to 32x32 and repeat the 1 channel to 3 so
   the ResNet backbone is untouched (we are testing plumbing, not accuracy).
4. Run ONE method (CSE-FSL or SplitFedv2), MNIST, ResNet-18, middle cut, IID, 1 seed, a few
   epochs. Goal: the loop COMPLETES.
5. Confirm all metrics log to a file: cut comm, (weights comm if separable), memory, latency,
   accuracy. If communication is one combined number, add a counter that tags bytes as
   cut-traffic vs weight-traffic — this is the key instrumentation step.

Smoke test PASSES when one run completes and all metrics are written to a readable log.
Do not scale to CIFAR or more methods until this passes.

## Rules
- Don't rewrite method algorithms for the code-available methods — port/run their code.
- Keep one shared training loop, one data pipeline, one metric logger across methods.
- Keep a run manifest (git commit, config, seed, dataset, cut) with every run.
- Ask before large refactors; prefer minimal edits for the smoke test.
