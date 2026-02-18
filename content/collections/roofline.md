+++
title = "Roofline Model"
date = 2026-02-18
draft = false
math = true
description = "attainable performance is the minimum of compute peak and memory bandwidth times arithmetic intensity."

[params]
    author = "Samuel Williams, Andrew Waterman, David Patterson"
    year = 2009
    category = "model"
    paper_title = "Roofline: An Insightful Visual Performance Model for Multicore Architectures"
    paper_url = "https://dl.acm.org/doi/10.1145/1498765.1498785"
+++

Your code does some number of floating-point operations for every byte it loads from memory. That ratio is the arithmetic intensity. Now draw two lines on a log-log plot: a flat ceiling at your processor's peak FLOPS, and a diagonal line rising with memory bandwidth. Your code lives under whichever line is lower.

If the diagonal hits first, you're memory-bound — the processor is starving for data. If the ceiling hits first, you're compute-bound — memory is fast enough, you just need more FLOPS. The point where the two lines meet is the ridge point: the minimum arithmetic intensity needed to fully use the hardware.

One picture, two numbers, and you know where the bottleneck is.

## formula

$$P \leq \min(\Pi_{\text{peak}},\ I \times W_{\text{peak}})$$

where:

- $P$ — attainable performance (FLOPS/s)
- $\Pi_{\text{peak}}$ — peak floating-point throughput of the processor (FLOPS/s)
- $W_{\text{peak}}$ — peak memory bandwidth (bytes/s)
- $I$ — arithmetic intensity of the kernel (FLOPS/byte), i.e. floating-point operations per byte transferred from memory

The ridge point occurs at:

$$I_{\text{ridge}} = \frac{\Pi_{\text{peak}}}{W_{\text{peak}}}$$

Below this intensity, performance is memory-bound. Above it, compute-bound.

## notes

**It's a ceiling, not a prediction.** The roofline tells you the *maximum* attainable performance. Real code usually falls below because of instruction-level bottlenecks, cache misses, load imbalance, or poor vectorization. The gap between the roofline and your measured performance is where the optimization opportunity lives.

**Arithmetic intensity is per kernel, not per program.** Different loops in the same code can land in completely different parts of the plot. Profile each hot loop separately.

**The model extends.** You can add lower ceilings for missing optimizations — no SIMD, no ILP, no prefetching — creating a staircase of rooflines. Each step shows what you'd gain by enabling that optimization.

**Bandwidth means DRAM bandwidth.** The peak bandwidth in the original model refers to main memory (DRAM), not cache. If your working set fits in L2, the effective bandwidth is much higher and the diagonal shifts left — but that's a different roofline for a different level of the hierarchy.

**Especially relevant for GPUs and accelerators.** The roofline model became a standard diagnostic tool for GPU computing, where the gap between compute peak and memory bandwidth is enormous and arithmetic intensity determines everything.
