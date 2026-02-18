+++
title = "Amdahl's Law"
date = 2026-02-18
draft = false
math = true
description = "the serial fraction sets a hard ceiling on parallel speedup."

[params]
    author = "Gene Amdahl"
    year = 1967
    category = "law"
    paper_title = "Validity of the Single Processor Approach to Achieving Large Scale Computing Capabilities"
    paper_url = "https://dl.acm.org/doi/10.1145/1465482.1465560"
+++

You throw 100 processors at a job that takes an hour. But 20 minutes of that hour is code that runs in sequence — one step after the other, no shortcut. Those 20 minutes don't care how many processors you have. So your theoretical best is 20 minutes plus whatever the parallel part takes split across your 100 cores.

Amdahl's law quantifies that ceiling. The serial fraction of your workload sets a hard limit on how much speedup you can ever get, no matter how much hardware you add.

## formula

$$S(N) = \frac{1}{(1 - p) + \frac{p}{N}}$$

where:

- $S(N)$ — theoretical maximum speedup of the whole task using $N$ processors
- $p$ — fraction of execution time that is parallelizable ($0 \leq p \leq 1$)
- $N$ — number of processors
- $(1 - p)$ — the serial fraction, the part that cannot be parallelized

As $N \to \infty$, the maximum speedup converges to $\frac{1}{1 - p}$.

If 5% of your code is serial ($p = 0.95$), you will never exceed a 20x speedup. Ever.

## notes

**Fixed problem size.** Amdahl assumes the total workload stays the same regardless of processor count. You're asking: "given this exact job, how much faster can I finish it?" This makes the law inherently pessimistic about scaling.

**Gustafson's rebuttal (1988).** In practice, when you get more processors you solve *larger* problems, not the same one faster. Gustafson's law flips the framing: with $N$ processors, you tackle $N$-sized work. The two laws aren't contradictory — they model different scenarios.

**Ignores overhead.** The formula doesn't account for communication costs, synchronization, memory contention, or load imbalance. Real speedups are often worse than Amdahl predicts.

**The serial fraction isn't fixed.** A common mistake is treating $p$ as an architectural constant. In practice, the serial fraction depends on problem size, algorithm choice, and implementation.
