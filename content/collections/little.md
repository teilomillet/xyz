+++
title = "Little's Law"
date = 2026-02-18
draft = false
math = true
description = "items in a system = arrival rate times time spent inside."

[params]
    author = "John Little"
    year = 1961
    category = "law"
    paper_title = "A Proof for the Queuing Formula: L = λW"
    paper_url = "https://pubsonline.informs.org/doi/10.1287/opre.9.3.383"
+++

Walk into a coffee shop. Count the people inside: that's $L$. Stand at the door and count how many people walk in per minute: that's $\lambda$. On average, each person stays $W$ minutes.

Little's law says $L = \lambda W$, and it holds for almost any system in steady state — no assumptions about distributions, queue discipline, or service order. It just works. Factories, web servers, hospital beds, highway traffic. If the system is stable, the law applies.

## formula

$$L = \lambda W$$

where:

- $L$ — long-term average number of items in the system
- $\lambda$ — long-term average arrival rate (items per unit time), which in steady state equals the throughput
- $W$ — long-term average time an item spends in the system

Rearranged:

- $W = \frac{L}{\lambda}$ — average time in system
- $\lambda = \frac{L}{W}$ — throughput

## notes

**Remarkably general.** The law is distribution-free. It doesn't depend on arrival patterns, service time distributions, number of servers, or queue discipline (FIFO, LIFO, random). Any system in steady state.

**Steady state is the real requirement.** The system must be stationary: the average arrival rate must equal the average departure rate over the observation window. Everything that enters must eventually leave. If work-in-progress accumulates unboundedly, the law doesn't apply.

**Averages, not snapshots.** $L$, $\lambda$, and $W$ are all long-run time-averages. At any given instant, the relationship may not hold. This is where practitioners trip up — measuring over short windows where the system isn't in steady state.

**Stronger than you'd think.** Little's original 1961 proof assumed stationarity and ergodicity. Stidham (1972) proved a purely deterministic version, showing the law is even more general than the original probabilistic assumptions suggested.
