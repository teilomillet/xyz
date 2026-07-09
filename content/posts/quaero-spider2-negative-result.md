+++
title = 'Why Trained Open Models Score Zero on Spider 2.0-DBT'
date = 2026-07-09
draft = false
description = 'Spider 2.0-DBT drops an AI agent into a real dbt data-transformation project and scores its output tables against hidden gold tables — exact match, no partial credit. The leaderboard is held by harnesses around closed frontier models (65.6%, Claude Sonnet 4.6). I spent four time-boxed weeks trying to beat it with trained open models instead: 279 commits, 140 training configs, 94 tracked runs, models from 4B to 35B. The trained models do the entire job on manufactured lookalike tasks (8/8) and score 0 on every real instance tested: everything locally checkable passes, the hidden check fails. A negative result with a diagnosis — the failure is at the reward layer, not the protocol layer — plus a handoff for continuing on this benchmark and a playbook for training open models on agentic benchmarks. Code available on request; will open-source with demand.'
tags = ['reinforcement-learning', 'RLVR', 'spider2', 'text-to-sql', 'dbt', 'negative-result', 'playbook', 'quaero', 'retrain']

[params]
    nextprev = true
    taglist = true
+++

> **Authorship.** This paper was written by Claude (Fable 5, via Claude
> Code) from the versioned record of Teïlo Millet's private `quaero`
> repository — dated diagnostics, signed training manifests, evaluation run
> directories, and Weights & Biases runs produced between 2026-06-11 and
> 2026-07-06. The experiments are his; the prose is the model's. It is
> written in the first person, and where a claim rests on a single
> artifact, the artifact is named.

## Summary

[Spider 2.0-DBT](https://spider2-sql.github.io/) is a benchmark for AI
agents doing real data-engineering work. Each task drops the agent into a
data-transformation project built with dbt — the standard tool for SQL
pipelines — and asks it to produce the correctly transformed output
tables. The answer is scored by comparing those tables against hidden
correct ones (the *gold tables*) that the agent never sees. Exact match,
pass or fail, no partial credit. The top score, 65.6%, is held by a
harness — agent scaffolding — around Claude Sonnet 4.6, and the entries
below it wrap GPT-5 and other closed models. Nobody on the leaderboard had
trained a model for the domain.

Between June 11 and July 6, 2026, I tried to close that gap with trained
open models instead. The benchmark ships no training data — its 68 tasks
are the test, and training on them would invalidate everything — so
training data had to be manufactured from scratch: new synthetic tasks
that copy the benchmark's *mechanics* (same tools, same episode shape,
same scoring contract) but none of its content, generated over schemas and
domains the benchmark does not contain, with a mechanical firewall (§2)
guaranteeing the separation. Each manufactured task is generated together
with its own correct answer, so every training attempt can be scored
automatically. That automatic score is the
*reward* the training climbs. The training itself was small and cheap:
LoRA adapters (add-on weight files of ~41 MB, adjusted instead of the
whole model) on open models from 4B to 35B parameters, one rented GPU at
$1.20/hr. Four time-boxed weeks produced 279 commits, 140 training
configurations, and 94 tracked runs.

The result is negative, and the failure is precisely localized. On
manufactured tasks held out from training, the trained 4B model does the
entire job, eight episodes out of eight — inspects the project, writes the
transformation, builds it, validates, submits — and the tables it submits
are the correct ones. On real benchmark tasks, the same procedure runs
just as cleanly, at every scale, trained and untrained: the project
builds, the model's own validation passes, the submission is well-formed.
From the inside, it looks like success. It is not, because the benchmark
does not score the procedure; it scores the numbers. Somewhere in the
submitted tables a value, a row, or a whole table differs from the hidden
gold answer — the score is 0, and the scorer does not say where. Every
model, every scale, every instance tested. The models do everything they
can check. What they cannot check is whether the answer is right.

The diagnosis is a failure every engineer has met: the tests pass and
production breaks, because the tests did not measure what production
measures. Here the mismatch is structural. Real enterprise tasks encode
business rules — which accounts count as active, which orders count as
paid — and those rules exist only in the hidden gold tables. Nothing the
model or its training loop can compute locally contains them. Training
improves whatever signal it is given, and here every available signal
stops short of the answer — so the models stop short of it too. The
8/8-versus-0 gap is that distance, measured.

This paper is written for people who want to do what I did — train an open
model on an agentic benchmark. Sections 1–7 describe what was built, what
happened, and why it stopped. Section 8 is a handoff for anyone continuing
on this benchmark. Section 9 is the playbook for starting on any other.
The code is private but available on request, and will be open-sourced if
there is demand.

## 1. The benchmark and the opening

Spider 2.0-DBT is the repository-level slice of Spider 2.0. Each of its 68
instances is a full dbt project — Fivetran-style packages,
`dbt_project.yml`, a starting DuckDB — and the agent must produce
transformed tables that diff clean against hidden gold tables. Episodes
run roughly 92–324 agent turns. Scoring is pass/fail per instance, no
partial credit.

The leaderboard on June 11, the day of quaero's first commit:

| Rank | Agent | Score | Model |
|---|---|---|---|
| 1 | SignalPilot | 65.6% | Claude Sonnet 4.6 |
| 2 | Databao (JetBrains) | 60.3% | undisclosed |
| 3 | Shadowfax | 41.2% | GPT-5 |

Every entry at the top is a harness around a frontier API model. Nobody on
the leaderboard had trained a model on the domain, while the tooling for
doing so — verifiable environments, open 4B–27B bases cheap to fine-tune,
reproducible RLVR trainers — had matured. That absence was the opening. A
small open model trained to competitive performance here would say
something general: that this kind of competence can be trained into an
open model instead of rented from a closed one.

The project charter (`AGENTS.md`) set the target at ≥ 43/64 and the
constraint that shapes everything below: one person, alongside other work,
roughly four weeks. The stopping rule was the calendar, fixed in advance.

## 2. Reproducing the evaluation

The first week went entirely to reproducing the official evaluation
locally and characterizing its scorer. Nothing else in the project could
be trusted without this, and it surfaced two facts that anyone entering
the benchmark should have:

**Four of the 68 instances have no gold database** in the official
`dbt_gold.zip` (`airbnb002`, `biketheft001`, `gitcoin001`,
`google_ads001`). They score 0 regardless of the agent. 68 − 4 = 64, which
is exactly SignalPilot's denominator: 42/64 = 65.62%.

**Leaderboard percentages are not comparable.** Databao's 60.29% is 41/68;
another entry's 37.50% is 24/64. The official `evaluate.py` also divides
by the number of *submitted* instances, so partial submissions inflate
scores. My wrapper reports /64 and /68, always.

The scorer's exact behavior matters because the training reward has to
reproduce it. `duckdb_match` works table by table. For every table named
in the answer key, the submission must contain a table with the same name
and exactly the same number of rows. The required columns are then checked
one at a time: a gold column passes if *some* column of the submitted
table holds the same values — column names don't matter, extra columns and
extra tables don't matter, numbers only need to match within 0.01, and one
submitted column may satisfy several required columns at once (matched
columns are not set aside). And in every comparison the suite performs —
all 119 tables across all 68 instances set `ignore_order=true` — each
column is sorted on its own before being compared.

Sorting each column on its own discards the row structure, and the
consequence is easiest to see in a demonstration run against the official
code. The gold table pairs customers to revenue. The submitted table
contains the same values with every pairing wrong:

```
gold                          submitted
customer   revenue            customer   revenue
alice      100                alice      375
bob        250                bob        100
carol      375                carol      250

official duckdb_match score: 1
```

Because each column is sorted separately, what the scorer actually
compares is customers = {alice, bob, carol} — correct — and revenue =
{100, 250, 375} — correct. It never asks whether alice's revenue is 100.
Every value is right, every row is wrong, and the submission passes.
(Verified directly against the benchmark's `eval_utils.py`. With
`ignore_order=false` the same submission scores 0 — but no table in the
suite uses that setting.) This is lax to the point of being a bug. The
environment's verifier reproduces it anyway, laxness included, because the
reward must be the official score, not a corrected version of it.

The laxness also sharpens everything that follows. Every rule above errs
toward accepting: names ignored, any column can match, rows need not line
up, numbers get a tolerance. The official check is *easier* than exact
correctness — and the models in §5 still never passed it once. The zeros
in this paper were scored by a lenient judge.

The same week fixed the contamination policy. The 68 eval instances are
the only real data in existence, and they are eval-only; enforcement is
mechanical because discipline degrades under deadline pressure. Three
layers: every task carries a provenance tag (`eval`/`factory`/`toy`) that
survives the wire protocol; the training-side environment raises if asked
to build a dataset from eval-tagged tasks; a linter, fed by an exclusion
list derived mechanically from the benchmark (29 domain packages, 52
project names, 52 database families, 68 instance ids), rejects factory
tasks that overlap. No number in this paper is contaminated.

## 3. The training system

Spider 2.0-DBT ships no training split, and Spider2-lite publishes no gold
SQL. Training data therefore had to be manufactured: a distribution that
matches the benchmark's mechanics while sharing zero content with it. The
system that does this is quaero. The code is private (available on
request), but the architecture is the transferable part:

```
retrain (GRPO/RLVR/SFT trainer, TOML-first)
   │  provider = "verifiers"
   ▼
quaero-dbt  (installable verifiers environment)
   │  OpenEnv wire protocol (typed Action/Observation, FastAPI)
   ▼
quaero env server
   ├── tasks: TaskSpec with provenance tag  eval | factory | toy
   ├── workspace: write policy enforced by the environment
   ├── tools: list_files · read_file · query · build_spec ·
   │          write_file · dbt · validate · submit
   ├── episode state machine + text-action protocol
   └── verifier: mirrors official duckdb_match semantics exactly
        ▲
   eval drivers (open models via OpenAI-compatible APIs; Claude API baseline)
```

The agent's world is eight tools, enforced by the environment rather than
requested by the prompt. An episode is a state machine ending at `submit`,
which triggers the same verifier semantics the official benchmark uses.

Training tasks came from a task factory: synthetic dbt/DuckDB agent tasks
built to be axis-equivalent to Spider2-DBT — same tool surface, episode
shape, and verifier contract — over schemas and domains the benchmark does
not contain. A second pipeline had an open model (GLM-5.2 via OpenRouter)
author text-to-SQL surrogates, keeping only candidates whose SQL executed
on both SQLite and DuckDB: generated gold, but execution-verified. As
failure modes appeared in evaluation, the factory grew matching repair
curricula — repo inspection, build-spec repair, semantic repair,
validation repair, dbt-error repair, trace-failure repair.

All runs went through [`retrain`](https://github.com/teilomillet/retrain),
an open-source TOML-first trainer: one TOML file per run, a one-step
capacity gate before overnight runs, resumable checkpoints. Adapters were
PEFT LoRA (r=8, α=16) on Qwen3.5-4B and later Qwen3.5-27B, on rented RTX
PRO 6000 Blackwell 96GB pods at $1.20/hr, spun up per run and verified
torn down. Every claim-bearing run has a signed manifest recording dataset
SHA-256, config, pod id, and a `claim_under_test` field. The totals — 140
configs, 94 tracked runs, 16 manifests — are why a negative result this
specific can be reported at all.

## 4. What the models learned

Progress was measured with a failure taxonomy that classifies every
evaluated episode. The categories form a ladder:

1. `no_submission_repeated_action` — loops on a tool call, never submits
2. `no_submission` — works, runs out of turns
3. `submitted`, dbt build fails
4. `submitted`, build ok, local validation fails
5. `submitted_official_fail` — build ok, local validation passes, official
   gold diff scores 0
6. official pass

On the synthetic distribution, the ladder was climbed completely. Early
checkpoints looped at rung 1; invalid-action gates and repair curricula
moved successive checkpoints upward; several branches that looked
plausible from the loss curve were classified as regressions to rung 1 and
killed the same day. The climb ended at the repo-bridge checkpoint (step
352, a 41 MB LoRA on Qwen3.5-4B trained on 704 SFT rows from 64 accepted
trajectories): on held-out factory tasks it went 8/8 — every episode
submitted, mean reward 1.0, zero invalid actions, and the intended tool
sequence (`list_files`, 4× `read_file`, `query`, `build_spec`,
`write_file`, `dbt`, `validate`, `submit`) reproduced on every episode. A
harder mixed held-out gate (`schemafinal`, the project's best checkpoint
by that measure) passes 5/8.

The record for that checkpoint
(`docs/qwen35-repobridge-publication-candidate-2026-07-02.md`) opens with
"Not publishable yet": the evidence was held-out but same-tier — tasks
from the same factory that produced the training data. The distinction
turned out to be the entire project.

## 5. Results on real instances

Every real-Spider2 evaluation in the record, June 11 – July 6:

| Model | Training | Instance(s) | Build | Local valid. | Official | Category |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | SFT `fanoutsource4164` | `airport001` | ✅ | ✅ | **0** | `submitted_official_fail` |
| Qwen3.5-4B | SFT `fanoutsemantic5444` | pilot | — | — | **0** | `no_submission_repeated_action` (regression) |
| Qwen3.5-4B | SFT `fanoutmultisemantic4804` | pilot | | | **0** | schema-contract failure |
| Qwen3.5-4B | SFT `schemawide640` | pilot | — | — | **0** | regressed from submit boundary |
| Qwen3.5-4B | SFT `fanoutsourcepres-refresh192` | pilot | — | — | **0** | repeated query loops |
| Qwen3.5-27B | base, no training | `airbnb001` | ✅ | ✅ | **0** | `submitted_official_fail` |
| Qwen3.5-27B | base | 8 dev episodes | | | **0/8** | run halted after 8 zeros |
| Qwen-AgentWorld-35B-A3B | base | `airbnb001` | — | — | **0** | `no_submission_repeated_action` |
| Qwen3.5-27B | SFT 320 steps (`fanoutmultisemantic4804`) | `airbnb001` | | | **0** | `submitted_official_fail` |
| Qwen3.5-27B | SFT `schema-contract96` | `airbnb001` | ✅ | ✅ | **0** | `submitted_official_fail` (45 turns) |

Sources: `training/qwen35_27b_fanoutmultisemantic_20260705.manifest.json`
(including its `debunked_4b_branches` list),
`training/qwen35_27b_capacity_diagnostic_20260704.manifest.json`,
`eval_runs/prime_schema_contract_20260705/RESULT.md`, and the per-run
taxonomy artifacts. The final manifest's status field reads
`executed_no_spider2_win`.

**One episode from the record, concretely.** July 5, the
`schema-contract96` checkpoint (27B) on the real instance `airbnb001`: 45
environment turns.
The model inspected the project, hit dbt errors eight times along the way
and continued past each, made five invalid tool calls and recovered,
reached a build that succeeded, passed local validation, and submitted.
Official score: 0. Every check the model could run passed. The one check
it could not run — the comparison against gold — failed.

That is the pattern to read out of the table: the best checkpoints and the
larger base models all reach rung 5 — build succeeds, local validation
passes, submission is well-formed — and none reaches rung 6, on any
instance, at any scale.

## 6. Diagnosis

Three hypotheses could explain the table. The record rules out two.

**Insufficient model capacity — mostly ruled out.** The untrained 27B base
already reaches rung 5 on `airbnb001`: it builds, validates, submits, and
scores 0. Eight dev episodes with the 27B base all scored 0. Fine-tuning
the 27B on the strongest available curriculum did not change the outcome,
and the 35B-A3B MoE performed worse than the dense 27B, not better. Scale
changed which rung a base model starts on; it did not change the outcome.
Whether a far larger open model clears rung 5 is untested here.

**Inability to operate the environment — ruled out.** Tool use is what
transferred. Repo inspection, dbt builds, the build-spec/validate/submit
protocol, and error repair all work, on synthetic tasks (8/8 with the
intended tool mix) and on real instances (clean submissions with passing
builds).

**Misaligned training signal — supported.** On factory tasks, gold tables
are built from each task's own gold SQL, so reward, validation, and score
coincide — and the models max them out. On real instances, gold is hidden
and "local validation" means dbt tests and schema checks. The project's
final run record states the finding directly:

> "The useful evidence is that the model can submit after local
> validation, but local validation still does not predict the official
> Spider2 diff on `airbnb001`. The next optimization target is
> validation/reward alignment against official diff failure, not more
> broad SFT throughput."
> — `eval_runs/prime_schema_contract_20260705/RESULT.md`

What occupies the gap is business semantics: domain logic — which accounts
count as active, which orders count as paid — that exists in the gold
tables and in no locally-checkable signal. (Those two examples are not
hypothetical; they are the names of actual repair slices in the final 27B
training run: `missing_active_account_filter`,
`missing_paid_status_filter`.) The late curricula targeted these patterns
directly, but this treats instances of the problem rather than the
problem: the list of such failures is only discoverable by failing against
gold, and the eval set is — correctly — off-limits. The official scorer
compounds the difficulty: it returns one bit per instance. The `airbnb001`
record holds the complete submitted database and the verdict — 0 — with no
indication of which table, column, or row differed.

The general form of the finding:

**For RLVR on agentic benchmarks, synthetic-task transfer fails at the
reward layer, not the protocol layer. If the locally-computable reward
does not predict the benchmark's hidden check, SFT and RL optimize the
model to the reward's ceiling — rung 5, never rung 6 — while held-out
synthetic scores remain excellent.**

The two halves of the evidence sit in §4 (the 8/8: `repobridge` step 352
on held-out factory tasks) and §5 (the 0: every real instance, every
scale). The gap between them is not a step on the way to a result; it is
the result — a measurement of reward misalignment. It is also, so far,
n = 1: one benchmark, one system. Confirming or disconfirming instances
from other benchmarks are invited (§10).

## 7. Why the project stopped

The four-week allocation ran out on July 6. That is the entire reason. The
thesis is not dead; its remaining branches each have an expected
time-to-signal of weeks, and those weeks were committed elsewhere before
the project began.

The distinction that matters for anyone reading this as a verdict: what
was falsified is narrower than what was abandoned.

*Falsified by the record:* synthetic-surrogate SFT at the 4B–27B scale,
under a correctly firewalled eval set, does not move the official Spider2
score off zero — even when it fully teaches the protocol.

*Abandoned unfunded, not falsified:* distribution-matched training data
from real dbt repositories; a formal, audited dev-signal budget; larger
open models; RL against a reward that predicts the official diff.

Whoever continues starts with the dead ends already mapped — four weeks
and a few hundred GPU-hours that do not need to be spent again. The
unfunded branches are open.

## 8. Continuing on Spider 2.0-DBT

**Do not re-run these.** From the final manifest's `debunked_4b_branches`:
`fanoutsemantic5444` (regressed to no-submission loops),
`fanoutmultisemantic4804` (scored 0; schema feedback later exposed
schema-contract failure), `schemawide640` (regressed from the submit
boundary), `fanoutsourcepres-refresh192` (repeated query loops). Also
tested and negative: the 27B base as-is, SFT on the 27B, and the 35B-A3B
MoE (worse than the dense 27B).

**Start from these facts.** 64 scoreable instances, not 68. All scored by
`duckdb_match` with the semantics in §2. Episodes run 92–324 turns. The
top harness scores 42/64. Base open models already reach
`submitted_official_fail`: the entire problem is rung 5 → 6.

**The three branches most likely to move the score:**

1. **Distribution-matched tasks with buildable gold.** Derive training
   tasks from real public dbt repositories, where gold tables can be built
   from the repo's own logic rather than guessed. This restores
   reward/score coincidence on realistic business semantics and attacks
   the surviving hypothesis directly. It is the branch I would fund first.
2. **A legitimate dev-signal budget.** Treating all 68 instances as
   untouchable was correct, but a formal train/dev firewall with a small,
   audited dev budget for failure-mode discovery is compatible with
   non-contamination. The semantics of "active account" and "paid order"
   are only discoverable by failing against gold somewhere.
3. **Establish that rung 6 exists.** One official pass on one real
   instance, by any open model at any scale, converts the misaligned
   reward from a wall into a gradient. Nobody has published this.

**Availability.** The environment, verifier, task factory, contamination
linter, eval stack, taxonomy, and every manifest: on request, open-sourced
if there is demand. Trained adapters and SFT datasets are mirrored to
private Hugging Face repos with model cards stating the non-contamination
boundary. Per the project's own publication rule, none of it is labeled a
"Spider2 model," because no Spider2 evidence supports that label.

## 9. Training an open model on an agentic benchmark

The playbook, in the order that matters:

1. **Reproduce the official scorer first.** Before the environment, before
   the data. Characterize its exact semantics and its bugs — there will be
   some (§2). Your verifier must mirror it, laxness included.
2. **Make contamination mechanical.** Provenance tags on every task,
   training-side refusal, a derived exclusion list. A firewall made of
   discipline fails silently and invalidates everything downstream.
3. **Build the failure taxonomy before training.** Without it, progress
   and regression are indistinguishable; several of this project's
   branches looked plausible from the loss curve and were regressions.
4. **Gate and manifest every run.** A one-step capacity gate before every
   overnight run; a manifest with a `claim_under_test` field for every run
   that produces a claim. This is what makes stopping cheap and publishing
   possible.
5. **Run the reward-alignment test in week one.** The lesson that cost
   this project four weeks and is testable in days: before scaling
   synthetic data, run a base model on a handful of real instances and
   check whether your locally-computable reward predicts the official
   score. If episodes that pass your local signal score 0 officially,
   stop: training will climb to your reward's ceiling, not the
   benchmark's. Everything else is downstream of this one correlation.
6. **Budget for signals, not compute.** The full loop — data generation,
   LoRA SFT at 4B–27B, real-benchmark evaluation — fits on one $1.20/hr
   GPU pod. The scarce resource is time, measured in honest signals per
   week.

## 10. Contact

If you are training — or want to train — an open model on an agentic
benchmark, this document is the four weeks I can hand you. Start with §9;
if you hit the reward-alignment wall on your own benchmark, I would like
to hear about it, because each new instance sharpens the general claim
in §6.

If you are continuing on Spider 2.0-DBT: the map above is the handoff, and
the code and artifacts are one email away. Open-sourcing happens when
there is demand.

If you are building RLVR stacks: every experiment here ran through
[`retrain`](https://github.com/teilomillet/retrain), which is open source
today. The training stack was not the failure mode — capacity gates,
resumable SFT, OpenEnv/verifiers integration, and manifest discipline held
under four weeks of daily use.

[teilomillet@gmail.com](mailto:teilomillet@gmail.com) ·
[github.com/teilomillet](https://github.com/teilomillet) ·
[@teilomillet](https://twitter.com/teilomillet)

---

*Written by Claude (Fable 5, Claude Code) on 2026-07-09, from the
experimental record of the `quaero` repository. Experiments,
infrastructure, and evaluation records: Teïlo Millet. Errors of synthesis
are the writer's; the numbers are the record's.*
