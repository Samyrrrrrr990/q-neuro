# Q-Neuro: From Helix to Nova

### Neural Computation, Falsification, Architecture Discovery, and the Search for Machine Intelligence

---

## How to use this book

This is written to take you from high-school mathematics to the point where you could run this
research programme yourself — and, more importantly, to the point where you could **catch yourself
being wrong**, which is the harder skill and the one the whole programme is really about.

**Scope, stated honestly.** Books I, II, IX, X, XII, XIII and XIV cover established material. They
are written as *orientation with worked intuition and exercises* — enough to follow everything else
here and to know what to read next — not as replacements for a linear algebra course or a molecular
biology text. Books III–VIII, XI, XV and XVI are the parts only this programme can teach, and they
are written at full depth, because the failures are original even though the mechanisms are not.

**Every number in this book is real.** Each comes from a stored artifact in this repository. Where
a claim is uncertain, it says so. Where a claim died, the book says how.

**Exercises** are marked ▶. Answers to the ones with definite answers are in Book XVI.

---

# BOOK I — MATHEMATICS FOR NEURAL MACHINES

## I.1 Why this book starts here

Everything in Q-Neuro is one of four operations: multiply a vector by a matrix, apply a nonlinearity,
take a derivative, or average over randomness. If those four are comfortable, the rest is
bookkeeping.

## I.2 Vectors, matrices, and what a layer is

A vector `x ∈ ℝ^d` is a list of `d` numbers. A matrix `W ∈ ℝ^{m×d}` maps `ℝ^d → ℝ^m` by
`(Wx)_i = Σ_j W_ij x_j`. A neural network layer is exactly this plus a shift and a nonlinearity:

```
h = φ(Wx + b)
```

**The one fact that matters most:** `W` is *linear*, so `W(ax + by) = aWx + bWy`. Composing linear
maps gives a linear map, which is why `φ` has to be there — without it, a hundred layers collapse to
one.

▶ **I.1** A `Linear(64, 128)` followed by `Linear(128, 64)` with no nonlinearity between them. What
single layer is it equivalent to? What is the rank of that layer's matrix, at most?

## I.3 Eigenvalues, and why models blow up

For a square `A`, if `Av = λv` then `v` is an eigenvector and `λ` its eigenvalue. Iterating `x_{t+1}
= Ax_t` gives `x_t = A^t x_0`, which grows like `λ_max^t`. That is the whole theory of why recurrent
computations explode or vanish.

This is not decoration. Q-Neuro measured it: gradient descent on a quadratic is stable exactly when
`ρ = η·λ_max(H)/2 < 1`, and across a **1.4% change** in a reparameterisation scale, paired divergence
moved **fourteen orders of magnitude** (Book IV).

▶ **I.2** The diagonal state-space layer in `nova/zoo.py` uses `h_t = a·h_{t-1} + u_t` with
`a = sigmoid(θ)`. Why is `a ∈ (0,1)` by construction the point? What would `a > 1` do over 64 steps?

## I.4 Derivatives, gradients, and Jacobians

The derivative of `f: ℝ → ℝ` is the local slope. The **gradient** `∇f` of `f: ℝ^d → ℝ` is the vector
of partial derivatives — the direction of steepest increase. The **Jacobian** `J` of `f: ℝ^d → ℝ^m`
is the `m×d` matrix `J_ij = ∂f_i/∂x_j`. The **Hessian** is the matrix of second derivatives, and its
eigenvalues are the curvature.

Backpropagation is the chain rule applied right-to-left because that is cheaper when the output is a
scalar.

▶ **I.3** For `L = ½‖Wx − y‖²`, derive `∂L/∂W`. Now explain why `Jᵀ J` appears in Gauss–Newton
methods and what its rank has to do with Book IV's dimension law.

## I.5 Probability, likelihood, and the loss you actually minimise

Cross-entropy loss is negative log-likelihood: if the model puts probability `p_y` on the true class,
the loss is `−log p_y`. Minimising it is maximising the likelihood of the data.

**Why this matters for halting.** Q-Neuro 3.0's halting objective is a likelihood over *which step
fires first*:

```
log P(first fire at k) = log p_k + Σ_{j<k} log(1 − p_j)
```

The masses sum to at most one, falling short by exactly `Π(1 − p_j)` — the probability of never
firing. That shortfall is *information*, not an error, and renormalising it away would be a bug.

▶ **I.4** Verify that identity algebraically for a length-3 sequence, then check it numerically
against `qneuro3.adaptive.first_arrival`.

## I.6 Information, entropy, and calibration

Entropy `H(p) = −Σ p_i log p_i` measures uncertainty. A model is **calibrated** if, among predictions
made with confidence 0.8, about 80% are right. Expected calibration error bins predictions by
confidence and measures the gap. Q-Neuro 3.0's final model measures ECE 0.0018–0.0021 — well
calibrated, and that is reported because a fast model that is confidently wrong is worse than a slow
one.

## I.7 Optimisation

Gradient descent: `θ ← θ − η∇L`. Adam adapts a per-parameter step from running estimates of the
gradient's first and second moments. **Adam's update is scale-free in the gradient**, and SGD's is
not — a fact that turns out to have real consequences (Book IV, §IV.3).

## I.8 Order statistics — the piece nobody teaches and Q-Neuro needed

If `X_1 … X_n` are independent with CDF `F`, the maximum has CDF `F(x)^n`, so

```
E[max] = Σ_k k · (F(k)^n − F(k−1)^n)
```

This one formula is the entire ceiling on batched adaptive computation (Book V). For a halt
distribution `P(k) ∝ 0.8^k` on 1..32: `E[halt] = 4.97` but `E[max]` over a batch of 64 is **20.96**.

▶ **I.5** Compute `E[max]` for that distribution at batches 1, 8, 256. Check against
`qneuro3.adaptive.expected_max_halt`. Why does the realisable speedup fall from 6.43× to 1.24×?

## I.9 Complexity, and what "cheap" means

`O(n)` vs `O(n²)` matters, but on real hardware constants dominate at small sizes. Q-Neuro measured
`c_launch = 119.65 µs` per iteration against `c_step = 2.66 µs` per example-step — **the fixed cost
of starting a step is 45× the cost of one example's work.** Asymptotics would have told you nothing
useful here.

---

# BOOK II — HOW MODERN AI COMPUTES

## II.1 The MLP

Stack `h = φ(Wx + b)`. Universal approximation says a wide enough one-hidden-layer MLP can
approximate any continuous function — and says nothing about whether gradient descent will find it,
or how much data it needs. Treat universality claims with suspicion; they are almost never the
binding constraint.

## II.2 Recurrence: GRU and LSTM

A recurrent network carries a state: `h_t = f(h_{t-1}, x_t)`. The LSTM adds a cell state with
gates controlling what is forgotten, written and read.

**The property that matters for this book:** the update `f` does not depend on `t`. A recurrent
network that learns the right update implements a finite automaton *exactly*, and an automaton does
not care how long the input is.

That is not a theoretical remark. Nova measured it: an LSTM trained on lengths 8–16 scores **1.000 on
parity and 0.992 on modular sum at length 64** — perfect extrapolation. No attention-based model
tested exceeded 0.39 on modular sum.

## II.3 Attention and the transformer

Attention computes `softmax(QKᵀ/√d)V`: every position forms a query, compares it against all keys,
and reads a weighted mixture of values. It is permutation-equivariant, so position must be injected
— learned embeddings, rotary (RoPE), or ALiBi's distance penalties.

**Attention's strength is retrieval.** Nova: transformers reach 0.656–0.764 on associative retrieval
at 4× the trained length where LSTMs reach 0.283.

**Attention's weakness is state.** The same models reach 0.20–0.39 on modular sum. And there is a
structural reason: the softmax denominator sums over *all* positions, so adding non-matching keys
changes the read. Nova measured this drift directly — 0.724 for softmax against 0.236 for a
max-normalised read.

▶ **II.1** Why can a transformer with *learned absolute* position embeddings never extrapolate to
lengths beyond training, no matter how well it is trained?

## II.4 State-space models

`h_t = A h_{t-1} + B u_t` with `A` diagonal and stable. S4/S4D fix `A`; Mamba makes it
input-dependent ("selective"). These promise recurrence's length-invariance with attention's
parallelism.

Nova's measurement, at matched parameters: diagonal SSM 0.282 mean, selective SSM 0.281, against
LSTM's 0.574. On this task suite, at this scale, they underperformed the recurrent baselines they
were designed to replace. **That is a statement about 120k-parameter models on algorithmic tasks and
nothing more** — but it is what was measured.

## II.5 Adaptive computation

- **ACT** (Graves 2016): a halting unit, a ponder cost, output is a halting-weighted mixture.
- **PonderNet** (Banino 2021): per-step Bernoulli halting, expected loss, KL to a geometric prior.
- **Early exit** (BranchyNet, DeeBERT): exit when a classifier is confident.

Q-Neuro 3.0 is a fourth point: halt on a *supervised predicate* — a condition the task defines. It
wins where a halt target exists and loses to ACT where one must be invented (Book V).

## II.6 Mixture-of-experts and conditional computation

Route each token to a few of many experts: more parameters, similar FLOPs. Nova tested the same idea
at the level of hidden units and it was **beaten by a smaller fixed model** at equal cost — the
routing added nothing (Book VI).

---

# BOOK III — Q-NEURO HELIX

## III.1 The idea, taken seriously

A real network stores a number per unit. A complex one stores a number *and a phase*. Phase lets
contributions interfere — reinforce or cancel — which is how waves carry structured information
through noise. A complex recurrent operator `ψ ← U(x)ψ` with a phase-sensitive readout ought to be
more robust than a real one of the same size.

**It is a good idea.** It has a mechanism, it makes predictions, and it is not obviously wrong. That
is exactly what makes it dangerous.

## III.2 Why it looked convincing

`QN-000008`: the complex model beat every control across five unseen worlds and four severities,
with world-level confidence intervals excluding zero, and **+0.054 to +0.063** over a two-channel
real control. `QN-000016` isolated apparent mechanisms: ordered composition contributed **+0.232**,
phase-sensitive readout **+0.104**.

Any of those alone would justify a paper.

## III.3 The cracks, in order

| Claim | What killed it |
|---|---|
| Complex is more sample-efficient | A properly tuned GRU reached 0.920 at 250 cases against complex's 0.699 |
| Calibration transfers under shift | It transferred for nobody — temperature scaling worsened every model |
| Complex represents ambiguity better | Pair NLL **2.581** against 1.148 for real |
| Complex states encode hierarchy uniquely | GRU probes were stronger on every factor |

**Notice the pattern.** Each claim died against a *better control*, not against a better idea. The
original comparisons had been made against under-tuned baselines.

## III.4 The decisive control

Any complex linear map `M = A + iB` has an exact real block form:

```
ℛ(M) = [[A, −B], [B, A]]
```

So every complex network has a real counterpart computing the *identical* function. Build it and
compare.

> **Result: top-1 matched in all 1,920 held-out cells.** Not approximately. And across 2,880
> discovery cells, **0 positive effects** for complex against the best-real envelope; the
> hierarchical bootstrap interval for complex-minus-best-real was [−0.01325, −0.00457], entirely
> below zero.

## III.5 The lesson Helix paid for

> **A comparison is only as good as your ability to say precisely what is being compared.**
> "The real version of this model" is not a specification. It is four unspecified choices.

That sentence created Sentinel.

▶ **III.1** Write the realification of a complex `2×2` matrix by hand and verify that
`ℛ(M)ℛ(N) = ℛ(MN)`. Why does this make the complex-advantage claim untestable *as originally posed*?

---

# BOOK IV — Q-NEURO SENTINEL

## IV.1 Equivalence is not a property of two models

It is a property of a **map**, at a stated **level**, on a stated **domain**, with a stated
**transport class**. Four questions, and almost every informal equivalence claim answers none:

| Level | Meaning |
|---|---|
| E0 | symbolic identity of the function |
| E1 | bit-exact in finite precision |
| E2 | survives an adversarial audit on a declared domain |
| E3 | distributional agreement |
| E4 | aggregate metrics only |

Transport levels T0–T5 grade what the map carries: parameters, gradients, optimiser moments, the
learning-rate policy, weight decay.

**Certificates that refuse.** Declaring E0 or E1 *with* a domain restriction raises at construction.
Certificates downgrade, never upgrade. A map that cannot transport gradients says so rather than
approximating — the dense↔factorised map is the canonical case, because factor descent
preconditions the product and no transport exists.

## IV.2 Transport-degeneracy, and how Sentinel caught its own programme

A pair is **transport-degenerate** when the parameter map is the identity on shared coordinates.
Every transport level is then vacuously satisfied.

Helix's "exact real" control shared coordinates with the complex model. It was the same model wearing
two labels, and 1,478 of the 1,920 "wins" were **equivalence-induced zeros**. The negative result
survived — on the *other* 442 cells, where genuinely distinct real architectures won — but the
headline had to be rewritten.

▶ **IV.1** Why is a transport-degenerate pair still useful for testing *numerical* implementation,
and useless for testing anything about optimisation?

## IV.3 The scaling orbit, derived

Under a uniform scale `s`, gradients scale as `s^{-1}`. Optimiser state scales by the gradient's
power: first moments by 1, second by 2. The learning-rate exponent is **2 for SGD and 1 for Adam**,
because Adam's update is scale-free in the gradient.

With `η → η s²`, the SGD gradient step transports **exactly** — bitwise zero, derived first and then
confirmed.

**And weight decay breaks it structurally.** The gradient step needs `η s²`; decoupled decay needs
`η s⁰`. No single policy satisfies both. That is a true fact about the pair, not a bug.

## IV.4 The transport-covariance conjecture, and its death

**Conjecture:** an accumulated defect statistic measuring the failure of `T∘U = U∘T` predicts final
predictive divergence across equivalence families, with a single calibration.

**Result: false.** Leave-one-family-out held-out R² of **−31.7** for the best candidate — worse than
predicting the mean.

**The failure mode is calibration, not absence of signal.** Within a family the candidate is the
strongest feature available (R² 0.962 on factorisation). Across families the medians span **6.5
orders of magnitude**, because permutation is *conjugate* — its discrepancy sits at 1e-7 with nothing
left to predict — while the scaling orbit sits at 1e-0.6.

## IV.5 The dimension law, and why seven searches were doomed

```
d_free = max(0, P − g_arch − n(C−1))
```

`g_arch` is the architecture's exact symmetry group: softmax common mode contributes `h_last + 1`;
each positively homogeneous layer contributes `h` more, because `(W₁,W₂) → (cW₁, W₂/c)` is exact.

Seven consecutive attempts to find "free directions" that preserve training behaviour while improving
out-of-distribution behaviour all failed. They ran at `n = 600` with `P − g = 193`, so **`d_free = 0`
exactly**. There was nothing to find. Confirmed in 9 of 9 cells including the transition at `n = 193`.

> **Three reusable controls.** Check `d_free` before searching for free directions. Check
> integrability before trusting a first-order subspace. Check whether the simplest use of the same
> information already dominates — it did: joint training beat every navigation method on all four axes.

## IV.6 How to not fool yourself: the freeze protocol

1. Serialise the prediction; SHA-256 the serialisation.
2. The test **reads its thresholds out of the frozen record** and verifies the hash at load.
3. One attempt. Record the verdict either way.

**And a rule learned the hard way:** a frozen prediction whose hash cannot be re-verified *from disk*
is not frozen. Integer keys sort numerically in memory and lexicographically after a JSON reload;
one hash did not round-trip and was caught before any evidence existed.

▶ **IV.2** `DFREE-LAW-P1`'s substance held 126/126, and its instrument bug was diagnosed to the exact
count (3 configs × 7 n-values = 21 mismatches). It is still recorded as **compromised**. Argue both
sides, then say which rule you would adopt and what it costs you.

---

# BOOK V — Q-NEURO PULSE

## V.1 The promise of adaptive computation

A fixed-depth model pays the worst case on every input. If difficulty genuinely varies per example,
a model that decides its own depth should win. This is old — ACT is 2016 — and it keeps almost
working.

## V.2 The task, and two attempts to build it

`chase_to_goal`: a permutation defines a single cycle through 24 nodes; the model starts somewhere
and reports how far away node 0 is. Difficulty is a property of the datum and the answer is
discoverable only by following the chain.

**The first task was unsolvable and the models said so in unison.** All ten candidates returned
**exactly 0.1441** — the target was ambiguous given the input. Ten architectures agreeing to four
decimals is not a tie; it is a message about the task.

**And a bug produced a perfectly plausible null.** Everything sat at the 0.136 guessing baseline
because `Core.advance` used one embedding for both keys and values, making the associative lookup
impossible. Separate embeddings, and depth-8 reached 1.0000 by epoch 2. One step earlier we would
have concluded that recurrent chain-following does not work.

## V.3 The ladder

| Model | Halting rule | Accuracy | Steps |
|---|---|---|---:|
| Q0 | fixed depth 8 | 1.0000 | 8.00 |
| Q1 | PonderNet-style mixture | 0.6241 | 3.27 |
| Q2 | hard commit, straight-through | 0.9999 | 8.00 |
| Q3 | halt on detected arrival | 0.9994–1.0000 | **4.54** |

**Mixture halting collapses where commit halting does not** — changing only the halting rule lifts
the same core from 0.6241 to 0.9999. Single-variable intervention, seed fixed. And Q2 buys nothing:
0.9999 at full depth is Q0 with extra machinery.

Q3 halts on *detected arrival* and lets the halt step be the answer: **1.77× less inference compute
at matched accuracy with fewer parameters.**

## V.4 The number that was real and not reportable

Q3's headline reproduces exactly on seeds 0, 2 and 3. It also **fails on 4 of 10 seeds**, landing at
0.42–0.51. The outcome is bimodal with nothing in between. Q0 under identical conditions is 10 of 10.

> **The failure mode is silent and mimics success.** A collapsed run reports 5.2–6.1 average steps —
> a plausible, adaptive-looking allocation below the fixed depth of 8. All thirteen failed runs look
> like working elastic models if you read only the step counter.

If one sentence from Pulse should outlive it: **a compute-saving figure without a matched accuracy
figure and a seed-reliability rate cannot distinguish a working elastic model from a broken one.**

The cause was diagnosed by accuracy *conditioned on distance*: failing runs are perfect at distance
1–2 and collapse from 3. RMS-normalising the state took reliability from 11/24 to **20/20**.
Normalisation in recurrent networks is textbook; no novelty is claimed. But it is an *interaction* —
the same change destroys the fixed-depth model (1.0000 → 0.13–0.25), because an unnormalised residual
state carries magnitude information the distance readout uses.

## V.5 The lockstep ceiling, derived

The saving looked like it should be `max_depth / E[distance]`. Under batching it is not.

> A **lockstep** batch cannot exit until its slowest member does, so its cost tracks
> `E[max halt over the batch]`, not `E[halt]`.

For `P(k) ∝ 0.8^k` on 1..32: `E[max]` is 4.97 at batch 1, 12.53 at 8, 20.96 at 64, 29.42 at 1024.
The realisable saving decays from 6.43× to 1.09×. **This is arithmetic and it applies to ACT,
PonderNet, early-exit transformers and depth-routed mixtures alike.**

`QNEURO3-NICHE-P1` froze this on one task family and confirmed it, unprompted, on another — all four
clauses, including the clause that says where the result *stops working*. It is the one frozen
prediction in the whole programme that passed as written.

## V.6 Nominal versus realised compute

A 6.5× step-count saving measured **1.0×** in wall-clock. The batched forward runs every step and
*then* selects — the saving was nominal. The step counter is exactly the instrument §V.4 warned
could not be trusted alone, and it was trusted alone for one more round.

Active-set compaction — prior art, the standard early-exit loop — recovers it, **conditionally**:
1.95× at batch 256 on a core costing 2.66 µs/example-step, and 1.07× on one costing 0.33 µs. The
deciding quantity is step cost against gather cost.

## V.7 Why a working efficiency mechanism did not become an architecture

On real data (UCI HAR, the dataset's own subject-disjoint split) the mechanism came **fourth of
five**:

| arm | accuracy | chunks | train s |
|---|---:|---:|---:|
| fixed depth | 0.9127 | 16.00 | 3.6 |
| **ACT (2016)** | **0.9006** | **3.61** | 4.5 |
| confidence exit @ matched compute | 0.8747 | 2.28 | 3.4 |
| **supervised halting (ours)** | 0.8112 | 2.39 | **8.4** |

Real data supplies no ground-truth halt step, so ours had to be distilled from a teacher — which is
early-exit distillation, prior art, and why it costs 2.3× the training time.

> **Supervised halting earns its place only where the task supplies a halt target.** Where it does,
> it attains optimal allocation reliably. Where it does not, methods needing no teacher win.

▶ **V.1** Derive `E[max]` for a *uniform* halt distribution on 1..L and show the batched saving
tends to 1. Then explain why heavy-tailed difficulty makes the batch-1 saving larger *and* the
batched saving worse.

---

# BOOK VI — DISCOVERING NOVA

## VI.1 The design of a search that can fail fast

Pulse's failure mode was spending weeks on beautiful theories about one architecture. Nova built the
instrument first:

- **Tier 0** — trains, stays finite, beats the majority class. Seconds. Most candidates die here.
- **Tier 1** — all tasks, two seeds, evaluated at 1×, 2× and 4× the trained length.
- **Tier 2+** — by hand, and Sentinel is not spent below Tier 2.
- **A registry** — append-only JSONL, so a killed idea cannot be silently rediscovered.

## VI.2 Audit the instrument before trusting it

Eight tasks. Then three degenerate predictors were asked what they could score without computing
anything: position alone, token alone, previous token.

| task | chance | best shortcut | verdict |
|---|---:|---:|---|
| parity_scan | 0.501 | 0.529 | kept |
| mod_sum | 0.145 | 0.196 | kept |
| copy | 0.126 | 0.127 | kept |
| reverse | 0.126 | 0.127 | kept |
| needle | 0.131 | 0.131 | kept |
| **cummax** | 0.609 | **0.887** | **dropped** |
| **sort** | 0.126 | **0.598** | **dropped** |

The deliberately weak `causal_mlp` control had scored **0.917 on sort** at length 64. Without the
audit that would have been a headline.

## VI.3 The baselines, and the inconvenient result

Ten architectures at matched parameters produced a clean complementarity: recurrence tracks state and
extrapolates perfectly on it; attention retrieves and extrapolates on that; neither does both; and
nothing extrapolates on ordered memory.

The *established* answer to that gap is the linear-attention family — recurrent state that is also
content-addressable. It was implemented **as a baseline on purpose**, so Nova could not rediscover
it. It came out mediocre at both ends (0.55 / 0.30 / 0.58). The gap was real.

## VI.4 Hypothesis 1 — dilution

Softmax's denominator sums over all positions, so non-matching keys dilute the read. A
max-normalised read should be length-invariant.

**The operator-level property is real:** read drift when 24 distractors are inserted is 0.236 against
softmax's 0.724.

**The task-level effect is not.** The confound control — softmax with the same post-read
normalisation — captures the whole apparent gain (copy 0.172 → 0.305; max lands at 0.321).

Two bugs first. The original `max` normaliser divided by the sum at the end, which is *algebraically
softmax*; it matched the control to three decimals and the hypothesis had not been tested at all.

> **An operator can have a property without the property mattering.** Those are separate questions
> and only a control separates them.

## VI.5 Hypothesis 2 — interference

An LSTM alone reaches 0.992 on mod-sum. Add attention and it collapses, identically for three
normalisers. Which is it — attention *drowning out* a working recurrence, or *pre-empting* it?

A test-time branch ablation answers without retraining. Turning attention off makes it **worse**
(0.291 → 0.157); turning the recurrence off gives 0.161. Neither branch works alone. The model found
a joint solution that needs both and does not extrapolate.

The frozen prediction said a handicap would repair it. **All four clauses failed.** Dropout does not
de-conflict; it slides the model along a trade-off until it *is* an LSTM again.

**And a validity threat surfaced that touched everything.** The dramatic version — mod-sum 0.291 —
was an 800-step undertraining artifact; at 2400 steps it reads **0.776**. Every number was
re-measured. The prediction still fails as scored; the narrative was corrected.

## VI.6 Hypothesis 3 — composition

If the competition is an artifact of which *two* routes were paired, three should compose. Mean 0.692
against a required 0.75 — the best of anything tested — and reverse fell to **0.146**, chance.

The third clause *passed*: adding attention relieved the state-tracking conflict exactly as
predicted (mod-sum 0.776 → 0.998, needle 0.977). Ordered memory died in the same change.

> **Capability competition is conserved.** Relieving it between one pair reintroduces it elsewhere.

## VI.7 The winner was a 2014 paper

`cursor` — an LSTM emitting relative shifts `{−1,0,+1}` that move a read pointer over memory — is
Neural Turing Machine location addressing, §3.3.2 of Graves, Wayne & Danihelka (2014). Copy with
generalisation to longer sequences is that paper's *first experiment*. Nova reproduces it at 0.398,
more weakly, with a read-only memory that is strictly less capable than an NTM.

The prior-art firewall caught it before any novelty was claimed. That is why it runs before the
comparison.

---

# BOOK VII — THE NOVA ARCHITECTURE

## VII.1 What was actually built

There is no Nova architecture to specify, because none earned specification. What exists is a
frontier and the code that produced it. This book documents the leading configuration honestly, as
an engineering artifact rather than a contribution.

## VII.2 `cursor_attn`, the best-performing configuration

Three routes over one LSTM controller. For input tokens `x ∈ {0..V}^L`, embeddings `M = E[x]`:

```
c      = LSTM(M)                                     (B, L, d)   controller
s_t    = softmax(W_move c_t)                         (B, 3)      relative shift
g_t    = sigmoid(W_reset c_t)                        (B,)        reset-to-start gate
p_t    = (1 − g_t)(p_{t−1} + s_t[2] − s_t[0])        (B,)        cursor position
w_t    = softmax(−β |i − p_t|) ⊙ [i ≤ t]             (B, L)      causal soft window
r_t    = Σ_i w_t[i] M_i                              (B, d)      location read
a      = MaxAttention(LayerNorm(c))                  (B, L, d)   content read
y      = W_out · LayerNorm([c ; r ; a])              (B, L, V)   readout
```

Every component is prior art: the LSTM; NTM relative-shift addressing; attention-augmented
recurrence. **Nothing here is claimed as a mechanism.**

## VII.3 A worked forward pass, by hand

Take `copy` with payload `[5, 7]`, separator, and two blanks: `x = [5, 7, SEP, ▁, ▁]`, target
`y = [·, ·, ·, 5, 7]`.

The procedure the cursor *can* express: hold `p = 0` while reading the payload; at SEP, reset;
then advance `p` by 1 each step so that `r_t` reads `M_0 = 5` at `t = 3` and `M_1 = 7` at `t = 4`.

The shift distribution has to be near-deterministic for this to survive length — a soft `s_t` makes
`p` drift, and drift accumulates linearly in `L`. **That is the honest explanation for why the
measured copy accuracy is 0.398 and not 1.0**, and it is why the NTM's sharpening step exists.

▶ **VII.1** With `β = 4` and the cursor off by 0.5, compute the read weight on the intended position
versus its neighbour. Now do it for `β = 12`. Nova measured `cursor_sharp` at 0.270 against
`cursor` at 0.398 — sharper was *worse*. Propose two explanations and an experiment separating them.

## VII.4 Training

Identical for every architecture, which is the point: AdamW at 3e-3, one-cycle schedule, 2400 steps,
batch 64, gradient-norm clip 1.0, cross-entropy over scored positions only, training lengths sampled
uniformly from 8–16, ~120k parameters matched within 13%.

---

# BOOK VIII — WHY NOVA DID NOT WORK

Separating what is **causally demonstrated**, **strongly supported** and **hypothesised** is the
whole discipline. Here is Nova's split.

## VIII.1 Causally demonstrated

- **Branch ablation.** Removing either route from the two-route hybrid destroys performance
  (0.291 → 0.157 / 0.161). The routes are co-dependent; this is an intervention, not a correlation.
- **Normalisation is an interaction.** Adding RMS normalisation takes arrival halting from 11/24 to
  20/20 seeds and simultaneously takes the fixed-depth model from 1.0000 to 0.13–0.25. One change,
  opposite signs, same task.
- **The confound control.** Giving softmax the candidate's normalisation captures the entire
  apparent effect of length invariance.

## VIII.2 Strongly supported

- **Capability competition.** Consistent across four architectures and three seeds, with the
  direction predicted in advance and the magnitude measured. Not causally isolated from a capacity
  confound — three routes each get a third of the width.
- **Recurrence extrapolates state; attention extrapolates retrieval.** Ten baselines, one pattern.

## VIII.3 Hypothesised

- **Why** copy and reverse resist every architecture. Cursor drift is a plausible account for the
  cursor family and says nothing about why attention fails.
- **Whether** any of this survives at scale. Nothing above 131k parameters was tested.

▶ **VIII.1** For each of the three tiers above, name the single experiment that would move a claim
up a tier. Then estimate its cost. This is the exercise that decides what research is worth doing.

---

# BOOK IX — SCALING

## IX.1 What scaling laws say

Loss typically falls as a power law in parameters, data and compute: `L ≈ L∞ + (P₀/P)^α`. The
practical question for an architecture is never "is it better at 120k" but **does the gap grow, hold,
or vanish as `P` increases**.

## IX.2 What Nova can and cannot say

Nova tested **one scale**: ~120k parameters on a laptop CPU. It therefore says nothing about
scaling, and the report says so rather than extrapolating a trend from a single point.

**What a scaling study would need:** the same frontier at 0.25M, 1M, 4M and 16M, with the
advantage `Q_nova(P) − Q_baseline(P)` estimated at each and a prediction frozen for the largest scale
before it is run. That is a GPU-days experiment and it is not justified by a candidate whose leading
mechanism is a weaker reproduction of a 2014 result.

▶ **IX.1** Nova's best candidate beats the best baseline by 0.118 mean at 120k parameters. Sketch
the three scaling curves — growing, constant, and vanishing gap — and say which measurement at 1M
would distinguish them.

---

# BOOK X — ENGINEERING ON APPLE SILICON

## X.1 The machine

An 8 GiB M2 MacBook Air, fanless. 2.08 GiB available, 8 physical cores, 4 torch threads, MPS with
`complex64`, **measured** CPU↔MPS crossover at 65,536 elements, working budget 1.04 GiB at a
deliberately conservative half of available memory — sustained swapping on a fanless machine costs
more than a smaller model does.

## X.2 The constants that actually govern small models

| quantity | lookup core | streaming core |
|---|---:|---:|
| `c_step` (µs/example-step) | 2.66 | 0.33 |
| `c_launch` (µs/iteration) | 119.65 | 49.97 |
| `c_compact` (µs) | 87.33 | 31.56 |

**Launch cost dominates.** At batch 1, one iteration costs 119.65 µs of overhead against 2.66 µs of
work — a ratio of 45. Every asymptotic argument about these models is wrong at this scale.

## X.3 Execution policies

- **lockstep** — advance all rows until the last halts. Cost `n · max_i d_i`.
- **compaction** — drop halted rows and continue. Cost `Σ_i d_i`, the ideal.
- **bucketing** — group by *predicted* depth; needs a predictor, because halt depth is an output.
- **continuous** — finished rows leave, queued rows enter; needs a request stream.

All four must produce identical answers, and the check that enforces it caught a real bug: with
deferred compaction a fired row keeps advancing and overwrites its own answer — 215 rows wrong at
batch 256.

## X.4 The measured frontier

| batch | select | lockstep | compacted |
|---:|---:|---:|---:|
| 1 | 1504.4 | **417.5** | 469.2 |
| 32 | 180.9 | 162.4 | **124.9** |
| 256 | 72.1 | 71.9 | **38.7** |

Throughput at batch 256: 13,872/s → 25,841/s. Analytic peak activation memory 1,536 KiB → 457 KiB.
The planner picks the measured optimum at 8 of 9 batch sizes.

▶ **X.1** Using the constants in §X.2, derive the batch at which compaction should overtake lockstep.
The cost model that did this predicted 45; the measured crossover is below 16. Find the term the
model got wrong.

---

# BOOK XI — EXPERIMENTAL SCIENCE: HOW NOT TO FOOL YOURSELF

This is the most useful book here, because every entry is a mistake this programme actually made.

## XI.1 The defect catalogue

| Defect | What it claimed | Cause | What caught it |
|---|---|---|---|
| NaN misclassification | runaway runs **converged** | a norm overflows before its entries do, so `inf/inf = nan`, and `nan > t` is `False` | an exact-ρ probe disagreeing with the sweep |
| Sign error in a bound | ratios **below 1.0** | `S⁻¹` applied once too often | the invariant was written as a test |
| Threshold artifact | 6/16 systems bifurcate | crossings of an arbitrary line | a proper bimodality statistic |
| Pre-asymptotic fit | a clean `T^0.625` exponent | fitting inside a transient | `M/√T` was not constant |
| Holonomy metric | a geometric phase | convergence drift | a stay-control that beat every loop |
| Tautological feature | perfect within-family agreement | `e₀ = 0` makes two features identical | identical to 16 significant figures |
| Probe saturation | a law violated in 21 cells | a 400-sample probe cannot saturate rank | exact arithmetic: 3 configs × 7 values |
| Unsolvable task | ten architectures tie | the target was ambiguous | all ten returned exactly 0.1441 |
| Shared key/value | recurrent lookup fails | the associative read was impossible | accuracy pinned at the guessing baseline |
| Normaliser identity | a novel operator | dividing by the sum after the max **is** softmax | identical to the control to three decimals |
| Missing confound control | length invariance helps | normalisation applied only to candidates | the control captured the whole effect |
| Undertraining | catastrophic interference | 800 steps was not convergence | re-running at 2400 |
| Weak instrument | a discovery on `sort` | position alone scores 0.598 | the shortcut audit |
| Hash non-determinism | a frozen prediction | int keys sort differently after a JSON round-trip | re-verifying from disk |

## XI.2 The rules that fell out

**Freeze before you look**, and have the test read its thresholds from the frozen record so code
cannot drift from the prediction.

**A freeze broken by a corrected instrument stays broken.** `DFREE-LAW-P1`'s substance held 126/126
and its bug was diagnosed exactly. It is still recorded as compromised. This cost a defensible result.

**Run prior art as Gate 1**, before any performance comparison. The navigator's novelty died at
cosine 0.66 with OGD before a single benchmark ran, which saved benchmarking it.

**Attack the boring explanation first.** It was right about the Q3 headline, the bifurcation count,
the holonomy metric, and the exponent.

**Match everything — including seeds.** That last one is not standard and it is what killed Q3.

**A control that everything passes is not a control.** The first decoupled task was caught because a
model halting at the wrong step still scored 1.0000.

**Predict where your result stops working.** The one prediction that passed did so because its
ceiling clause was checkable and checked.

**Audit your instrument before your hypothesis.** Two of Nova's own tasks were disqualified.

▶ **XI.1** Pick three defects above. For each, write the *specific check* that would have caught it
one step earlier, and say what that check costs to run routinely.

---

# BOOK XII — SCIENTIFIC PREDICTION

## XII.1 From model to instrument

A model becomes a scientific instrument when its predictions are **prospective** (made before the
outcome is known), **calibrated** (confidence tracks correctness), and **robust to the shifts that
actually occur** in the domain.

Q-Neuro's HAR evaluation used the dataset's own **subject-disjoint** split — the model is tested on
people it never saw. That is the minimum realistic shift for a physiological model, and it is why
the split was not ours to choose.

## XII.2 Predictive is not causal

A model that predicts an outcome from a biomarker has not shown the biomarker causes it. Prediction
under intervention is a different and harder question. Nothing in this programme addresses it.

## XII.3 What went wrong when this programme tried

`QN-LAW-001` was fitted on discovery families, frozen, and opened on confirmation families: held-out
**R² = −30.94**. The sign transferred; the magnitude did not. No refit, no threshold change.

---

# BOOK XIII — BIOLOGY, THE PREREQUISITES

Written for orientation, not as a biology text.

**Sequence.** DNA → RNA → protein. A protein's function follows from its folded structure, which
follows from its amino-acid sequence. Sequence models are natural here because the data is literally
a string over a small alphabet with long-range dependencies — the same structure as this book's
algorithmic tasks, at vastly greater scale.

**Time.** Physiological signals are non-stationary and irregularly sampled, and the *timing* of a
measurement is often informative (a test ordered at 3am means something). Models that assume regular
sampling quietly discard that.

**Variation.** Between-subject variation usually exceeds within-subject variation, which is why
subject-disjoint evaluation is mandatory and random splits are misleading.

**The honest position.** This programme tested one real dataset of anonymised accelerometer
recordings. It has no biological result and makes no biological claim.

---

# BOOK XIV — PREDICTIVE MEDICINE

## XIV.1 The ladder, and why you cannot skip rungs

```
architecture advantage → scientific prediction advantage → biological robustness → medical usefulness
```

Q-Neuro reached the first rung and then lost it. It is nowhere near the second.

## XIV.2 What evaluation would have to mean

Discrimination (AUROC) is the least of it. A deployable model needs **calibration** (a stated 10%
risk means 10%), **temporal generalisation** (trained on last year, tested on this year),
**subgroup behaviour** (does it fail for a demographic), and **uncertainty** (does it know when it
does not know).

## XIV.3 Limitations, stated plainly

No patient data was used at any point in this programme. No clinical claim is made or supported.
A synthetic or activity-recognition result is not medical evidence, and treating it as such would be
the most serious error in this book — worse than any of the fourteen in Book XI, because those only
cost research time.

---

# BOOK XV — THE FAILURE ATLAS

The beautiful results, why each was convincing, and exactly what killed it.

**The 14-order-of-magnitude phase boundary.** Two models, provably the same predictor, on opposite
sides of a stability boundary because of coordinates. Prediction accuracy 0.9912, **zero** false
alarms in 1,476 cells, a confirmed differential prediction against Adam, a clean dimensionless
control parameter. *Killed:* against a nonlinear system it failed **96 of 96**. Initialisation
curvature does not govern a nonlinear trajectory — which was written into the frozen prediction's
anticipated failure modes beforehand.

**`rms(M) ~ T^0.625`.** A clean nonequilibrium exponent, visibly distinct from the equilibrium 0.5.
*Killed:* it was a transient. `M/√T` drifted 0.52 → 1.02 and the process was still non-stationary at
`t = 64000`.

**Six of sixteen systems bifurcate.** A discrete count, in a programme hunting for two distinguishable
fates. *Killed:* the count was of crossings of an arbitrary 0.9 line. A proper bimodality check gave
0.20–0.47, all unimodal.

**Curriculum holonomy.** Loops in curriculum space appeared to leave a geometric signature.
*Killed:* the stay-control, which traversed no loop at all, scored 4.19 — higher than every loop.

**A frozen prediction that passed.** `DISCOVERY-001-P1` returned `passes: True` on a hashed,
prospective, one-attempt test. *Killed:* its grid gave 197 chances to false-alarm and **zero** chances
to miss. The strongest safeguard in the programme produced a meaningless pass.

**Q3's 1.77× speedup.** Matched accuracy, fewer parameters, reproducing exactly on the seeds where
it works. *Killed as a claim:* it works 6 times in 10 against a baseline that works 10 times in 10,
and it fails silently.

**The complex hypothesis.** The idea the project is named after. *Killed twice:* realification proved
exact, removing expressivity as a route; and the last experiment of the programme closed the other —
a genuinely complex model at matched real parameters scored 1.0000 accuracy and 6.15 steps against
the real control's 1.0000 and 6.14. Identical, on every seed.

**Adaptive width's 2× Pareto win.** *Killed:* first by an accounting bug (fixed-depth arms charged
only to their selected step), then properly by a statically narrow model that matched it with 43%
fewer parameters.

**Length-invariant attention.** A real operator-level property, 3× less read drift than softmax.
*Killed:* the confound control captured the entire task-level effect.

**The best architecture Nova found.** *Killed as a contribution:* it is a 2014 paper, reproduced more
weakly than the original.

---

# BOOK XVI — OPEN PROBLEMS AND ANSWERS

## XVI.1 What this programme could not answer

1. **Why do copy and reverse resist every architecture at 4× length?** Best 0.470 and 0.371 against
   a chance level of 0.126. An NTM reportedly does better; a faithful reimplementation with a
   writable memory and sharpening is the obvious next experiment.
2. **Is capability competition fundamental or a capacity artifact?** Three routes at a fixed budget
   each get a third of the width. Repeating the composition experiment at matched *per-route* width
   would separate them.
3. **Does any of this survive at scale?** One scale was tested. Everything about scaling is open.
4. **A correct cost model for the compaction crossover.** The frozen one predicted batch 45; the
   truth is below 16, and it was not patched.
5. **Why do two task constructions with the same distance distribution give 6/10 versus 1/10
   training success?** Unexplained since Pulse.
6. **The sharp fixed-depth capability threshold between depth 4 and depth 8** on lookup tasks. The
   frozen carry-distance explanation is false and nothing replaced it.

## XVI.2 Selected answers

**I.1** Equivalent to a single `Linear(64, 64)` with matrix `W₂W₁`, of rank at most 64 — the 128-unit
bottleneck buys nothing without a nonlinearity.

**I.2** `a ∈ (0,1)` makes `a^t` decay, so the state cannot explode over 64 steps. With `a > 1`,
`a^64` overflows for any `a > 1.3` in float32. The first version of this layer took `log()` of a
negative linspace and produced NaNs — caught by a causality probe that reported "not causal" when
the real answer was "not finite".

**II.1** A learned position embedding table has a fixed number of rows. Positions beyond the trained
range have never received a gradient, so their embeddings are whatever initialisation left there.

**III.1** Because the "real comparator" and the complex model compute the identical function, so any
measured difference is numerical implementation. The claim as posed is untestable; the testable
version compares against *genuinely different* real architectures, which is what eventually decided it.

**VII.1** At `β = 4`, offset 0.5: weights ∝ `exp(−4·0.5) = 0.135` and `exp(−4·0.5) = 0.135` — a tie,
the worst case. At `β = 12`: still a tie at exactly 0.5 offset, but the *neighbourhood* is far
sharper, so small drift is punished harder. Two explanations for sharper being worse: sharper windows
give near-zero gradient away from the cursor, so position errors cannot be corrected; or sharpness
amplifies drift into hard misses. Separate them by measuring cursor position error against target
position across training.

**X.1** `n* = c_compact · E[max] / (c_step · (E[max] − E[d]))` gives ≈45. The term it got wrong is the
compaction count: the model assumed one compaction per halting iteration, `min(n−1, E[max]−1) = 15`
at batch 16, against **10** measured; and `c_compact` was measured on a synthetic gather of all six
state tensors, over-stating the marginal cost inside the loop.
