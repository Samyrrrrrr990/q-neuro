# Nova prior-art firewall

Run against every mechanism that showed a signal, by computational structure rather than by name.
The question is never "did anyone use this word" but "does anyone's update equation do this".

---

## 1. The cursor result is a reproduction of the Neural Turing Machine

The only Nova candidate to lead the frontier is `CursorMemory`: an LSTM controller emits a
distribution over relative shifts `{−1, 0, +1}`, that shift moves a read pointer over a memory of
past inputs, and the read is a soft window centred on the pointer.

**This is Neural Turing Machine location-based addressing** (Graves, Wayne & Danihelka, 2014), §3.3.2
"Interpolation and convolutional shift": a controller emits a shift kernel over relative positions
which is convolved with the previous read weighting. The Differentiable Neural Computer (Graves et
al., 2016) extends the same idea with temporal-link matrices.

| Property | Nova `cursor` | NTM (2014) |
|---|---|---|
| Controller | LSTM | LSTM or feedforward |
| Addressing | relative shift over `{−1,0,+1}` | convolutional shift over a relative kernel |
| Read | soft window centred on the pointer | weighted sum over the same soft weighting |
| Reset-to-start gate | yes, one learned gate | interpolation gate against the previous weighting |
| Memory | the input embeddings themselves | a learned, writable memory matrix |
| Writes | none | yes — erase and add |

The only substantive difference is that Nova's memory is read-only and is just the input sequence,
which makes it *strictly weaker* than an NTM, not different in kind.

**And the headline result is theirs too.** The NTM paper's first experiment is copy, and its
central claim is generalisation to sequences longer than those seen in training. Nova measures copy
accuracy of 0.403 at four times the trained length against a chance level of 0.126 — a real effect,
and a weaker version of a result published in 2014.

**Conclusion: no novelty. The cursor is a reproduction, and a partial one.**

---

## 2. Everything else that was tested

| Nova mechanism | Closest prior art | Verdict |
|---|---|---|
| `max` / `threshold` attention normalisers | Hard and sparse attention; sparsemax (Martins & Astudillo 2016); top-k attention | Prior art, and killed by its own confound control anyway |
| `logl` temperature scaling | Entropy-invariant / length-scaled attention temperature, discussed in long-context work | Prior art |
| Post-read RMS normalisation | Standard normalisation placement | Prior art; the only surviving effect from that family |
| `rnn_attn_*` | Attention-augmented RNNs; every pre-transformer encoder–decoder (Bahdanau 2014) | Prior art |
| Attention-branch dropout | Stochastic depth (Huang 2016), DropPath, shake-shake | Prior art |
| `late_fusion` disjoint routes | Two-tower / product-of-experts / mixture ensembles | Prior art |
| `looped*` weight-shared depth | Universal Transformer (Dehghani 2018); looped transformers | Prior art |
| `linear_attention`, `retentive` | Fast-weight programmers (Schmidhuber 1992); linear transformers (Katharopoulos 2020); RetNet | Prior art — implemented deliberately as **baselines** so Nova could not rediscover them |
| `ssm_selective` | Mamba-family input-dependent gating | Prior art — same reason |
| Capability-competition account | Gradient starvation (Pezeshki 2021); shortcut learning (Geirhos 2020) | The concept is prior art; the specific measurement is not |

---

## 3. What is *not* obviously in the literature

Two measurements, both negative or diagnostic, neither a mechanism:

**3.1 A hybrid can be strictly worse than its own component.** An LSTM alone reaches 0.992 on
`mod_sum` at four times the trained length. Adding a parallel attention branch drops it to 0.291 —
identically for three different attention normalisers. Test-time ablation shows the recurrence never
learned the automaton at all: removing attention gives 0.157, removing the recurrence gives 0.161,
and neither branch works alone. The model found a joint solution that needs both routes and does not
extrapolate.

Gradient starvation predicts that an easily-fit feature suppresses others. What is measured here is
the architectural consequence: **adding a second computational route can remove a capability the
first route had by itself.** Branch dropout does not repair it — it slides the model along a
trade-off until it simply *is* the LSTM again (needle 0.841 → 0.260 against LSTM-alone 0.283).

**3.2 An operator-level length-invariance probe.** Measuring read drift when non-matching keys are
inserted separates "the operator has the invariance" (max/threshold 0.236 vs softmax 0.724, a real
3× difference) from "the invariance helps" (it does not — the confound control captures the whole
task-level effect). The two are different questions and the probe makes that separable.

Neither is an architecture. Both are diagnostics.

---

## 4. Position, stated so it cannot drift

**Nova discovered no new computational mechanism.** Its best-performing candidate reproduces a 2014
result more weakly than the original. Its two frozen hypotheses were both falsified. What it
produced is a capability matrix over ten baselines and roughly twenty candidates, a shortcut audit
that disqualified two of its own tasks, and one negative finding about architecture composition
that is sharper than the surrounding literature makes explicit.
