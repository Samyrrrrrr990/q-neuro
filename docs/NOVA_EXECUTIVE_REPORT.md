# Q-Neuro Nova — Final Executive Report

**Verdict: NO — NO NEW SUPERIOR ARCHITECTURE SURVIVED.**

---

## The forty-six questions

**1. What is Q-Neuro Nova?** A clean-slate search for a new principle of neural computation: 32
architectures, six mechanism families, eight algorithmic tasks with known optimal procedures, judged
by whether the learned procedure survives at four times the trained length. It found nothing new.

**2. To a 12-year-old.** We tried about thirty ways of building a thinking machine and tested each on
puzzles where we know the right method. We checked whether they still worked on puzzles four times
longer than the ones they practised on. The best one turned out to be an idea somebody else published
in 2014, and ours was a worse copy of it. So the honest answer is: we did not find anything new.

**3. To an ML PhD.** A length-extrapolation capability matrix over ten baselines (transformers with
three position schemes, gated RNNs, S4D, selective SSM, linear attention, retention, causal MLP) and
~22 candidates, at matched parameters (~120k, ±13%), matched optimiser and matched steps, three
seeds. Five tasks retained after a shortcut audit disqualified two. Three hypotheses frozen and
hashed; all three falsified, one by its own confound control. Leading candidate is a read-only
NTM-style relative-shift cursor over an LSTM, which reproduces Graves et al. 2014 more weakly.

**4. What principle did we discover?** None. The surviving statement is negative: **capability
competition is conserved** — relieving a conflict between two capabilities by adding a third
computational route reintroduces one elsewhere.

**5. Is that principle genuinely new?** The concept is not (gradient starvation; multi-task
interference). The specific measurement — that a hybrid can be strictly worse at length extrapolation
than its own recurrent component, and that a third route relocates rather than resolves the conflict
— is not obviously stated elsewhere in this form. It is a diagnostic, not a mechanism.

**6. Closest prior art.** NTM location-based addressing (Graves, Wayne & Danihelka 2014 §3.3.2) for
the cursor; attention-augmented RNNs (Bahdanau 2014) for the leading hybrid; gradient starvation
(Pezeshki 2021) for the competition account. Full matrix in `docs/NOVA_PRIOR_ART.md`.

**7. What does Nova beat?** At matched parameters and steps on this suite: every transformer variant
(0.414–0.438 mean), both state-space families (0.281–0.282), the causal MLP (0.309), and the LSTM
(0.574) — with a best of 0.692. All of that gain comes from adding two known mechanisms to an LSTM.

**8. What beats Nova?** Nothing on this suite, and that is not the point. On real data (UCI HAR) the
Q-Neuro halting mechanism is beaten by ACT (2016) and by a softmax threshold. And the cursor is
beaten by the NTM it copies.

**9. Matched parameters?** 0.692 against the best baseline's 0.574. **10. Matched compute?** Same
optimiser, schedule and 2400 steps for every arm. **11. Matched latency?** Not measured for Nova; the
cursor is sequential and slower per step than a transformer. **12. Matched training budget?**
Identical; the budget was itself audited after 800 steps proved to be undertrained.

**13. Does it generalise?** Across five tasks, no single architecture is near the per-task best on
all of them. **14. Does it extrapolate?** On state tracking, perfectly (1.000 at 4×). On ordered
memory, no architecture exceeds 0.470 against chance 0.126. **15. Does it scale?** Unknown — one
scale was tested. **16. Does the advantage grow with scale?** Unmeasured, and not claimed.

**17. Strongest result.** An LSTM reaching 1.000 on parity and 0.992 on modular sum at four times the
trained length, while no attention-based model exceeds 0.389 on modular sum. That is a baseline
result, not ours.

**18. Strongest negative result.** Composing three routes moved the conflict instead of resolving it:
mod-sum 0.776 → 0.998 and reverse 0.348 → 0.146 in the same change.

**19. Strongest frozen prediction.** `QNEURO3-NICHE-P1` from the previous era — the only pass in
nineteen — because its fourth clause predicted *where the result stops working* and was confirmed on
a family it was not derived from.

**20. Most surprising failed prediction.** H-DILUTION. The operator-level property was real and
measurable (3× less read drift), and it made no difference at all once the confound control existed.

**21. Five Sentinel attacks survived.** Causality checks on every model; finiteness checks that caught
a NaN masquerading as non-causality; parameter matching within 13% across families; a shortcut audit
that removed two tasks; and a training-budget audit that re-measured every headline number.

**22. The attack that still worries me most.** Capacity. Three routes at a fixed 120k budget each
receive roughly a third of the width, so part of the "competition" could be capacity starvation. The
control for that is approximate and I did not resolve it.

**23. Established.** The capability matrix; the shortcut ceilings; the operator-level invariance
measurement; the branch-ablation co-dependence result. **24. Conjectural.** That capability
competition is fundamental rather than a capacity artifact; why ordered memory resists everything.

**25. Architecture novel?** No. **26. Training method novel?** No. **27. Empirical phenomenon novel?**
Partly — the conserved-competition measurement. **28. Scaling behaviour novel?** Unmeasured.

**29. Is GPU scaling justified?** **No.** A scaling study is justified by a candidate with an
unexplained advantage. Nova's leading candidate is a weaker reproduction of a 2014 result, and asking
for compute to scale it would be asking to scale someone else's paper.

**30. What should the next large-compute experiment test?** Not Nova. The honest next experiment is a
faithful NTM/DNC reimplementation — writable memory, sharpening, interpolation gate — against modern
baselines on ordered-memory extrapolation, because that is the one gap nothing here closed and there
is a published method that reportedly does better.

**31. Biological prediction?** No. **32. Medical research?** No. Neither was earned, and the ladder
was not skipped.

**33. What could Nova realistically change?** How people report adaptive-compute and architecture
results: the shortcut audit, the confound control, the budget audit and the batch-maximum ceiling are
each cheap and each caught a false positive here.

**34. Publication readiness /10 — 7.** The negative results are clean and reproducible; there is no
positive claim to publish. **35. Novelty /10 — 2.** **36. Capability /10 — 3.** **37. Efficiency /10
— 5.** **38. Reproducibility /10 — 9.** **39. Theoretical importance /10 — 3.** **40. Engineering
value /10 — 5.** **41. Scientific significance /10 — 3.**

**42. Probability current conclusions survive independent replication — high, ~0.85** for the
negative verdict and the capability matrix; **~0.5** for the conserved-competition claim, which rests
on three seeds and an imperfect capacity control.

**43. What should the paper claim?** That a systematic, budget-matched, shortcut-audited search over
32 architectures found no mechanism beating prior-art compositions on length extrapolation; that
capability competition appears conserved; and that ordered-memory extrapolation is open.

**44. What must the paper never claim?** A new computational principle; a new architecture; that the
cursor is novel; anything about scale; anything biological or medical.

**45. Is Nova complete?** Yes, on its boundary condition. **46. Is further architecture search
justified?** Not locally. The last three rounds all produced prior-art compositions and further
variants would be tiny mutations. A faithful NTM reimplementation is a different question and is
justified — as a reproduction, not a discovery.

---

## THE VERDICT

> **Did Q-Neuro Nova discover a genuinely better way for neural networks to compute?**
>
> ## NO — NO NEW SUPERIOR ARCHITECTURE SURVIVED.
