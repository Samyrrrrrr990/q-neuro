# Prior Art and Novelty Ledger

Last searched: 2026-08-13. This is a structured, primary-source-led review of the mechanisms
implemented in Q-Neuro; it is not a systematic review. Searches covered official proceedings,
publisher pages, and archival records for complex and unitary recurrence, quantum-inspired
language/cognition, density-state evolution, dynamical and energy-based networks, adaptive
computation, active feature acquisition, local learning, multi-task optimization, uncertainty,
state-space models, and medical differential diagnosis. A missing paper remains possible.

Novelty labels: **KNOWN**, **KNOWN VARIANT**, **NEW COMBINATION**, **POTENTIALLY NOVEL**, and
**UNRESOLVED**. “New combination” is descriptive and is not a priority or first-in-literature
claim.

## Mechanism matrix

| Q-Neuro idea | Nearest primary work | Similarity | Difference actually tested here | Status | Confidence |
|---|---|---|---|---|---|
| Complex-valued trainable state | [Trabelsi et al., Deep Complex Networks, ICLR 2018](https://openreview.net/forum?id=H1T2hmZAb) | Complex weights, activations, normalization, and real objectives | Disease-indexed sequential state and magnitude-squared measurement | KNOWN VARIANT | High |
| Norm-preserving complex recurrence | [Arjovsky et al., Unitary Evolution RNNs, ICML 2016](https://proceedings.mlr.press/v48/arjovsky16.html); [Jing et al., EUNN, ICML 2017](https://proceedings.mlr.press/v70/jing17a.html) | Complex hidden states and structured unitary/orthogonal evolution | Evidence-specific low-rank updates instead of one shared recurrent unitary map | NEW COMBINATION | Medium |
| Real orthogonal control | [Helfrich et al., scoRNN, ICML 2018](https://proceedings.mlr.press/v80/helfrich18a.html) | Stable real recurrent evolution without complex arithmetic | Q-Neuro's real operator is low-rank and evidence-conditioned, not Cayley-orthogonal | KNOWN VARIANT | High |
| Ordered non-commuting measurements | [Wang et al., PNAS 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4084470/) | Non-commuting projections explain measured question-order effects | Supervised evidence transforms on a synthetic causal chronology task | NEW COMBINATION | Medium |
| Quantum-probability causal order | [Trueblood & Busemeyer, Cognitive Science 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3350941/) | Incompatible causes produce order effects in causal judgment | Trainable classifier state rather than a cognitive probability model | KNOWN VARIANT | Medium |
| Complex tensor coupling | [Zhang et al., quantum many-body-inspired language model, 2018](https://arxiv.org/abs/1808.09891) | Complex amplitudes and tensor interactions for semantic composition | Factor-coupled diagnostic evidence tested against separable controls | KNOWN VARIANT | Medium |
| Density matrices for unresolved alternatives | [Meyer & Lewis, CoNLL 2020](https://aclanthology.org/2020.conll-1.21/) | Density matrices encode distributions over word senses and compose meanings | Diagnosis-indexed low-rank PSD state with explicit trace/PSD tests | KNOWN VARIANT | High |
| Lindblad-inspired density evolution and interference measurement | [Yan et al., LI-QiLM, NAACL 2024](https://aclanthology.org/2024.naacl-long.116/) | Learned Lindblad master-equation evolution plus interference measurement | Synthetic diagnosis, evidence tokens, low-rank factors, and causal controls | KNOWN VARIANT | High |
| Hidden quantum Markov sequence state | [Srinivasan et al., AISTATS 2018](https://proceedings.mlr.press/v84/srinivasan18a.html) | Quantum-probabilistic state evolves over observations | Discriminatively trained hypothesis coordinates and neural evidence operators | KNOWN VARIANT | Medium |
| Hamiltonian-inspired learned dynamics | [Greydanus et al., HNN, NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/26cd8ecadce0d4efd6cc8a8725cbd1f8-Abstract.html) | Learned rotational/conservation-inspired flow | Inference dynamics for competing labels, with damping and real controls | NEW COMBINATION | Medium |
| Continuous-depth state evolution | [Chen et al., Neural ODEs, NeurIPS 2018](https://papers.nips.cc/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html) | Hidden state defined by a differential equation; adaptive solver computation | Fixed discretizations and diagnostic state rather than a black-box solver | KNOWN VARIANT | High |
| Fixed-point diagnostic state | [Bai et al., Deep Equilibrium Models, NeurIPS 2019](https://papers.nips.cc/paper/2019/hash/01386bd6d8e091c2ab4c7c7de644d37b-Abstract.html) | Prediction through convergence of a tied transformation | Explicit hypothesis geometry and evidence-conditioned updates | NEW COMBINATION | Medium |
| Modern Hopfield/metastable states | [Ramsauer et al., ICLR 2021](https://openreview.net/forum?id=tL89RnzIiCd) | Continuous energy, fixed points, metastable subsets, association | Disease hypotheses and controlled ambiguity/order evaluation | KNOWN VARIANT | High |
| State-space sequence control | [Gu et al., S4, ICLR 2022](https://openreview.net/forum?id=uYLFoz1vlAC) | Structured recurrent state dynamics for long sequences | Tiny diagonal state-space baseline on short clinical-like evidence | KNOWN | High |
| Dynamic routing | [Sabour et al., Capsules, NeurIPS 2017](https://papers.nips.cc/paper/2017/hash/2cad8fa47bbef282badbb8de5374b894-Abstract.html) | State-dependent routing through agreement | Amplitude-sparse hypothesis routing was proposed but not established here | UNRESOLVED | Medium |
| Variable internal computation | [Graves, Adaptive Computation Time, 2016](https://arxiv.org/abs/1603.08983) | Differentiable per-input recurrent steps | Halting from state velocity and measured active-index execution | KNOWN VARIANT | High |
| Early-exit inference | [Bolukbasi et al., ICML 2017](https://proceedings.mlr.press/v70/bolukbasi17a.html); [Jazbec et al., UAI 2024](https://proceedings.mlr.press/v244/jazbec24a.html) | Case-dependent termination and uncertainty-aware exits | Validation-selected velocity threshold; result degenerates to fixed depth | KNOWN VARIANT | High |
| Active evidence acquisition | [Li & Oliva, ICML 2021](https://proceedings.mlr.press/v139/li21p.html); [Valancius et al., ICML 2024](https://proceedings.mlr.press/v235/valancius24a.html) | Sequential feature queries chosen to reduce uncertainty/cost | Synthetic findings, equal query cost, predictor-conditioned entropy policy | KNOWN VARIANT | High |
| Gradient-conflict projection | [Yu et al., PCGrad, NeurIPS 2020](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html) | Detect and modify conflicting task gradients | PCGrad is a direct control for phase-coded rotation | KNOWN | High |
| Phase-coded gradient rotation | PCGrad and complex-valued optimization are the closest located ingredients | Magnitude/direction of task updates are altered under conflict | Rotation in explicit real/imaginary parameter pairs; no gain in QN-000021 | POTENTIALLY NOVEL | Low |
| Local error/plasticity learning | [Lillicrap et al., Nature Communications 2016](https://www.nature.com/articles/ncomms13276); [Scellier & Bengio, Frontiers 2017](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2017.00024/full) | Alternatives to exact global reverse-mode weight transport | Transition-local complex prototype rule, evaluated against backprop | KNOWN VARIANT | High |
| Zero-backprop random dynamics plus readout | Reservoir/random-feature traditions and feedback alignment are nearby | Frozen internal dynamics with a learned output mechanism | Class-centroid complex readout; strongly refuted in QN-000021 | KNOWN VARIANT | Medium |
| Source temperature scaling | [Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) | Single validation-fitted scalar corrects confidence | Explicit test of calibration transport under causal shift | KNOWN METHOD | High |
| Maximum-softmax OOD score | [Hendrycks & Gimpel, ICLR 2017](https://openreview.net/forum?id=Hkg4TI9xl) | Low maximum softmax probability as OOD signal | Completely omitted synthetic disease and paired representation scores | KNOWN METHOD | High |
| Differential diagnosis from temporal clinical data | [Thiagarajan et al., DDxNet, Scientific Reports 2020](https://www.nature.com/articles/s41598-020-73126-9) | Deep sequence models applied to EEG/ECG/EHR diagnostic targets | NeuroWorld is synthetic and mechanism-focused, not a clinical EHR model | KNOWN DOMAIN | High |
| Large-scale learned diagnostic association | [Liang et al., Nature Medicine 2019](https://www.nature.com/articles/s41591-018-0335-9) | Machine learning over clinical records for differential evaluation | Q-Neuro has no patient data, clinician comparison, or clinical validation | KNOWN DOMAIN | High |
| Tensor coupling in medical ML | [Selvan & Dam, MIDL 2020](https://proceedings.mlr.press/v121/selvan20a.html) | Tensor-network factorization used in medical classification | Structured finding-factor interaction instead of medical imaging | KNOWN VARIANT | Medium |

## Findings that changed the project’s claims

1. Complex networks, unitary recurrence, amplitude measurements, and non-commutative cognitive
   order models are established. Q-Neuro cannot claim to introduce complex hypothesis states or
   order-sensitive operators as concepts.
2. Density matrices for ambiguity are established, and LI-QiLM already combines Lindblad-inspired
   evolution with interference measurement. Diagnostic Density Dynamics (D3) is therefore a known
   domain/task variant, not a new computational law.
3. Adaptive computation, equilibrium states, Hamiltonian networks, modern Hopfield dynamics,
   active feature acquisition, local learning, and gradient surgery all have direct primary prior
   work. The Q-Neuro implementations are controlled combinations and negative/positive empirical
   tests.
4. Phase Gradient Optimization is the only located mechanism without a close exact match, but the
   search is non-systematic and QN-000021 shows no advantage over ordinary multi-objective AdamW.
   There is no novelty or performance claim to make.
5. The defensible contribution is the **experimental framework and evidence package**: a causal
   synthetic task family in which chronology twins, unseen-world shifts, ambiguity, active
   acquisition, hidden syndromes, hard compute, learning laws, trajectories, and mechanistic
   ablations are evaluated under one registry. Whether that combination is publication-novel
   remains unresolved until a formal systematic review and independent peer review.

## Review limitations

- Searches were performed by one investigator and one coding agent, without dual screening or a
  preregistered database protocol.
- The search emphasized English-language primary sources accessible through official archives.
- Patent literature, theses, non-English work, unpublished industrial systems, and every medical
  expert-system lineage were not exhaustively screened.
- “No close exact match located” never means “first.” All manuscript wording avoids priority claims.
