# Prior art and novelty audit v2

Last searched: 2026-08-14  
Scope: primary-source-led, non-systematic review  
Conclusion: no claim that Q-Neuro invented complex neural computation, noncommutative sequence
updates, magnitude-squared measurement, or structured recurrent state

## Central novelty boundary

The exact-real falsifier is not a new mathematical identity. A complex matrix
`W = A + iB` acting on `h = x + iy` can be written as the structured real block
`[[A,-B],[B,A]] [x,y]`. Arjovsky, Shah, and Bengio explicitly used this conversion to implement
unitary recurrence on real-valued frameworks in 2016. Q-Neuro's contribution is the use of that
identity as a mandatory, parameter-matched adversarial control and the prospective preservation of
the resulting negative evidence—not the identity itself.

The identity also does not imply that all complex and real networks train identically. Complex
multiplication constrains the real block; parameterization, nonlinearities, and optimization can
change inductive bias. Theory and experiments have reported both potential complex learnability
advantages and slower optimization. The defensible Q-Neuro conclusion is therefore narrow: for
the implemented operator, complex arithmetic supplies no unique function unavailable to its exact
real block, and the tested best-real envelope removes the observed performance advantage.

## Primary-source matrix

| Topic | Primary source | Relevance to Q-Neuro | Novelty implication |
|---|---|---|---|
| Deep complex networks | [Trabelsi et al., ICLR 2018](https://openreview.net/forum?id=H1T2hmZAb) | Established trainable complex layers, initialization, normalization, and complex convolution/recurrent components | Complex neural computation is established |
| Complex unitary recurrence and real-block implementation | [Arjovsky et al., ICML 2016](https://proceedings.mlr.press/v48/arjovsky16.html) | Uses complex hidden states and gives the explicit two-channel real matrix representation | Exact-real mapping is known; using it as a falsifier is methodological |
| Efficient unitary recurrence | [Jing et al., ICML 2017](https://proceedings.mlr.press/v70/jing17a.html) | Structured unitary transitions for long-term dependencies | Complex recurrent structure predates Q-Neuro |
| Real orthogonal recurrence | [Helfrich et al., ICML 2018](https://proceedings.mlr.press/v80/helfrich18a.html) | A scaled Cayley transform provides norm-preserving real recurrence and can outperform unitary controls with fewer parameters | Strong real structured controls are mandatory |
| Complex training dynamics | [Tan et al., NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/dc06d4d2792265fb5454a6092bfd5c6a-Abstract-Conference.html) | Shows that commonly used real-valued backpropagation can reduce infinite-width complex training dynamics to ordinary real dynamics for many activations | Architecture and optimizer must be separated from arithmetic claims |
| Complex learnability boundary | [Wu et al., NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4ac4365b98bc242acd5ab974a05c68a8-Abstract-Conference.html) | Proves settings where a complex neuron has representational/learnability differences and can also learn more slowly | Q-Neuro's negative result is architecture-specific, not a universal theorem |
| Complex capacity under real identification | [Altunören, TMLR 2026](https://openreview.net/forum?id=jfeJnfST36) | Studies modReLU networks through the identification `C^d ≅ R^(2d)` and bounds their VC dimension | Current theory treats real identification as fundamental while leaving capacity gaps open |
| Hidden quantum Markov models | [Srinivasan et al., AISTATS 2018](https://proceedings.mlr.press/v84/srinivasan18a.html) | Learns quantum-probabilistic sequential latent states from synthetic data | Quantum-inspired sequential state models predate Q-Neuro |
| Quantum-inspired density evolution in NLP | [Yan et al., NAACL 2024](https://aclanthology.org/2024.naacl-long.116/) | Combines Lindblad-inspired evolution with interference measurement for sentiment analysis | Density evolution and interference measurement are known combinations |
| Noncommutative cognitive order effects | [Wang et al., PNAS 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4084470/) | Uses noncommuting projections to predict survey question-order effects | Noncommutativity as an order-effect mechanism is established outside physics |
| Quantum causal reasoning | [Trueblood & Busemeyer, Cognitive Science 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3350941/) | Models causal judgment order effects with quantum probability | Quantum-like causal-order modeling predates the project |
| Structured state-space sequence models | [Gu et al., ICLR 2022](https://openreview.net/forum?id=uYLFoz1vlAC) | Efficient structured recurrent/convolutional state dynamics for long sequences | State-space baselines are part of the modern comparator set |

## What may be publishable

The potentially publishable object is the **falsification workflow and negative result**:

1. a synthetic chronology benchmark initially yields a positive complex-versus-two-channel result;
2. a mapped exact-real block shows that the computation is not uniquely complex;
3. stronger real sequential controls reverse the best-envelope comparison;
4. a prespecified candidate law fits discovery data but fails held-out magnitude prediction;
5. a named grand test is blocked rather than weakened when readiness gates fail.

This sequence is useful because architecture papers often compare a structured candidate with an
insufficiently equivalent control. Whether this complete evidence package is publication-novel
requires formal peer review. No priority language is warranted.

## Search limitations

- One investigator and one coding agent performed the search; there was no dual screening.
- The review is not registered, exhaustive, or suitable for a systematic-review claim.
- Patent, thesis, non-English, and unpublished industrial literature were not exhaustively covered.
- Search results through 2026 can change; all novelty labels remain provisional.
