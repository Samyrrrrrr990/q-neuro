# Claim Ledger

| Claim | Evidence | Counterevidence / confounders | Confidence | Status |
|---|---|---|---|---|
| Ordered computation is required for the chronology-twin task | QN-000003: ordered models 1.0 pair accuracy; MLP 0.0; twin vectors are identical by construction | This is true by task construction and does not establish a broad medical phenomenon | High | Replicated in simulator |
| Low-rank operator states are more sample-efficient than the tested tiny Transformer | QN-000004: real/complex exceed Transformer at every size through 5,000 cases | Transformer received limited tuning; no GRU or state-space baseline; one generator | Medium | Preliminary |
| The complex model uses relative phase | QN-000003: zero/random phase top-1 near 0.21 versus 1.0 learned | Post-training ablation disrupts the representation; a trained phase-free equivalent is stronger evidence | High for dependence, low for benefit | Preliminary |
| Complex operators improve top-1 after 500 cases | QN-000004: complex exceeds real from 500–5,000 cases across three seeds | Real wins at 250; complex NLL/ECE/runtime are worse; task saturates; no two-channel real control | Low–medium | Preliminary |
| Complex arithmetic is superior overall | No supporting Pareto result | Real is better calibrated, faster, and stronger at 250 cases | Low | Not supported |
| Q-Neuro is novel | No systematic novelty review completed | Many components have direct prior art; combinations may also exist | Very low | Unresolved |
| Q-Neuro has clinical diagnostic value | None; only synthetic archetypes tested | No real data, external validation, prospective study, or medical-device evaluation | None | Unsupported |

