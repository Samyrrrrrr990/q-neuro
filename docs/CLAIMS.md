# Claim Ledger

| Claim | Evidence | Counterevidence / confounders | Confidence | Status |
|---|---|---|---|---|
| Ordered computation is required for the chronology-twin task | QN-000003: ordered models 1.0 pair accuracy; MLP 0.0; twin vectors are identical by construction | This is true by task construction and does not establish a broad medical phenomenon | High | Replicated in simulator |
| Low-rank operator states are more sample-efficient than the tested tiny Transformer | QN-000004 and QN-000006: real/complex exceed Transformer through 1,000 cases | QN-000006 tuned GRU strongly exceeds operators at 250 cases | Medium for the narrow Transformer comparison | Preliminary |
| Operators are the most sample-efficient in-domain mechanism | QN-000004 initially suggested this | QN-000006 tuned GRU reaches 0.920 at 250 cases versus 0.774 real and 0.699 complex | High | Refuted under tested setup |
| The complex model uses relative phase | QN-000003: zero/random phase top-1 near 0.21 versus 1.0 learned | Post-training ablation disrupts the representation; a trained phase-free equivalent is stronger evidence | High for dependence, low for benefit | Preliminary |
| Complex operators improve top-1 after 500 cases | QN-000004: complex exceeds real from 500–5,000 cases across three seeds | Real wins at 250; complex NLL/ECE/runtime are worse; task saturates; no two-channel real control | Low–medium | Preliminary |
| Complex operators are more robust to the declared generator shifts than the tested controls | QN-000006 at 1,000 cases: complex 0.896/0.660 versus two-channel 0.828/0.597 and real 0.773/0.531; paired three-seed CIs exclude zero | Only two project-designed shifts, three seeds, one training world, no external simulator | Medium-low | Preliminary |
| Complex structure adds robustness beyond two real output channels | QN-000006 complex-minus-two-channel top-1 is +0.068 nuisance and +0.064 noisy/sparse at 1,000 cases | Two-channel control is not algebraically exhaustive; calibration is not better; small seed count | Low | Preliminary |
| Complex arithmetic is superior overall | Shift robustness supports one dimension | GRU wins low-data in-domain; real is better calibrated; complex is slower; evidence is synthetic | Low | Not supported |
| Q-Neuro is novel | No systematic novelty review completed | Many components have direct prior art; combinations may also exist | Very low | Unresolved |
| Q-Neuro has clinical diagnostic value | None; only synthetic archetypes tested | No real data, external validation, prospective study, or medical-device evaluation | None | Unsupported |
