# Independent-task measurement amendment 001

Trigger: post-run inspection of structural audit `QN-000035`  
Affected version: generator `1.0.0`  
Corrected version: generator `1.0.1`  
Outcome status: no architecture was trained or evaluated on these tasks

`QN-000035` reported a field named `empirical_order_target_mutual_information`. The implementation
computed mutual information between the **latent direction variable** and the target. In
causal-order families the target contains that direction by construction, so the value remained
near `ln(2)` even when the observed marker order had been randomized. The field therefore did not
measure observed order dependence.

Version `1.0.1` makes the following frozen correction:

1. infer the observed direction from the actual positions of the two marker tokens;
2. report `empirical_observed_order_target_mutual_information` from that observed direction;
3. retain the former quantity under the accurate diagnostic name
   `latent_direction_target_mutual_information`;
4. add the generator version to every dataset and make the audit fail on a version mismatch;
5. preserve `QN-000035` unchanged and rerun the structural audit under a new experiment ID.

No task outcomes, model comparisons, thresholds, labels, sequences, or counterfactual rules were
changed. The correction occurred before discovery-model training and cannot be based on an
architecture result.
