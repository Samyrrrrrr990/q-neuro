# Independent-task generator amendment 002

Trigger: world-unit review before architecture discovery  
Parent version: `1.0.1`  
New version: `1.1.0`  
Outcome status: no architecture has been trained on any independent task

Changing only the case-sampling seed does not create a structurally independent environment. To
avoid pseudoreplication, version `1.1.0` introduces an explicit `world_seed`. It deterministically
changes the label-independent support-finding distribution, background sign-noise rate, and
nuisance polarity while preserving:

- causal marker identities;
- label rules;
- controlled order dependence;
- counterfactual evidence multisets;
- class space;
- split and shift definitions.

The structural audit now records `world_seed` and requires generator version `1.1.0`. Version
`1.0.1` artifacts remain unchanged. This amendment occurs before independent-task model outcomes,
so it cannot favor an architecture.
