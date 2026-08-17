"""Q-Neuro 3.0 — clean-slate architecture program.

Q-Neuro 2.0 (`qneuro.equivalence`) is the immune system for this work and its conclusions are
frozen. Nothing here may modify them.

Design constraint discovered by measurement, not assumption: this machine has 8 GiB of unified
memory with under 2 GiB typically free. Quadratic MEMORY is therefore the binding constraint, ahead
of FLOPs. Architectures are selected accordingly.
"""
