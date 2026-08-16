"""Tasks whose per-example difficulty is a property of the DATA and is inferable only by working.

`chase_to_goal` is the reference task. A permutation defines a single cycle through the nodes; the
model starts somewhere on it and must report how many hops away the goal node is. The answer is
discoverable only by following the chain, the required number of steps varies per example, and
nothing in the input announces it. That is what makes adaptive depth meaningful rather than
handed over.
"""

from __future__ import annotations

import torch


def chase_to_goal_weighted(
    batch: int, n_nodes: int, max_hops: int, seed: int, weights=None, n_labels: int = 6
) -> dict:
    """As chase_to_goal, with a controllable difficulty distribution and a goal LABEL.

    The label is attached to the goal node and is independent of the distance, so a model can only
    emit it by arriving. That decouples the answer from the step count."""

    import torch as _t
    g = _t.Generator().manual_seed(seed)
    w = _t.ones(max_hops) if weights is None else _t.tensor(weights, dtype=_t.float)
    perms, starts, dists, labels = [], [], [], []
    while len(perms) < batch:
        order = _t.randperm(n_nodes, generator=g)
        perm = _t.empty(n_nodes, dtype=_t.long); perm[order] = order.roll(-1)
        pos = int((order == 0).nonzero()[0])
        want = int(_t.multinomial(w, 1, generator=g)) + 1
        s_idx = (pos - want) % n_nodes
        perms.append(perm); starts.append(order[s_idx]); dists.append(want)
        labels.append(int(_t.randint(0, n_labels, (1,), generator=g)))
    return {"perm": _t.stack(perms), "start": _t.stack(starts),
            "target": _t.tensor(dists), "label": _t.tensor(labels)}


def chase_to_goal(batch: int, n_nodes: int, max_hops: int, seed: int) -> dict[str, torch.Tensor]:
    """Distance from `start` to node 0 along a random single cycle, truncated to `max_hops`."""

    generator = torch.Generator().manual_seed(seed)
    perms, starts, dists = [], [], []
    while len(perms) < batch:
        order = torch.randperm(n_nodes, generator=generator)
        perm = torch.empty(n_nodes, dtype=torch.long)
        perm[order] = order.roll(-1)  # one cycle covering every node
        pos = int((order == 0).nonzero()[0])
        s_idx = int(torch.randint(0, n_nodes, (1,), generator=generator))
        d = (pos - s_idx) % n_nodes
        if d == 0 or d > max_hops:
            continue
        perms.append(perm)
        starts.append(order[s_idx])
        dists.append(d)
    return {
        "perm": torch.stack(perms),
        "start": torch.stack(starts),
        "target": torch.tensor(dists),
    }
