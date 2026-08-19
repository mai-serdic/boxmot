"""
Global trajectory stitching: one person is one path, not a chain of guesses.

The user's last point:

    "the fact that iou only track consecutive frames is really limited - it
    should track the whole motion path of one person, right? as again we need
    to understand how does that person presence in the space."

This is a structural criticism, not a tuning one, and it is correct. Everything
downstream of the detector in this system makes *local* decisions:

  * BoT-SORT associates frame N to frame N+1 by IoU and a Kalman prediction.
  * `ghost_pool` rebinds one lost track to one new track, greedily, at the
    moment the new track appears - with no ability to revise it later.

Both answer "does this box continue that box". Neither ever asks "is the
resulting *path* a plausible way for one person to have moved through this
room". Three things are lost by that:

  1. **Transitivity.** If A-B is a good link and B-C is a good link, then A-C is
     implied. A greedy pairwise matcher can accept A-B and B-C while the implied
     A-C is physically impossible, and never notice.
  2. **Exclusivity.** Two tracklets that overlap in time cannot be the same
     person. Locally that only blocks one pair; globally it constrains the whole
     assignment, because ruling out A-B frees B for C.
  3. **Revision.** A greedy match made at frame 400 is permanent. Evidence
     arriving at frame 900 can make it obviously wrong and cannot undo it.

The fix is the standard one for this shape of problem: treat tracklets as nodes
in a time-ordered graph and solve for the set of *paths* through it that
minimises total cost, rather than picking edges one at a time. Each unit of flow
from source to sink is one person's trajectory through the space, which is
exactly the object the user is asking us to model. Costs come from the step 9
geodesic prior, so the physical scene model is what holds the paths together.

This is offline by construction - it needs the future to revise the past - so it
is a post-pass over a recording, not a replacement for the online tracker.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import networkx as nx
except Exception:                                   # pragma: no cover
    nx = None

OVERLAP_TOL_F = 20      # tracklets may overlap this much and still be the same
                        # person: measured, these are tracker handovers where
                        # the replacement track is spawned before the old dies
P_MIN = 1e-3            # floor on link probability, keeps -log finite
SCALE = 1000            # costs must be integral for the min-cost-flow solver
COVER_REWARD = int(SCALE * 12.0)    # > SCALE * -log(P_MIN); see stitch_global


@dataclass
class Tracklet:
    tid: int
    frames: np.ndarray          # (N,) ascending
    xy: np.ndarray              # (N,2) metres, may contain NaN
    emb: np.ndarray | None = None   # (D,) unit-norm mean embedding, optional

    @property
    def t0(self) -> int:
        return int(self.frames[0])

    @property
    def t1(self) -> int:
        return int(self.frames[-1])

    def end_state(self, fps: float, win: int = 5):
        """(xy, velocity m/s) at the last usable observation."""
        return _state(self.frames, self.xy, len(self.frames) - 1, fps, win, -1)

    def start_state(self, fps: float, win: int = 5):
        return _state(self.frames, self.xy, 0, fps, win, +1)


def _state(f, xy, i, fps, win, direction):
    ok = np.where(np.all(np.isfinite(xy), axis=1))[0]
    if not len(ok):
        return None, (0.0, 0.0)
    i = ok[-1] if direction < 0 else ok[0]
    j = ok[max(0, len(ok) - 1 - win)] if direction < 0 else ok[min(len(ok) - 1, win)]
    dt = (f[i] - f[j]) / fps
    v = (xy[i] - xy[j]) / dt if dt not in (0.0,) else np.zeros(2)
    if direction < 0:
        return tuple(xy[i]), (float(v[0]), float(v[1]))
    # At a tracklet start the velocity measured forward is the one to compare
    # against, so flip the sign convention to "where they came from".
    return tuple(xy[i]), (float(-v[0]), float(-v[1]))


def link_prob(reach, a: Tracklet, b: Tracklet, fps: float, params,
              w_emb: float = 0.0) -> float:
    """P(b continues a). 0 means the graph gets no edge at all."""
    from reachability import rebind_prior

    if b.t0 <= a.t1 - OVERLAP_TOL_F:
        return 0.0                      # too much overlap to be one body
    xy_a, v_a = a.end_state(fps)
    xy_b, _ = b.start_state(fps)
    if xy_a is None or xy_b is None:
        return 0.0
    elapsed = max(b.t0 - a.t1, 1) / fps
    r = rebind_prior(reach, xy_a, v_a, xy_b, elapsed, params)
    if not r["feasible"]:
        return 0.0                      # could not have walked there: no edge
    p = float(r["prior"])
    if w_emb > 0.0 and a.emb is not None and b.emb is not None:
        # Appearance enters as a soft multiplier only. Sections 10-12 measured
        # it at AUC 0.803 across tracklets on a site where people change
        # clothes, so it must not be able to veto a physically sound link.
        d = 1.0 - float(a.emb @ b.emb)
        p *= float(np.clip(1.0 - w_emb * d, 0.05, 1.0))
    return max(p, 0.0)


def stitch_greedy(reach, tracklets, fps=15.0, params=None, w_emb=0.0,
                  thresh=0.25) -> dict[int, int]:
    """What the ghost pool effectively does: walk forward in time, attach each
    new tracklet to the best-scoring earlier one, never revise."""
    from reachability import ReachParams
    params = params or ReachParams()
    ts = sorted(tracklets, key=lambda t: t.t0)
    owner = {t.tid: t.tid for t in ts}          # tid -> identity id
    tail = {t.tid: t for t in ts}               # identity -> its last tracklet
    for b in ts[1:]:
        best, best_p = None, thresh
        for ident, a in list(tail.items()):
            if a.tid == b.tid:
                continue
            p = link_prob(reach, a, b, fps, params, w_emb)
            if p > best_p:
                best, best_p = ident, p
        if best is not None:
            owner[b.tid] = best
            tail[best] = b
        else:
            tail[b.tid] = b
    return owner


def stitch_global(reach, tracklets, fps=15.0, params=None, w_emb=0.0,
                  birth_cost=2.0) -> dict[int, int]:
    """Min-cost flow over the tracklet graph: each unit of flow is one person.

    `birth_cost` is what it costs to declare a new identity. Raising it merges
    more aggressively; it is the one knob, and it has a meaning - the number of
    nats of link implausibility you are willing to tolerate rather than invent
    a new person.
    """
    from reachability import ReachParams
    if nx is None:
        raise RuntimeError("networkx is required for global stitching")
    params = params or ReachParams()
    ts = sorted(tracklets, key=lambda t: t.t0)
    if not ts:
        return {}

    G = nx.DiGraph()
    G.add_node("SRC"); G.add_node("SNK")
    for t in ts:
        u, v = (t.tid, "i"), (t.tid, "o")
        # Capacity 1 across every tracklet: it belongs to exactly one person.
        # The weight is a *reward*, not a cost. Without it the cheapest flow is
        # the one that explains as few tracklets as possible — the solver has no
        # reason to route through a tracklet at all. COVER_REWARD must exceed
        # the worst link cost so that including a tracklet is always worth it,
        # and the assignment is then decided by link costs and the overlap
        # constraints rather than by how much of the data it can ignore.
        G.add_edge(u, v, capacity=1, weight=-COVER_REWARD)
        G.add_edge("SRC", u, capacity=1, weight=int(SCALE * birth_cost / 2))
        G.add_edge(v, "SNK", capacity=1, weight=int(SCALE * birth_cost / 2))
    n_edge = 0
    for i, a in enumerate(ts):
        for b in ts[i + 1:]:
            p = link_prob(reach, a, b, fps, params, w_emb)
            if p <= 0.0:
                continue
            G.add_edge((a.tid, "o"), (b.tid, "i"), capacity=1,
                       weight=int(SCALE * -np.log(max(p, P_MIN))))
            n_edge += 1
    if n_edge == 0:
        return {t.tid: t.tid for t in ts}

    # Model selection over the number of identities. The cost is convex in K,
    # so scanning it is cheap and exact rather than a heuristic stopping rule.
    best, best_cost = None, np.inf
    for k in range(1, len(ts) + 1):
        H = G.copy()
        H.nodes["SRC"]["demand"] = -k
        H.nodes["SNK"]["demand"] = k
        try:
            cost, flow = nx.network_simplex(H)
        except nx.NetworkXUnfeasible:
            continue
        if cost < best_cost:
            best, best_cost = flow, cost
    if best is None:
        return {t.tid: t.tid for t in ts}

    nxt = {}
    for a in ts:
        for dst, f in best.get((a.tid, "o"), {}).items():
            if f > 0 and isinstance(dst, tuple) and dst[1] == "i":
                nxt[a.tid] = dst[0]
    owner, seen = {}, set()
    for t in ts:
        if t.tid in seen:
            continue
        ident, cur = t.tid, t.tid
        while cur is not None:
            owner[cur] = ident
            seen.add(cur)
            cur = nxt.get(cur)
            if cur in seen:
                break
    return owner
