"""Global room-path partition from sustained co-presence constraints."""

from __future__ import annotations

from collections import Counter
from itertools import combinations, product

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None


def bipartite_room_partition(traj, baseline_owner, *, min_overlap_frames=2):
    """Collapse fragmented paths only when the overlap graph proves two sides.

    Sustained simultaneous tracklets must represent different bodies. A
    one-frame overlap is ignored because tracker handovers and duplicate boxes
    commonly coexist for one frame. Disconnected co-presence episodes are
    oriented to preserve as much of the conservative baseline grouping as
    possible. Non-bipartite rooms are returned unchanged: they contain at
    least three mutually conflicting observations or unresolved tracker noise.
    """
    if nx is None:
        return dict(baseline_owner), {"status": "networkx_unavailable"}
    frames = {
        int(tid): {int(row[0]) for row in rows}
        for tid, rows in traj.items()
        if int(tid) in baseline_owner
    }
    graph = nx.Graph()
    graph.add_nodes_from(frames)
    overlaps = {}
    for tid_a, tid_b in combinations(sorted(frames), 2):
        count = len(frames[tid_a] & frames[tid_b])
        if count >= min_overlap_frames:
            graph.add_edge(tid_a, tid_b)
            overlaps[(tid_a, tid_b)] = count
    diagnostic = {
        "status": "not_applied",
        "min_overlap_frames": int(min_overlap_frames),
        "nodes": len(graph),
        "conflict_edges": len(graph.edges),
        "components": nx.number_connected_components(graph) if graph else 0,
        "baseline_paths": len(set(baseline_owner.values())),
    }
    if not graph.edges:
        diagnostic["reason"] = "no sustained co-presence"
        return dict(baseline_owner), diagnostic
    if not nx.is_bipartite(graph):
        diagnostic["reason"] = "co-presence graph is not bipartite"
        diagnostic["maximal_clique_size"] = max(
            (len(clique) for clique in nx.find_cliques(graph)), default=0
        )
        return dict(baseline_owner), diagnostic

    components = []
    for nodes in nx.connected_components(graph):
        subgraph = graph.subgraph(nodes)
        colors = (
            nx.bipartite.color(subgraph) if subgraph.edges else {next(iter(nodes)): 0}
        )
        components.append((sorted(nodes), colors))
    if len(components) > 20:
        diagnostic["reason"] = "too many disconnected episodes for exact orientation"
        return dict(baseline_owner), diagnostic

    duration = {tid: len(frames[tid]) for tid in frames}
    baseline_paths = sorted(set(baseline_owner.values()))
    best = None
    # Global colour inversion is equivalent, so anchor the first component.
    for tail in product((0, 1), repeat=max(len(components) - 1, 0)):
        orientations = (0, *tail)
        owner = {
            tid: colors[tid] ^ orientation
            for orientation, (nodes, colors) in zip(orientations, components)
            for tid in nodes
        }
        retained = 0
        for path_id in baseline_paths:
            counts = Counter()
            for tid, current_path in baseline_owner.items():
                if current_path == path_id and tid in owner:
                    counts[owner[tid]] += duration[tid]
            retained += max(counts.values(), default=0)
        key = (retained, tuple(-owner[tid] for tid in sorted(owner)))
        if best is None or key > best[0]:
            best = (key, owner)
    owner = {tid: color + 1 for tid, color in best[1].items()}
    total = sum(duration.values())
    diagnostic.update(
        {
            "status": "applied",
            "reason": "sustained co-presence graph has exactly two sides",
            "n_path": 2,
            "baseline_consistency": best[0][0] / max(total, 1),
            "groups": {
                str(path_id): sorted(
                    tid for tid, value in owner.items() if value == path_id
                )
                for path_id in (1, 2)
            },
        }
    )
    return owner, diagnostic
