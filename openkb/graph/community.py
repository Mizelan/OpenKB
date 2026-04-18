"""Community detection using Louvain algorithm.

Provides community partitioning, cohesion computation, and sparse community flagging.
"""
from __future__ import annotations

import networkx as nx


def detect_communities(graph: nx.Graph) -> dict[int, list[str]]:
    """Run Louvain community detection. Returns {community_id: [node_names]}."""
    import community as community_louvain

    partition = community_louvain.best_partition(graph)
    communities: dict[int, list[str]] = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)
    return communities


def compute_cohesion(graph: nx.Graph, members: list[str]) -> float:
    """Compute cohesion for a community: internal edge density / max possible edges.

    Single-node community returns 0.0.
    """
    n = len(members)
    if n <= 1:
        return 0.0

    member_set = set(members)
    internal_edges = 0
    for u, v in graph.edges:
        if u in member_set and v in member_set:
            internal_edges += 1

    max_edges = n * (n - 1) / 2
    return internal_edges / max_edges if max_edges > 0 else 0.0


def flag_sparse(
    graph: nx.Graph,
    communities: dict[int, list[str]],
    threshold: float = 0.15,
) -> list[tuple[int, float]]:
    """Return community ids and cohesion values below threshold.

    Returns [(community_id, cohesion), ...].
    """
    sparse = []
    for comm_id, members in communities.items():
        cohesion = compute_cohesion(graph, members)
        if cohesion < threshold:
            sparse.append((comm_id, cohesion))
    return sparse