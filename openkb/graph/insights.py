"""Insight generation: orphans, sparse communities, bridge nodes, surprising connections.

Combines graph structure and community assignments to surface knowledge gaps
and unexpected cross-community links.
"""
from __future__ import annotations

from typing import Callable

import networkx as nx

from openkb.graph.community import detect_communities, compute_cohesion
from openkb.graph.relevance import relevance_score


def find_orphans(
    graph: nx.Graph, max_degree: int = 1
) -> list[tuple[str, int]]:
    """Find nodes with degree <= max_degree, sorted ascending by degree."""
    orphans = []
    for node in graph.nodes:
        deg = graph.degree(node)
        if deg <= max_degree:
            orphans.append((node, deg))
    orphans.sort(key=lambda x: x[1])
    return orphans


def find_sparse_communities(
    graph: nx.Graph,
    communities: dict[int, list[str]],
    threshold: float = 0.15,
) -> list[tuple[int, str, float]]:
    """Find sparse communities with a representative label and cohesion.

    Representative label is the most-connected node name in the community.
    Returns [(community_id, representative, cohesion), ...].
    """
    result = []
    for comm_id, members in communities.items():
        cohesion = compute_cohesion(graph, members)
        if cohesion < threshold:
            # Find most-connected member
            rep = max(members, key=lambda n: graph.degree(n)) if members else ""
            result.append((comm_id, rep, cohesion))
    return result


def _build_node_to_comm(communities: dict[int, list[str]]) -> dict[str, int]:
    """Build reverse mapping: node -> community_id."""
    node_to_comm: dict[str, int] = {}
    for comm_id, members in communities.items():
        for m in members:
            node_to_comm[m] = comm_id
    return node_to_comm


def find_bridge_nodes(
    graph: nx.Graph,
    communities: dict[int, list[str]],
    min_communities: int = 3,
) -> list[tuple[str, int]]:
    """Find nodes whose neighbours span >= min_communities distinct partitions.

    Returns [(node_name, community_span_count), ...].
    """
    node_to_comm = _build_node_to_comm(communities)

    bridges = []
    for node in graph.nodes:
        neighbor_comms = set()
        for nb in graph.neighbors(node):
            if nb in node_to_comm:
                neighbor_comms.add(node_to_comm[nb])
        # Exclude own community
        own_comm = node_to_comm.get(node)
        if own_comm is not None:
            neighbor_comms.discard(own_comm)
        if len(neighbor_comms) >= min_communities:
            bridges.append((node, len(neighbor_comms)))
    bridges.sort(key=lambda x: x[1], reverse=True)
    return bridges


def find_surprising_connections(
    graph: nx.Graph,
    communities: dict[int, list[str]],
    relevance_fn: Callable[[nx.Graph, str, str], float] = relevance_score,
) -> list[tuple[str, str, str, float]]:
    """Find cross-community edges and label them.

    Returns [(source, target, reason, score), ...].
    Reason is 'cross-community' or 'type-variant' (different entity_type).
    """
    node_to_comm = _build_node_to_comm(communities)

    surprising = []
    for u, v in graph.edges:
        comm_u = node_to_comm.get(u)
        comm_v = node_to_comm.get(v)
        if comm_u is None or comm_v is None:
            continue
        if comm_u != comm_v:
            type_u = graph.nodes[u].get("entity_type", "")
            type_v = graph.nodes[v].get("entity_type", "")
            reason = "type-variant" if (type_u and type_v and type_u != type_v) else "cross-community"
            score = relevance_fn(graph, u, v)
            surprising.append((u, v, reason, score))

    surprising.sort(key=lambda x: x[3], reverse=True)
    return surprising


def generate_insights(graph: nx.Graph) -> dict:
    """Convenience wrapper: run all insight functions + community detection.

    Returns dict with keys: orphans, sparse_communities, bridge_nodes,
    surprising_connections, communities_summary.
    """
    if graph.number_of_nodes() == 0:
        return {
            "orphans": [],
            "sparse_communities": [],
            "bridge_nodes": [],
            "surprising_connections": [],
            "communities_summary": {},
        }

    communities = detect_communities(graph)

    orphans = find_orphans(graph)
    sparse = find_sparse_communities(graph, communities)
    bridges = find_bridge_nodes(graph, communities)
    surprising = find_surprising_connections(graph, communities)

    communities_summary: dict[int, dict] = {}
    for comm_id, members in communities.items():
        cohesion = compute_cohesion(graph, members)
        rep = max(members, key=lambda n: graph.degree(n)) if members else ""
        communities_summary[comm_id] = {
            "size": len(members),
            "cohesion": round(cohesion, 3),
            "representative": rep,
        }

    return {
        "orphans": orphans,
        "sparse_communities": sparse,
        "bridge_nodes": bridges,
        "surprising_connections": surprising,
        "communities_summary": communities_summary,
    }