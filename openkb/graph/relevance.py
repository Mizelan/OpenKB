"""5-signal relevance scoring for graph nodes.

Signals:
  - direct_link: 3.0 if edge exists between nodes
  - source_overlap: shared source count * 4.0
  - adamic_adar: sum of 1/log(degree(n)) for shared neighbors * 1.5
  - type_affinity: 1.0 if same non-empty entity_type
  - entity_mention: shared mentioned_entities count * 2.0
"""
from __future__ import annotations

import math

import networkx as nx


def relevance_score(graph: nx.Graph, node_a: str, node_b: str) -> float:
    """Compute weighted relevance score between two nodes."""
    score = 0.0

    # 1. Direct link
    if graph.has_edge(node_a, node_b):
        score += 3.0

    # 2. Source overlap
    sources_a = set(graph.nodes[node_a].get("sources", []) or [])
    sources_b = set(graph.nodes[node_b].get("sources", []) or [])
    shared_sources = len(sources_a & sources_b)
    score += shared_sources * 4.0

    # 3. Adamic-Adar
    neighbors_a = set(graph.neighbors(node_a))
    neighbors_b = set(graph.neighbors(node_b))
    shared_neighbors = neighbors_a & neighbors_b
    aa_sum = 0.0
    for n in shared_neighbors:
        deg = graph.degree(n)
        if deg > 1:
            aa_sum += 1.0 / math.log(deg)
    score += aa_sum * 1.5

    # 4. Type affinity
    type_a = graph.nodes[node_a].get("entity_type", "")
    type_b = graph.nodes[node_b].get("entity_type", "")
    if type_a and type_b and type_a == type_b:
        score += 1.0

    # 5. Entity mention overlap
    entities_a = set(graph.nodes[node_a].get("mentioned_entities", []) or [])
    entities_b = set(graph.nodes[node_b].get("mentioned_entities", []) or [])
    shared_entities = len(entities_a & entities_b)
    score += shared_entities * 2.0

    return score


def top_related(graph: nx.Graph, seed: str, k: int = 5) -> list[tuple[str, float]]:
    """Return top-k nodes by relevance score relative to seed. Skip seed itself."""
    if seed not in graph.nodes:
        return []

    candidates = []
    for node in graph.nodes:
        if node == seed:
            continue
        score = relevance_score(graph, seed, node)
        if score > 0:
            candidates.append((node, score))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:k]