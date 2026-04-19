"""Graph construction from wiki/ markdown files.

Walks wiki/ subdirectories, extracts [[target]] wikilinks and YAML frontmatter
(sources, entity_type), builds a networkx Graph, and serialises to graph.json.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from openkb.frontmatter import parse_fm

logger = logging.getLogger(__name__)

# Regex for [[target]] wikilinks
_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")

# Wiki subdirectories to scan
_WIKI_SUBDIRS = ("summaries", "concepts", "explorations")


class GraphLoadError(Exception):
    """Raised when graph.json cannot be loaded (corrupted or missing)."""


def _node_id(rel_path: str) -> str:
    """Convert a relative path like 'concepts/foo.md' to a node ID 'concepts/foo'."""
    if rel_path.endswith(".md"):
        rel_path = rel_path[:-3]
    return rel_path


def build_graph(wiki_dir: Path) -> nx.Graph:
    """Build a networkx Graph from wiki/ markdown files.

    Nodes are wiki page slugs (e.g. 'concepts/attention').
    Edges carry attributes: edge_type (wikilink, source_overlap, entity_mention), weight.
    Node attributes: entity_type, brief, sources, mentioned_entities.
    """
    g = nx.Graph()

    # Collect all .md files
    page_data: dict[str, dict] = {}  # node_id -> {meta, body, targets}

    for subdir in _WIKI_SUBDIRS:
        sub_path = wiki_dir / subdir
        if not sub_path.exists():
            continue
        for md_file in sorted(sub_path.glob("*.md")):
            rel = md_file.relative_to(wiki_dir)
            nid = _node_id(str(rel))
            text = md_file.read_text(encoding="utf-8")
            meta, body = parse_fm(text)

            # Extract wikilinks from body
            targets = []
            for match in _WIKILINK_RE.finditer(body):
                raw_target = match.group(1).strip()
                # Normalise: strip .md suffix
                if raw_target.endswith(".md"):
                    raw_target = raw_target[:-3]
                targets.append(raw_target)

            page_data[nid] = {
                "meta": meta,
                "body": body,
                "targets": targets,
            }

            # Extract mentioned entity names for relevance signal
            ent_names = []
            for ent in meta.get("entities", []):
                if isinstance(ent, dict) and "name" in ent:
                    ent_names.append(ent["name"])

            g.add_node(nid,
                       entity_type=meta.get("entity_type", ""),
                       brief=meta.get("brief", ""),
                       sources=meta.get("sources", []),
                       mentioned_entities=ent_names)

    # Add wikilink edges
    for nid, data in page_data.items():
        for target in data["targets"]:
            if target == nid:
                continue  # skip self-loops
            # Create placeholder node if target doesn't exist
            if target not in g.nodes:
                g.add_node(target, entity_type="", brief="", sources=[], mentioned_entities=[])
            if not g.has_edge(nid, target):
                g.add_edge(nid, target, edge_type="wikilink", weight=1)

    # Add source_overlap edges
    # Group pages by shared source strings
    source_to_nodes: dict[str, list[str]] = {}
    for nid, data in page_data.items():
        for src in data["meta"].get("sources", []):
            source_to_nodes.setdefault(src, []).append(nid)

    for src, nodes in source_to_nodes.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if a == b:
                    continue
                if g.has_edge(a, b):
                    # Augment existing edge weight for source overlap
                    edge_data = g.get_edge_data(a, b)
                    current_weight = edge_data.get("weight", 1)
                    g[a][b]["weight"] = current_weight + 1
                    g[a][b]["edge_type"] = "wikilink+source_overlap"
                else:
                    g.add_edge(a, b, edge_type="source_overlap", weight=1)

    # Add entity_mention edges
    entity_to_nodes: dict[str, list[str]] = {}
    for nid, data in page_data.items():
        for ent in data["meta"].get("entities", []):
            if isinstance(ent, dict) and "name" in ent and "type" in ent:
                key = f"{ent['name']}|{ent['type']}"
                entity_to_nodes.setdefault(key, []).append(nid)

    for key, nodes in entity_to_nodes.items():
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a, b = nodes[i], nodes[j]
                if g.has_edge(a, b):
                    edge_data = g.get_edge_data(a, b)
                    current_weight = edge_data.get("weight", 1)
                    g[a][b]["weight"] = current_weight + 1
                    existing_type = edge_data.get("edge_type", "wikilink")
                    g[a][b]["edge_type"] = f"{existing_type}+entity_mention"
                else:
                    g.add_edge(a, b, edge_type="entity_mention", weight=1)

    return g


def save_graph(graph: nx.Graph, path: Path) -> None:
    """Serialise graph to JSON file."""
    nodes: dict = {}
    for nid in graph.nodes:
        attr = dict(graph.nodes[nid])
        attr["degree"] = graph.degree(nid)
        nodes[nid] = attr

    edges: list[dict] = []
    for u, v in graph.edges:
        edge_data = dict(graph.edges[u, v])
        edges.append({
            "source": u,
            "target": v,
            "weight": edge_data.get("weight", 1),
            "edge_type": edge_data.get("edge_type", "wikilink"),
        })

    data = {
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "build_timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_graph(path: Path) -> nx.Graph:
    """Deserialise graph from JSON file. Raises GraphLoadError on failure."""
    if not path.exists():
        raise GraphLoadError(str(path), "File not found")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise GraphLoadError(str(path), str(exc)) from exc
    g = nx.Graph()

    for nid, attr in data.get("nodes", {}).items():
        # Remove degree from node attributes (it's derived)
        attr_copy = {k: v for k, v in attr.items() if k != "degree"}
        g.add_node(nid, **attr_copy)

    for edge in data.get("edges", []):
        g.add_edge(
            edge["source"],
            edge["target"],
            weight=edge.get("weight", 1),
            edge_type=edge.get("edge_type", "wikilink"),
        )

    return g


def build_and_save_graph(wiki_dir: Path, openkb_dir: Path | None = None) -> tuple[nx.Graph, Path]:
    """Build graph and save to .openkb/graph.json. Returns (graph, path)."""
    if openkb_dir is None:
        # Walk up from wiki_dir to find .openkb
        current = wiki_dir.parent
        while current != current.parent:
            if (current / ".openkb").is_dir():
                openkb_dir = current / ".openkb"
                break
            current = current.parent
        if openkb_dir is None:
            openkb_dir = wiki_dir.parent / ".openkb"

    graph_path = openkb_dir / "graph.json"
    g = build_graph(wiki_dir)
    save_graph(g, graph_path)
    logger.info("Graph built: %d nodes, %d edges → %s",
                g.number_of_nodes(), g.number_of_edges(), graph_path)
    return g, graph_path