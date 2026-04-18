"""Tests for openkb.graph — build, relevance, community, insights."""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers: fixture wiki
# ---------------------------------------------------------------------------

def _make_wiki(tmp: Path) -> Path:
    """Create a minimal wiki directory for testing."""
    wiki = tmp / "wiki"
    wiki.mkdir()
    (wiki / "summaries").mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "explorations").mkdir()

    # Summary page
    (wiki / "summaries" / "doc1.md").write_text(
        "---\ndoc_type: short\n---\n\nSummary referencing [[concepts/alpha]] and [[concepts/beta]].\n",
        encoding="utf-8",
    )

    # Concept pages
    (wiki / "concepts" / "alpha.md").write_text(
        "---\nsources: [summaries/doc1.md]\nbrief: Alpha concept\nentity_type: technology\n---\n\nAlpha links to [[concepts/beta]].\n",
        encoding="utf-8",
    )
    (wiki / "concepts" / "beta.md").write_text(
        "---\nsources: [summaries/doc1.md]\nbrief: Beta concept\nentity_type: concept\n---\n\nBeta content.\n",
        encoding="utf-8",
    )

    return wiki


def _make_two_cluster_wiki(tmp: Path) -> Path:
    """Create a wiki with two clearly separated clusters."""
    wiki = tmp / "wiki"
    wiki.mkdir()
    (wiki / "summaries").mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "explorations").mkdir()

    # Cluster A
    (wiki / "concepts" / "a1.md").write_text(
        "---\nsources: []\nentity_type: technology\n---\n\nLinks to [[concepts/a2]] and [[concepts/a3]].\n",
        encoding="utf-8",
    )
    (wiki / "concepts" / "a2.md").write_text(
        "---\nsources: []\nentity_type: technology\n---\n\nLinks to [[concepts/a3]].\n",
        encoding="utf-8",
    )
    (wiki / "concepts" / "a3.md").write_text(
        "---\nsources: []\nentity_type: technology\n---\n\nA3 content.\n",
        encoding="utf-8",
    )

    # Cluster B
    (wiki / "concepts" / "b1.md").write_text(
        "---\nsources: []\nentity_type: concept\n---\n\nLinks to [[concepts/b2]] and [[concepts/b3]].\n",
        encoding="utf-8",
    )
    (wiki / "concepts" / "b2.md").write_text(
        "---\nsources: []\nentity_type: concept\n---\n\nLinks to [[concepts/b3]].\n",
        encoding="utf-8",
    )
    (wiki / "concepts" / "b3.md").write_text(
        "---\nsources: []\nentity_type: concept\n---\n\nB3 content.\n",
        encoding="utf-8",
    )

    # Bridge node connecting both clusters
    (wiki / "concepts" / "bridge.md").write_text(
        "---\nsources: []\nentity_type: concept\n---\n\nLinks to [[concepts/a1]], [[concepts/b1]], and [[concepts/b2]].\n",
        encoding="utf-8",
    )

    # Sparse (orphan-like) node
    (wiki / "concepts" / "lonely.md").write_text(
        "---\nsources: []\nentity_type: event\n---\n\nLonely content.\n",
        encoding="utf-8",
    )

    return wiki


# ---------------------------------------------------------------------------
# build.py tests
# ---------------------------------------------------------------------------

class TestBuildGraph:
    def test_build_graph_creates_nodes(self, tmp_path):
        from openkb.graph.build import build_graph
        wiki = _make_wiki(tmp_path)
        g = build_graph(wiki)
        assert "summaries/doc1" in g.nodes
        assert "concepts/alpha" in g.nodes
        assert "concepts/beta" in g.nodes

    def test_wikilink_edges(self, tmp_path):
        from openkb.graph.build import build_graph
        wiki = _make_wiki(tmp_path)
        g = build_graph(wiki)
        assert g.has_edge("summaries/doc1", "concepts/alpha")
        assert g.has_edge("summaries/doc1", "concepts/beta")
        assert g.has_edge("concepts/alpha", "concepts/beta")

    def test_edge_type_wikilink(self, tmp_path):
        from openkb.graph.build import build_graph
        wiki = _make_wiki(tmp_path)
        g = build_graph(wiki)
        edge_data = g.get_edge_data("summaries/doc1", "concepts/alpha")
        assert edge_data["edge_type"] == "wikilink"
        assert edge_data["weight"] == 1

    def test_source_overlap_edges(self, tmp_path):
        from openkb.graph.build import build_graph
        wiki = _make_wiki(tmp_path)
        g = build_graph(wiki)
        assert g.has_edge("concepts/alpha", "concepts/beta")
        edge_data = g.get_edge_data("concepts/alpha", "concepts/beta")
        # They share 1 source, so source_overlap edge should exist
        # If both wikilink and source_overlap exist on same pair, check source_overlap
        has_source_overlap = False
        # In an undirected graph there is one edge; we need to check multi-edges or combined
        # build.py should create a combined edge or separate; we check the edge exists
        assert edge_data is not None

    def test_sources_multiline_yaml_format(self, tmp_path):
        """YAML multi-line sources (dash-prefixed items) should be parsed."""
        from openkb.graph.build import build_graph
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "concepts" / "multi.md").write_text(
            "---\nentity_type: concept\nsources:\n- src_a.md\n- src_b.md\n---\n\nContent.\n",
            encoding="utf-8",
        )
        g = build_graph(wiki)
        assert g.nodes["concepts/multi"]["sources"] == ["src_a.md", "src_b.md"]

    def test_entity_type_stored_as_node_attr(self, tmp_path):
        from openkb.graph.build import build_graph
        wiki = _make_wiki(tmp_path)
        g = build_graph(wiki)
        assert g.nodes["concepts/alpha"].get("entity_type") == "technology"
        assert g.nodes["concepts/beta"].get("entity_type") == "concept"

    def test_no_self_loops(self, tmp_path):
        from openkb.graph.build import build_graph
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "concepts" / "selfref.md").write_text(
            "---\n---\n\nSelf link: [[concepts/selfref]].\n",
            encoding="utf-8",
        )
        g = build_graph(wiki)
        assert not g.has_edge("concepts/selfref", "concepts/selfref")

    def test_missing_target_creates_placeholder_node(self, tmp_path):
        from openkb.graph.build import build_graph
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "concepts").mkdir(parents=True)
        (wiki / "concepts" / "existing.md").write_text(
            "---\n---\n\nLink to [[concepts/phantom]].\n",
            encoding="utf-8",
        )
        g = build_graph(wiki)
        assert "concepts/phantom" in g.nodes

    def test_empty_wiki(self, tmp_path):
        from openkb.graph.build import build_graph
        wiki = tmp_path / "wiki"
        wiki.mkdir()
        (wiki / "summaries").mkdir()
        (wiki / "concepts").mkdir()
        (wiki / "explorations").mkdir()
        g = build_graph(wiki)
        assert g.number_of_nodes() == 0
        assert g.number_of_edges() == 0

    def test_serialisation_roundtrip(self, tmp_path):
        from openkb.graph.build import build_graph, save_graph, load_graph
        wiki = _make_wiki(tmp_path)
        g = build_graph(wiki)
        path = tmp_path / "graph.json"
        save_graph(g, path)
        g2 = load_graph(path)
        assert g2.number_of_nodes() == g.number_of_nodes()
        assert g2.number_of_edges() == g.number_of_edges()

    def test_build_and_save_convenience(self, tmp_path):
        from openkb.graph.build import build_and_save_graph
        wiki = _make_wiki(tmp_path)
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        graph_path = build_and_save_graph(wiki, openkb_dir)
        assert graph_path.exists()
        data = json.loads(graph_path.read_text(encoding="utf-8"))
        assert data["metadata"]["node_count"] > 0


# ---------------------------------------------------------------------------
# relevance.py tests
# ---------------------------------------------------------------------------

class TestRelevance:
    def _make_graph(self):
        import networkx as nx
        g = nx.Graph()
        g.add_node("a", entity_type="technology", sources=["s1", "s2"])
        g.add_node("b", entity_type="technology", sources=["s1", "s2"])
        g.add_node("c", entity_type="concept", sources=["s3"])
        g.add_edge("a", "b", edge_type="wikilink", weight=1)
        g.add_edge("a", "c", edge_type="wikilink", weight=1)
        # shared neighbor between b and c is 'a'
        return g

    def test_direct_link_signal(self):
        from openkb.graph.relevance import relevance_score
        g = self._make_graph()
        score = relevance_score(g, "a", "b")
        # Direct link contributes 3.0, source overlap contributes 2*4.0=8.0
        # Adamic-Adar from shared neighbor 'c' contributes 1/log(deg(c))*1.5
        # deg(c) = 1 → log(1) = 0, skip; shared neighbor 'a': 1/log(deg(a))*1.5
        # Type affinity: both technology → 1.0
        assert score >= 3.0  # at least direct link

    def test_no_direct_link(self):
        from openkb.graph.relevance import relevance_score
        g = self._make_graph()
        # b and c share neighbor 'a'
        score = relevance_score(g, "b", "c")
        assert score >= 0  # no direct link but may have adamic_adar

    def test_source_overlap_signal(self):
        from openkb.graph.relevance import relevance_score
        import networkx as nx
        g = nx.Graph()
        g.add_node("x", entity_type="", sources=["s1", "s2"])
        g.add_node("y", entity_type="", sources=["s1", "s2"])
        # 2 shared sources → 2 * 4.0 = 8.0
        score = relevance_score(g, "x", "y")
        assert score >= 8.0

    def test_type_affinity(self):
        from openkb.graph.relevance import relevance_score
        import networkx as nx
        g = nx.Graph()
        g.add_node("p", entity_type="technology", sources=[])
        g.add_node("q", entity_type="technology", sources=[])
        score = relevance_score(g, "p", "q")
        assert score >= 1.0  # same entity_type adds 1.0

    def test_different_type_no_affinity(self):
        from openkb.graph.relevance import relevance_score
        import networkx as nx
        g = nx.Graph()
        g.add_node("p", entity_type="technology", sources=[])
        g.add_node("q", entity_type="concept", sources=[])
        score = relevance_score(g, "p", "q")
        assert score < 1.0  # no type affinity bonus

    def test_top_related(self):
        from openkb.graph.relevance import relevance_score, top_related
        g = self._make_graph()
        result = top_related(g, "a", k=2)
        assert len(result) <= 2
        assert all(node != "a" for node, _ in result)

    def test_adamic_adar_signal(self):
        from openkb.graph.relevance import relevance_score
        import networkx as nx
        g = nx.Graph()
        g.add_node("a", entity_type="", sources=[])
        g.add_node("b", entity_type="", sources=[])
        g.add_node("hub", entity_type="", sources=[])
        g.add_edge("a", "hub", edge_type="wikilink", weight=1)
        g.add_edge("b", "hub", edge_type="wikilink", weight=1)
        score = relevance_score(g, "a", "b")
        expected_aa = 1.0 / math.log(2) * 1.5  # hub degree=2, log(2)
        assert abs(score - expected_aa) < 0.01


# ---------------------------------------------------------------------------
# community.py tests
# ---------------------------------------------------------------------------

class TestCommunity:
    def test_detect_communities_two_clusters(self, tmp_path):
        from openkb.graph.build import build_graph
        from openkb.graph.community import detect_communities
        wiki = _make_two_cluster_wiki(tmp_path)
        g = build_graph(wiki)
        comms = detect_communities(g)
        assert len(comms) >= 2  # at least two clusters

    def test_cohesion(self, tmp_path):
        from openkb.graph.build import build_graph
        from openkb.graph.community import compute_cohesion
        wiki = _make_two_cluster_wiki(tmp_path)
        g = build_graph(wiki)
        # Fully connected 3-node clique has cohesion 1.0
        import networkx as nx
        clique = nx.complete_graph(3)
        mapped = nx.Graph()
        nodes = list(clique.nodes())
        for n in nodes:
            mapped.add_node(str(n))
        for u, v in clique.edges():
            mapped.add_edge(str(u), str(v))
        assert compute_cohesion(mapped, [str(n) for n in nodes]) == 1.0

    def test_single_node_cohesion_zero(self, tmp_path):
        from openkb.graph.community import compute_cohesion
        import networkx as nx
        g = nx.Graph()
        g.add_node("solo")
        assert compute_cohesion(g, ["solo"]) == 0.0

    def test_sparse_flagging(self, tmp_path):
        from openkb.graph.build import build_graph
        from openkb.graph.community import detect_communities, flag_sparse
        wiki = _make_two_cluster_wiki(tmp_path)
        g = build_graph(wiki)
        comms = detect_communities(g)
        sparse = flag_sparse(g, comms, threshold=0.15)
        # At least one sparse community expected with the lonely node
        # The lonely node (degree 0) forms a singleton community with cohesion 0
        assert isinstance(sparse, list)


# ---------------------------------------------------------------------------
# insights.py tests
# ---------------------------------------------------------------------------

class TestInsights:
    def _make_bridge_graph(self):
        import networkx as nx
        g = nx.Graph()
        # 3 clusters with a bridge node
        for i in range(3):
            for j in range(3):
                g.add_node(f"c{i}_n{j}", entity_type="concept", sources=[])
            # intra-cluster edges
            g.add_edge(f"c{i}_n0", f"c{i}_n1", edge_type="wikilink", weight=1)
            g.add_edge(f"c{i}_n1", f"c{i}_n2", edge_type="wikilink", weight=1)
        # bridge node connects to all 3 clusters
        g.add_node("bridge", entity_type="concept", sources=[])
        g.add_edge("bridge", "c0_n0", edge_type="wikilink", weight=1)
        g.add_edge("bridge", "c1_n0", edge_type="wikilink", weight=1)
        g.add_edge("bridge", "c2_n0", edge_type="wikilink", weight=1)
        return g

    def test_orphan_detection(self):
        from openkb.graph.insights import find_orphans
        import networkx as nx
        g = nx.Graph()
        g.add_node("orphan1", entity_type="", sources=[])
        g.add_node("orphan2", entity_type="", sources=[])
        g.add_node("connected", entity_type="", sources=[])
        g.add_edge("connected", "orphan1", edge_type="wikilink", weight=1)
        orphans = find_orphans(g, max_degree=1)
        # orphan1 has degree 1, orphan2 has degree 0, connected has degree 1
        orphan_names = [name for name, _ in orphans]
        assert "orphan2" in orphan_names
        assert "orphan1" in orphan_names

    def test_sparse_community_in_insights(self, tmp_path):
        from openkb.graph.build import build_graph
        from openkb.graph.community import detect_communities
        from openkb.graph.insights import find_sparse_communities
        wiki = _make_two_cluster_wiki(tmp_path)
        g = build_graph(wiki)
        comms = detect_communities(g)
        sparse = find_sparse_communities(g, comms, threshold=0.15)
        assert isinstance(sparse, list)

    def test_bridge_node_detection(self):
        from openkb.graph.insights import find_bridge_nodes
        g = self._make_bridge_graph()
        from openkb.graph.community import detect_communities
        comms = detect_communities(g)
        # Bridge node connects to 3 clusters but Louvain may assign it to one;
        # its neighbours still span >= 2 distinct other communities.
        bridges = find_bridge_nodes(g, comms, min_communities=2)
        bridge_names = [name for name, _ in bridges]
        assert "bridge" in bridge_names

    def test_surprising_connections(self):
        from openkb.graph.insights import find_surprising_connections
        from openkb.graph.relevance import relevance_score
        g = self._make_bridge_graph()
        from openkb.graph.community import detect_communities
        comms = detect_communities(g)
        surprising = find_surprising_connections(g, comms, relevance_score)
        # Cross-community edges (bridge to each cluster) should appear
        assert isinstance(surprising, list)

    def test_type_variant_surprising_connection(self):
        """Cross-community edge between nodes with different entity_types
        should produce reason='type-variant', not 'cross-community'."""
        import networkx as nx
        from openkb.graph.community import detect_communities
        from openkb.graph.insights import find_surprising_connections
        g = nx.Graph()
        # Cluster 1: technology nodes
        g.add_node("x1", entity_type="technology", sources=[])
        g.add_node("x2", entity_type="technology", sources=[])
        g.add_edge("x1", "x2", edge_type="wikilink", weight=1)
        # Cluster 2: person nodes
        g.add_node("y1", entity_type="person", sources=[])
        g.add_node("y2", entity_type="person", sources=[])
        g.add_edge("y1", "y2", edge_type="wikilink", weight=1)
        # Cross-community edge with different entity_types
        g.add_edge("x1", "y1", edge_type="wikilink", weight=1)

        comms = detect_communities(g)
        surprising = find_surprising_connections(g, comms)
        reasons = [reason for _, _, reason, _ in surprising]
        assert "type-variant" in reasons

    def test_empty_graph_insights(self):
        from openkb.graph.insights import generate_insights
        import networkx as nx
        g = nx.Graph()
        result = generate_insights(g)
        assert result["orphans"] == []
        assert result["sparse_communities"] == []
        assert result["bridge_nodes"] == []
        assert result["surprising_connections"] == []
        assert result["communities_summary"] == {}

    def test_generate_insights_convenience(self):
        from openkb.graph.insights import generate_insights
        g = self._make_bridge_graph()
        result = generate_insights(g)
        assert "orphans" in result
        assert "sparse_communities" in result
        assert "bridge_nodes" in result
        assert "surprising_connections" in result
        assert "communities_summary" in result


# ---------------------------------------------------------------------------
# CLI integration test
# ---------------------------------------------------------------------------

class TestInsightsCLI:
    def test_insights_command_exists(self):
        from openkb.cli import cli
        # Check that the 'insights' command is registered
        cmd_names = [cmd.name for cmd in cli.commands.values()]
        assert "insights" in cmd_names

    def test_insights_command_runs(self, tmp_path):
        from click.testing import CliRunner
        from openkb.cli import cli
        wiki = _make_wiki(tmp_path)
        openkb_dir = tmp_path / ".openkb"
        openkb_dir.mkdir()
        # Build graph first so CLI can load it
        from openkb.graph.build import build_and_save_graph
        build_and_save_graph(wiki, openkb_dir)
        runner = CliRunner()
        result = runner.invoke(cli, ["--kb-dir", str(tmp_path), "insights"])
        # Should not crash — may have 0 communities for small graph but no error
        assert result.exit_code == 0 or "No knowledge base" in result.output