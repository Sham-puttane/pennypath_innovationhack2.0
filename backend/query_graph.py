#!/usr/bin/env python3
"""
First Dollar — Knowledge Graph Query API
Maps user intake answers (Q1-Q5) to graph entry nodes,
traverses relationships, and returns structured results
for the FOO engine and frontend visualization.

Usage:
    from query_graph import GraphQuery
    gq = GraphQuery()
    result = gq.traverse_from_profile(q1="no", q2="gig", q3="credit_card", q4="nothing", q5=["none"])
    # Or for Q6 free text:
    result = gq.search("landlord threatening eviction")
"""

import json
import re
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np
import os
import requests

ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "output"

# For embedding queries (Q6 semantic search)
def _load_api_keys():
    keys_str = os.environ.get("GEMINI_API_KEYS", "")
    if keys_str:
        return [k.strip() for k in keys_str.split(",") if k.strip()]
    single = os.environ.get("GRAPHRAG_API_KEY", "")
    if single:
        return [single]
    return ["YOUR_API_KEY_HERE"]

EMBEDDING_API_KEYS = _load_api_keys()
EMBEDDING_MODEL = "gemini-embedding-001"


# ─── Answer-to-Entity Mapping ───────────────────────────────────────────────
# Maps each Q1-Q5 answer to keywords that find entry nodes in the graph.
# Multiple keywords per answer increase coverage.

Q1_MAP = {
    # Do you have a bank account?
    "yes": ["CHECKING ACCOUNT", "SAVINGS ACCOUNT", "BANK ACCOUNT"],
    "no": ["PREPAID CARD", "MONEY ORDER"],
    "have_but_dont_use": ["CHECKING ACCOUNT", "BANK ACCOUNT", "FEES"],
}

Q2_MAP = {
    # How do you get paid?
    "salary": ["INCOME", "DIRECT DEPOSIT", "PAYCHECK"],
    "gig": ["INCOME", "IRREGULAR INCOME", "SELF-EMPLOYMENT"],
    "cash": ["INCOME", "CASH", "MONEY ORDER"],
    "irregular": ["INCOME", "IRREGULAR INCOME", "PUBLIC BENEFITS"],
}

Q3_MAP = {
    # Do you have debt? What kind?
    "none": [],
    "credit_card": ["CREDIT CARD DEBT", "CREDIT CARD", "DEBT"],
    "student": ["STUDENT LOANS", "FEDERAL STUDENT LOANS", "DEBT"],
    "medical": ["MEDICAL DEBT", "DEBT", "DEBT COLLECTOR"],
    "multiple": ["DEBT", "CREDIT CARD DEBT", "STUDENT LOANS", "MEDICAL DEBT", "DEBT COLLECTOR"],
}

Q4_MAP = {
    # How much have you saved?
    "nothing": ["SAVING", "EMERGENCY FUND", "SAVINGS ACCOUNT"],
    "under_500": ["SAVING", "EMERGENCY FUND", "SAVINGS ACCOUNT"],
    "500_to_1000": ["SAVINGS ACCOUNT", "SAVING", "EMERGENCY SAVINGS FUND"],
    "1000_to_5000": ["SAVINGS ACCOUNT", "SAVINGS PLAN", "EMERGENCY FUND"],
    "over_5000": ["SAVINGS ACCOUNT", "SAVINGS PLAN"],
}

# Q5 is multi-select (checkboxes)
Q5_MAP = {
    # Do you have insurance?
    "renters": ["RENTERS INSURANCE", "STATE FARM"],
    "health": ["HEALTH INSURANCE"],
    "auto": ["AUTO INSURANCE", "CAR INSURANCE", "STATE FARM"],
    "life": ["LIFE INSURANCE", "STATE FARM"],
    "none": ["INSURANCE", "RENTERS INSURANCE", "AUTO INSURANCE", "LIFE INSURANCE", "STATE FARM"],
}

# Node color mapping based on entity type
TYPE_TO_COLOR = {
    "FINANCIAL_CONCEPT": "info",       # blue/purple — concepts
    "ACTION": "green",                 # green — action steps
    "FINANCIAL_PRODUCT": "purple",     # purple — products (entry-like)
    "ORGANIZATION": "amber",           # amber — resources
    "RISK": "red",                     # red — risk chains
    "GEO": "gray",                     # gray — locations
    "PERSON": "gray",                  # gray — people
    "EVENT": "amber",                  # amber — life events
}

# For FOO categorization
FOO_CATEGORIES = {
    "ACTION": "action",
    "RISK": "risk",
    "FINANCIAL_PRODUCT": "product",
    "ORGANIZATION": "resource",
    "FINANCIAL_CONCEPT": "concept",
    "EVENT": "event",
}


class GraphQuery:
    """Query interface for the First Dollar knowledge graph."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self._load_data()

    def _load_data(self):
        """Load graph, entities, relationships, and communities."""
        # Load NetworkX graph
        self.G = nx.read_graphml(str(self.output_dir / "knowledge_graph.graphml"))

        # Load entities for lookup
        entities_raw = json.loads(
            (self.output_dir / "entities.json").read_text(encoding="utf-8")
        )
        self.entities = {e["name"]: e for e in entities_raw}

        # Load communities
        self.communities = json.loads(
            (self.output_dir / "communities.json").read_text(encoding="utf-8")
        )

        # Build entity -> community mapping
        self.entity_to_community = {}
        for comm in self.communities:
            for member in comm.get("members", []):
                self.entity_to_community[member] = comm["id"]

        # Load community reports if available
        reports_path = self.output_dir / "community_reports.json"
        if reports_path.exists():
            self.community_reports = json.loads(
                reports_path.read_text(encoding="utf-8")
            )
        else:
            self.community_reports = None

        # Load entity embeddings if available
        npz_path = self.output_dir / "entity_embeddings.npz"
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=True)
            self.embedding_names = list(data["names"])
            self.embedding_vectors = data["vectors"]  # (N, 3072)
            # Normalize for cosine similarity
            norms = np.linalg.norm(self.embedding_vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.embedding_vectors_normed = self.embedding_vectors / norms
            self.has_embeddings = True
            print(f"Embeddings loaded: {len(self.embedding_names)} entities, dim={self.embedding_vectors.shape[1]}")
        else:
            self.has_embeddings = False

        print(f"Graph loaded: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges, {len(self.communities)} communities")

    def _find_entry_nodes(self, keywords: list[str]) -> list[str]:
        """Find graph nodes matching keywords (exact or fuzzy)."""
        found = []
        for kw in keywords:
            kw_upper = kw.upper()
            # Exact match first
            if kw_upper in self.G.nodes:
                found.append(kw_upper)
            else:
                # Fuzzy: find nodes containing the keyword
                for node in self.G.nodes:
                    if kw_upper in node:
                        found.append(node)
                        break  # Take first match per keyword
        return list(dict.fromkeys(found))  # deduplicate preserving order

    def _bfs_traverse(self, entry_nodes: list[str], max_depth: int = 2, max_nodes: int = 80) -> dict:
        """BFS from entry nodes, collecting typed nodes by depth."""
        visited = set()
        result = {
            "entry_nodes": [],
            "risk_nodes": [],
            "action_nodes": [],
            "resource_nodes": [],
            "concept_nodes": [],
            "all_nodes": [],
            "edges": [],
        }

        queue = [(node, 0) for node in entry_nodes if node in self.G.nodes]
        visited.update(entry_nodes)

        while queue and len(result["all_nodes"]) < max_nodes:
            node, depth = queue.pop(0)

            if node not in self.G.nodes:
                continue

            node_data = self.G.nodes[node]
            entity_info = self.entities.get(node, {})
            node_type = node_data.get("type", entity_info.get("type", "UNKNOWN")).upper().strip('"')

            node_record = {
                "name": node,
                "type": node_type,
                "color": TYPE_TO_COLOR.get(node_type, "gray"),
                "description": entity_info.get("description", node_data.get("description", ""))[:300],
                "community_id": self.entity_to_community.get(node),
                "depth": depth,
                "degree": self.G.degree(node),
                "foo_category": FOO_CATEGORIES.get(node_type, "concept"),
            }

            result["all_nodes"].append(node_record)

            # Categorize
            if depth == 0:
                node_record["color"] = "purple"  # entry nodes always purple
                result["entry_nodes"].append(node_record)
            elif node_type == "RISK":
                result["risk_nodes"].append(node_record)
            elif node_type == "ACTION":
                result["action_nodes"].append(node_record)
            elif node_type == "ORGANIZATION":
                result["resource_nodes"].append(node_record)
            else:
                result["concept_nodes"].append(node_record)

            # Traverse neighbors
            if depth < max_depth:
                neighbors = sorted(
                    self.G.neighbors(node),
                    key=lambda n: self.G.degree(n),
                    reverse=True,
                )
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))

                        # Record edge for visualization
                        edge_data = self.G.edges.get((node, neighbor), {})
                        result["edges"].append({
                            "source": node,
                            "target": neighbor,
                            "description": edge_data.get("description", "")[:200],
                            "weight": float(edge_data.get("weight", 1)),
                        })

        return result

    def traverse_from_profile(
        self,
        q1: str,
        q2: str,
        q3: str,
        q4: str,
        q5: list[str],
        q6: str | None = None,
    ) -> dict:
        """
        Main query: map Q1-Q5 answers to entry nodes, traverse graph.

        Args:
            q1: "yes" | "no" | "have_but_dont_use"
            q2: "salary" | "gig" | "cash" | "irregular"
            q3: "none" | "credit_card" | "student" | "medical" | "multiple"
            q4: "nothing" | "under_500" | "500_to_1000" | "1000_to_5000" | "over_5000"
            q5: list of "renters" | "health" | "auto" | "life" | "none"
            q6: optional free text query

        Returns:
            Dict with entry_nodes, risk_nodes, action_nodes, resource_nodes,
            edges, and profile metadata.
        """
        # Collect keywords from all answers
        keywords = []
        keywords.extend(Q1_MAP.get(q1, []))
        keywords.extend(Q2_MAP.get(q2, []))
        keywords.extend(Q3_MAP.get(q3, []))
        keywords.extend(Q4_MAP.get(q4, []))
        for ins in q5:
            keywords.extend(Q5_MAP.get(ins, []))

        # Find entry nodes
        entry_nodes = self._find_entry_nodes(keywords)

        # If Q6 provided, add keyword-matched nodes
        if q6 and q6.strip():
            q6_nodes = self.search(q6, top_k=5)
            entry_nodes.extend([n["name"] for n in q6_nodes])
            entry_nodes = list(dict.fromkeys(entry_nodes))

        # Traverse
        traversal = self._bfs_traverse(entry_nodes, max_depth=2, max_nodes=80)

        # Build profile metadata
        profile = {
            "has_bank_account": q1,
            "income_type": q2,
            "debt_type": q3,
            "savings_level": q4,
            "insurance_types": q5,
            "free_text": q6,
        }

        # Determine persona
        persona = self._determine_persona(q1, q2, q3, q4, q5)

        # Determine protection gaps (for State Farm card placement)
        protection_gaps = self._find_protection_gaps(q5)

        return {
            "profile": profile,
            "persona": persona,
            "protection_gaps": protection_gaps,
            "entry_nodes": traversal["entry_nodes"],
            "risk_nodes": traversal["risk_nodes"],
            "action_nodes": traversal["action_nodes"],
            "resource_nodes": traversal["resource_nodes"],
            "concept_nodes": traversal["concept_nodes"],
            "all_nodes": traversal["all_nodes"],
            "edges": traversal["edges"],
            "communities_touched": list(set(
                n["community_id"] for n in traversal["all_nodes"]
                if n["community_id"] is not None
            )),
        }

    def _determine_persona(self, q1, q2, q3, q4, q5) -> str:
        """Determine user persona from answers."""
        if q1 == "no":
            return "unbanked"
        if q2 == "gig":
            return "gig_worker"
        if q2 == "cash":
            return "cash_income"
        if q3 == "student":
            return "student"
        if q4 == "nothing":
            return "zero_savings"
        if "none" in q5:
            return "uninsured"
        return "general"

    def _find_protection_gaps(self, q5: list[str]) -> list[str]:
        """Find insurance types the user is missing (for State Farm card)."""
        all_types = {"renters", "health", "auto", "life"}
        has = set(q5) - {"none"}
        return sorted(all_types - has)

    def _embed_query(self, text: str) -> np.ndarray | None:
        """Embed a query string using Gemini embedding API."""
        for i, key in enumerate(EMBEDDING_API_KEYS):
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key={key}",
                    json={
                        "model": f"models/{EMBEDDING_MODEL}",
                        "content": {"parts": [{"text": text[:2000]}]},
                    },
                    timeout=10,
                )
                if resp.status_code == 200:
                    vec = np.array(resp.json()["embedding"]["values"], dtype=np.float32)
                    return vec / (np.linalg.norm(vec) + 1e-10)
                # Rate limited — try next key
            except Exception:
                continue
        return None

    def semantic_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Semantic search using embeddings. Falls back to keyword search."""
        if not self.has_embeddings:
            return self.keyword_search(query, top_k)

        query_vec = self._embed_query(query)
        if query_vec is None:
            return self.keyword_search(query, top_k)

        # Cosine similarity against all entity embeddings
        scores = self.embedding_vectors_normed @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            name = self.embedding_names[idx]
            entity = self.entities.get(name, {})
            results.append({
                "name": name,
                "type": entity.get("type", "UNKNOWN"),
                "description": entity.get("description", "")[:300],
                "score": float(scores[idx]),
                "community_id": self.entity_to_community.get(name),
            })
        return results

    def keyword_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Keyword search across entity names and descriptions."""
        query_words = set(re.findall(r'\w+', query.upper()))
        scored = []

        for name, entity in self.entities.items():
            name_words = set(re.findall(r'\w+', name))
            desc_words = set(re.findall(r'\w+', entity.get("description", "").upper()))

            name_overlap = len(query_words & name_words)
            desc_overlap = len(query_words & desc_words)
            score = name_overlap * 3 + desc_overlap

            if score > 0:
                scored.append((score, name, entity))

        scored.sort(key=lambda x: (-x[0], -x[2].get("count", 0)))

        results = []
        for score, name, entity in scored[:top_k]:
            results.append({
                "name": name,
                "type": entity.get("type", "UNKNOWN"),
                "description": entity.get("description", "")[:300],
                "score": score,
                "community_id": self.entity_to_community.get(name),
            })
        return results

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search entities — uses semantic search if embeddings available, else keyword."""
        return self.semantic_search(query, top_k)

    def get_community_info(self, community_id: int) -> dict | None:
        """Get information about a community."""
        for comm in self.communities:
            if comm["id"] == community_id:
                result = {
                    "id": comm["id"],
                    "size": comm["size"],
                    "leaders": comm["leaders"],
                    "dominant_types": comm["dominant_types"],
                }
                # Add report if available
                if self.community_reports and community_id < len(self.community_reports):
                    result["report"] = self.community_reports[community_id].get("summary", "")
                return result
        return None

    def get_node_neighborhood(self, node_name: str, depth: int = 1) -> dict:
        """Get a node's immediate neighborhood (for click-to-expand)."""
        if node_name not in self.G.nodes:
            return {"error": f"Node '{node_name}' not found"}

        return self._bfs_traverse([node_name], max_depth=depth, max_nodes=30)

    def get_graph_for_visualization(self, nodes: list[dict], edges: list[dict]) -> dict:
        """Format graph data for frontend visualization library (e.g., D3, vis.js)."""
        return {
            "nodes": [
                {
                    "id": n["name"],
                    "label": n["name"][:30],
                    "group": n["type"],
                    "color": n["color"],
                    "size": max(5, min(30, n["degree"])),
                    "title": n["description"][:200],
                    "community": n.get("community_id"),
                }
                for n in nodes
            ],
            "edges": [
                {
                    "from": e["source"],
                    "to": e["target"],
                    "title": e["description"][:100],
                    "value": e["weight"],
                }
                for e in edges
            ],
        }


# ─── CLI Test ────────────────────────────────────────────────────────────────

def main():
    """Test with sample personas from the project brief."""
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    gq = GraphQuery()

    print("\n" + "=" * 60)
    print("  TEST 1: Unbanked Gig Worker with Credit Card Debt")
    print("=" * 60)
    result = gq.traverse_from_profile(
        q1="no", q2="gig", q3="credit_card", q4="nothing", q5=["none"]
    )
    print(f"  Persona: {result['persona']}")
    print(f"  Protection gaps: {result['protection_gaps']}")
    print(f"  Entry nodes ({len(result['entry_nodes'])}):")
    for n in result["entry_nodes"]:
        print(f"    [{n['type']}] {n['name']}")
    print(f"  Risk nodes ({len(result['risk_nodes'])}):")
    for n in result["risk_nodes"][:5]:
        print(f"    [{n['type']}] {n['name']}: {n['description'][:60]}")
    print(f"  Action nodes ({len(result['action_nodes'])}):")
    for n in result["action_nodes"][:5]:
        print(f"    [{n['type']}] {n['name']}: {n['description'][:60]}")
    print(f"  Resource nodes ({len(result['resource_nodes'])}):")
    for n in result["resource_nodes"][:5]:
        print(f"    [{n['type']}] {n['name']}")
    print(f"  Communities touched: {result['communities_touched']}")
    print(f"  Total nodes: {len(result['all_nodes'])}, Edges: {len(result['edges'])}")

    print("\n" + "=" * 60)
    print("  TEST 2: Student with Student Loans, Has Health Insurance")
    print("=" * 60)
    result2 = gq.traverse_from_profile(
        q1="yes", q2="salary", q3="student", q4="under_500", q5=["health"]
    )
    print(f"  Persona: {result2['persona']}")
    print(f"  Protection gaps: {result2['protection_gaps']}")
    print(f"  Entry nodes: {[n['name'] for n in result2['entry_nodes']]}")
    print(f"  Action nodes ({len(result2['action_nodes'])}):")
    for n in result2["action_nodes"][:5]:
        print(f"    {n['name']}: {n['description'][:60]}")
    print(f"  Communities: {result2['communities_touched']}")

    print("\n" + "=" * 60)
    print("  TEST 3: Q6 Semantic Search — 'landlord threatening eviction'")
    print("=" * 60)
    results = gq.search("landlord threatening eviction", top_k=8)
    for r in results:
        print(f"  [{r['type']}] {r['name']} (score: {r['score']:.3f})")
        print(f"    {r['description'][:80]}")

    print("\n" + "=" * 60)
    print("  TEST 3b: Semantic Search — 'I just moved to America'")
    print("=" * 60)
    results = gq.search("I just moved to America and need help with money", top_k=8)
    for r in results:
        print(f"  [{r['type']}] {r['name']} (score: {r['score']:.3f})")
        print(f"    {r['description'][:80]}")

    print("\n" + "=" * 60)
    print("  TEST 4: Visualization Format")
    print("=" * 60)
    viz = gq.get_graph_for_visualization(
        result["all_nodes"][:20], result["edges"][:30]
    )
    print(f"  Vis.js format: {len(viz['nodes'])} nodes, {len(viz['edges'])} edges")
    print(f"  Sample node: {json.dumps(viz['nodes'][0], indent=2)}")


if __name__ == "__main__":
    main()
