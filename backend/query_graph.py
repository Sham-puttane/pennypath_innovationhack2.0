#!/usr/bin/env python3
"""
First Dollar — Knowledge Graph Query API
Maps user intake answers (Q1-Q5) to graph entry nodes,
traverses relationships, and returns structured results
for the FOO engine and frontend visualization.

Personas are determined by which entity clusters the traversal touches,
not by hardcoded if/else logic. Per the brief: "Personas are discovered
from the corpus, not invented."

Usage:
    from query_graph import GraphQuery
    gq = GraphQuery()
    result = gq.traverse_from_profile(q1="no", q2="gig", q3="credit_card", q4="nothing", q5=["none"])
"""

import json
import re
import os
from pathlib import Path
from collections import defaultdict

import networkx as nx
import numpy as np
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
# These map to the brief's table: Q answer -> GraphRAG entry node

Q1_MAP = {
    # Q1: Do you have a bank account?
    "yes": ["CHECKING ACCOUNT", "SAVINGS ACCOUNT", "BANK ACCOUNT"],
    "no": ["PREPAID CARD", "MONEY ORDER"],                           # -> [Unbanked] entry node
    "have_but_dont_use": ["CHECKING ACCOUNT", "BANK ACCOUNT", "FEES"],
}

Q2_MAP = {
    # Q2: How do you get paid?
    "salary": ["INCOME", "DIRECT DEPOSIT", "PAYCHECK"],              # -> [Salaried income]
    "gig": ["INCOME", "IRREGULAR INCOME", "SELF-EMPLOYMENT"],        # -> [Irregular income]
    "cash": ["INCOME", "CASH", "MONEY ORDER"],                       # -> [Irregular income]
    "irregular": ["INCOME", "IRREGULAR INCOME", "PUBLIC BENEFITS"],  # -> [Irregular income]
}

Q3_MAP = {
    # Q3: Do you have debt? What kind?
    "none": [],                                                       # -> [No debt]
    "credit_card": ["CREDIT CARD DEBT", "CREDIT CARD", "DEBT"],      # -> [Credit card debt]
    "student": ["STUDENT LOANS", "FEDERAL STUDENT LOANS", "DEBT"],   # -> [Student loan debt]
    "medical": ["MEDICAL DEBT", "DEBT", "DEBT COLLECTOR"],
    "multiple": ["DEBT", "CREDIT CARD DEBT", "STUDENT LOANS", "MEDICAL DEBT", "DEBT COLLECTOR"],
}

Q4_MAP = {
    # Q4: How much do you have saved?
    "nothing": ["SAVING", "EMERGENCY FUND", "SAVINGS ACCOUNT"],
    "under_500": ["SAVING", "EMERGENCY FUND", "SAVINGS ACCOUNT"],
    "500_to_1000": ["SAVINGS ACCOUNT", "SAVING", "EMERGENCY SAVINGS FUND"],
    "1000_to_5000": ["SAVINGS ACCOUNT", "SAVINGS PLAN", "EMERGENCY FUND"],
    "over_5000": ["SAVINGS ACCOUNT", "SAVINGS PLAN"],
}

# Q5 is multi-select (checkboxes)
Q5_MAP = {
    # Q5: Do you have insurance?
    "renters": ["RENTERS INSURANCE", "STATE FARM"],
    "health": ["HEALTH INSURANCE"],
    "auto": ["AUTO INSURANCE", "CAR INSURANCE", "STATE FARM"],
    "life": ["LIFE INSURANCE", "STATE FARM"],
    "none": ["INSURANCE", "RENTERS INSURANCE", "AUTO INSURANCE", "LIFE INSURANCE", "STATE FARM"],
}

# ─── Graph node colors (per brief section 2.7) ─────────────────────────────
# Purple nodes: entry nodes from user's answers
# Red nodes: risk chains discovered by traversal
# Green nodes: action nodes (each maps to a step in the queue)
# Amber nodes: resource nodes (free tools, websites, hotlines)

TYPE_TO_COLOR = {
    "FINANCIAL_CONCEPT": "info",
    "ACTION": "green",
    "FINANCIAL_PRODUCT": "purple",
    "ORGANIZATION": "amber",
    "RISK": "red",
    "GEO": "gray",
    "PERSON": "gray",
    "EVENT": "amber",
}

FOO_CATEGORIES = {
    "ACTION": "action",
    "RISK": "risk",
    "FINANCIAL_PRODUCT": "product",
    "ORGANIZATION": "resource",
    "FINANCIAL_CONCEPT": "concept",
    "EVENT": "event",
}

# ─── Persona detection: entity clusters from the corpus ─────────────────────
# Per brief: "Every persona you offer the user must map to an entity cluster
# that actually exists in the indexed knowledge graph."
#
# These are the persona-defining entities extracted from the 6 data sources:

PERSONA_ENTITY_CLUSTERS = {
    "international_student": [
        "INTERNATIONAL STUDENTS", "NEWCOMER'S GUIDES TO MANAGING MONEY",
        "IMMIGRANTS", "INDIVIDUAL TAXPAYER IDENTIFICATION NUMBER (ITIN)",
    ],
    "recent_immigrant": [
        "IMMIGRANTS", "NEWCOMER'S GUIDES TO MANAGING MONEY",
        "FOREIGN DRIVERS", "INDIVIDUAL TAXPAYER IDENTIFICATION NUMBER (ITIN)",
        "REMITTANCE TRANSFER PROVIDERS",
    ],
    "gig_worker": [
        "IRREGULAR INCOME", "SELF-EMPLOYMENT", "CASH",
    ],
    "single_parent": [
        "FAMILY MEMBER", "FAMILY", "PUBLIC BENEFITS",
        "LOW INCOME HOME ENERGY ASSISTANCE PROGRAM (LIHEAP)",
    ],
    "first_gen_college_student": [
        "STUDENT LOANS", "FEDERAL STUDENT LOANS", "BORROWER",
        "STUDENT LOAN DEBT",
    ],
    "unbanked": [
        "PREPAID CARD", "MONEY ORDER", "CHECK CASHING STORE",
    ],
    "veteran": [
        "VETERAN", "ACTIVE DUTY SERVICEMEMBERS", "VETERANS' BENEFITS",
        "MILITARY LENDING ACT (MLA)", "SERVICEMEMBERS",
    ],
    "reentry": [
        "FOCUS ON REENTRY COMPANION GUIDE", "FOCUS ON REENTRY", "REENTRY",
    ],
}


class GraphQuery:
    """Query interface for the First Dollar knowledge graph."""

    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self._load_data()

    def _load_data(self):
        """Load graph, entities, relationships, and communities."""
        self.G = nx.read_graphml(str(self.output_dir / "knowledge_graph.graphml"))

        entities_raw = json.loads(
            (self.output_dir / "entities.json").read_text(encoding="utf-8")
        )
        self.entities = {e["name"]: e for e in entities_raw}

        self.communities = json.loads(
            (self.output_dir / "communities.json").read_text(encoding="utf-8")
        )

        self.entity_to_community = {}
        for comm in self.communities:
            for member in comm.get("members", []):
                self.entity_to_community[member] = comm["id"]

        reports_path = self.output_dir / "community_reports.json"
        if reports_path.exists():
            self.community_reports = json.loads(
                reports_path.read_text(encoding="utf-8")
            )
        else:
            self.community_reports = None

        npz_path = self.output_dir / "entity_embeddings.npz"
        if npz_path.exists():
            data = np.load(str(npz_path), allow_pickle=True)
            self.embedding_names = list(data["names"])
            self.embedding_vectors = data["vectors"]
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
            if kw_upper in self.G.nodes:
                found.append(kw_upper)
            else:
                for node in self.G.nodes:
                    if kw_upper in node:
                        found.append(node)
                        break
        return list(dict.fromkeys(found))

    def _bfs_traverse(self, entry_nodes: list[str], max_depth: int = 2, max_nodes: int = 80) -> dict:
        """BFS from entry nodes, collecting typed nodes by depth."""
        visited = set()
        result = {
            "entry_nodes": [], "risk_nodes": [], "action_nodes": [],
            "resource_nodes": [], "concept_nodes": [], "all_nodes": [], "edges": [],
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
                "name": node, "type": node_type,
                "color": TYPE_TO_COLOR.get(node_type, "gray"),
                "description": entity_info.get("description", node_data.get("description", ""))[:300],
                "community_id": self.entity_to_community.get(node),
                "depth": depth, "degree": self.G.degree(node),
                "foo_category": FOO_CATEGORIES.get(node_type, "concept"),
            }

            result["all_nodes"].append(node_record)

            if depth == 0:
                node_record["color"] = "purple"
                result["entry_nodes"].append(node_record)
            elif node_type == "RISK":
                result["risk_nodes"].append(node_record)
            elif node_type == "ACTION":
                result["action_nodes"].append(node_record)
            elif node_type == "ORGANIZATION":
                result["resource_nodes"].append(node_record)
            else:
                result["concept_nodes"].append(node_record)

            if depth < max_depth:
                neighbors = sorted(self.G.neighbors(node), key=lambda n: self.G.degree(n), reverse=True)
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
                        edge_data = self.G.edges.get((node, neighbor), {})
                        result["edges"].append({
                            "source": node, "target": neighbor,
                            "description": edge_data.get("description", "")[:200],
                            "weight": float(edge_data.get("weight", 1)),
                        })

        return result

    def _determine_persona(self, traversal: dict, q1: str, q2: str, q3: str) -> str:
        """
        Determine persona by checking which corpus entity clusters
        the traversal touched. Per brief: personas are discovered from
        the corpus, not invented.
        """
        # Use only entry nodes (depth 0) for persona detection
        # Deeper traversal nodes cause false matches (e.g., CASH reached from INCOME)
        touched_names = {n["name"] for n in traversal["all_nodes"] if n["depth"] == 0}

        # Score each persona by how many of its defining entities were touched
        scores = {}
        for persona, cluster_entities in PERSONA_ENTITY_CLUSTERS.items():
            overlap = sum(1 for e in cluster_entities if e in touched_names)
            if overlap > 0:
                scores[persona] = overlap

        # Also boost based on Q1-Q3 answers that strongly signal a persona
        if q1 == "no" and "unbanked" in scores:
            scores["unbanked"] = scores.get("unbanked", 0) + 2
        if q2 in ("gig", "cash") and "gig_worker" in scores:
            scores["gig_worker"] = scores.get("gig_worker", 0) + 2
        if q3 == "student" and "first_gen_college_student" in scores:
            scores["first_gen_college_student"] = scores.get("first_gen_college_student", 0) + 2

        # Fallback: if no cluster matched, infer from Q1-Q3
        if not scores:
            if q1 == "no":
                return "unbanked"
            if q2 in ("gig", "cash"):
                return "gig_worker"
            if q3 == "student":
                return "first_gen_college_student"
            return "general"

        # Return highest-scoring persona
        return max(scores, key=scores.get)

    def traverse_from_profile(self, q1, q2, q3, q4, q5, q6=None, persona_tags=None) -> dict:
        """Main query: map Q1-Q5 answers + persona tags to entry nodes, traverse graph."""
        keywords = []
        keywords.extend(Q1_MAP.get(q1, []))
        keywords.extend(Q2_MAP.get(q2, []))
        keywords.extend(Q3_MAP.get(q3, []))
        keywords.extend(Q4_MAP.get(q4, []))
        for ins in q5:
            keywords.extend(Q5_MAP.get(ins, []))

        # Add persona-specific entity clusters to entry keywords
        if persona_tags:
            for tag in persona_tags:
                cluster = PERSONA_ENTITY_CLUSTERS.get(tag, [])
                keywords.extend(cluster)

        entry_nodes = self._find_entry_nodes(keywords)

        if q6 and q6.strip():
            q6_nodes = self.search(q6, top_k=5)
            entry_nodes.extend([n["name"] for n in q6_nodes])
            entry_nodes = list(dict.fromkeys(entry_nodes))

        traversal = self._bfs_traverse(entry_nodes, max_depth=2, max_nodes=80)

        profile = {
            "has_bank_account": q1,
            "income_type": q2,
            "debt_type": q3,
            "savings_level": q4,
            "insurance_types": q5,
            "free_text": q6,
        }

        # Use user-selected personas if provided, else detect from graph
        if persona_tags and len(persona_tags) > 0:
            persona = persona_tags[0]  # primary persona for FOO (doesn't affect ordering)
        else:
            persona = self._determine_persona(traversal, q1, q2, q3)

        protection_gaps = self._find_protection_gaps(q5)

        return {
            "profile": profile,
            "persona": persona,
            "persona_tags": persona_tags or [persona],
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

    def _find_protection_gaps(self, q5):
        all_types = {"renters", "health", "auto", "life"}
        has = set(q5) - {"none"}
        return sorted(all_types - has)

    # ─── Search methods ──────────────────────────────────────────────────

    def _embed_query(self, text):
        for i, key in enumerate(EMBEDDING_API_KEYS):
            try:
                resp = requests.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{EMBEDDING_MODEL}:embedContent?key={key}",
                    json={"model": f"models/{EMBEDDING_MODEL}", "content": {"parts": [{"text": text[:2000]}]}},
                    timeout=10,
                )
                if resp.status_code == 200:
                    vec = np.array(resp.json()["embedding"]["values"], dtype=np.float32)
                    return vec / (np.linalg.norm(vec) + 1e-10)
            except Exception:
                continue
        return None

    def semantic_search(self, query, top_k=10):
        if not self.has_embeddings:
            return self.keyword_search(query, top_k)
        query_vec = self._embed_query(query)
        if query_vec is None:
            return self.keyword_search(query, top_k)
        scores = self.embedding_vectors_normed @ query_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            name = self.embedding_names[idx]
            entity = self.entities.get(name, {})
            results.append({
                "name": name, "type": entity.get("type", "UNKNOWN"),
                "description": entity.get("description", "")[:300],
                "score": float(scores[idx]),
                "community_id": self.entity_to_community.get(name),
            })
        return results

    def keyword_search(self, query, top_k=10):
        query_words = set(re.findall(r'\w+', query.upper()))
        scored = []
        for name, entity in self.entities.items():
            name_words = set(re.findall(r'\w+', name))
            desc_words = set(re.findall(r'\w+', entity.get("description", "").upper()))
            score = len(query_words & name_words) * 3 + len(query_words & desc_words)
            if score > 0:
                scored.append((score, name, entity))
        scored.sort(key=lambda x: (-x[0], -x[2].get("count", 0)))
        return [{"name": n, "type": e.get("type", "UNKNOWN"), "description": e.get("description", "")[:300],
                 "score": s, "community_id": self.entity_to_community.get(n)} for s, n, e in scored[:top_k]]

    def search(self, query, top_k=10):
        return self.semantic_search(query, top_k)

    def get_community_info(self, community_id):
        for comm in self.communities:
            if comm["id"] == community_id:
                result = {"id": comm["id"], "size": comm["size"], "leaders": comm["leaders"],
                          "dominant_types": comm["dominant_types"]}
                if self.community_reports and community_id < len(self.community_reports):
                    result["report"] = self.community_reports[community_id].get("summary", "")
                return result
        return None

    def get_node_neighborhood(self, node_name, depth=1):
        if node_name not in self.G.nodes:
            return {"error": f"Node '{node_name}' not found"}
        return self._bfs_traverse([node_name], max_depth=depth, max_nodes=30)

    def get_graph_for_visualization(self, nodes, edges):
        return {
            "nodes": [{"id": n["name"], "label": n["name"][:30], "group": n["type"],
                       "color": n["color"], "size": max(5, min(30, n["degree"])),
                       "title": n["description"][:200], "community": n.get("community_id")} for n in nodes],
            "edges": [{"from": e["source"], "to": e["target"],
                       "title": e["description"][:100], "value": e["weight"]} for e in edges],
        }


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    gq = GraphQuery()

    tests = [
        ("Unbanked gig worker, CC debt", {"q1": "no", "q2": "gig", "q3": "credit_card", "q4": "nothing", "q5": ["none"]}),
        ("Student, student loans, health ins", {"q1": "yes", "q2": "salary", "q3": "student", "q4": "under_500", "q5": ["health"]}),
        ("Immigrant, cash, no debt", {"q1": "no", "q2": "cash", "q3": "none", "q4": "nothing", "q5": ["none"], "q6": "I just moved to America and need help with money"}),
    ]
    for name, args in tests:
        result = gq.traverse_from_profile(**args)
        print(f"\n{name}: persona={result['persona']}, gaps={result['protection_gaps']}, "
              f"nodes={len(result['all_nodes'])}, communities={result['communities_touched']}")
