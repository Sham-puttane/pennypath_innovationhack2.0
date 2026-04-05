#!/usr/bin/env python3
"""
First Dollar — Backend API Server

Chains: query_graph -> foo_engine -> personalize -> JSON response

Endpoints:
    POST /api/query            — Full pipeline (intake -> personalized queue + graph)
    POST /api/whatif            — What-If slider (re-runs FOO only, no GraphRAG, ~50ms)
    POST /api/search           — Semantic search (Q6 free text)
    GET  /api/community/<id>   — Community info with report
    GET  /api/node/<name>      — Node neighborhood (click-to-expand)
    GET  /api/health           — Health check
    GET  /api/personas         — Pre-built demo personas
"""

import sys
import os
import json
import copy
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from query_graph import GraphQuery
from foo_engine import order_actions
from personalize import personalize_steps

# ─── Initialize ──────────────────────────────────────────────────────────────

app = Flask(__name__)
CORS(app)

print("Loading knowledge graph...", file=sys.stderr)
gq = GraphQuery()
print("Server ready.", file=sys.stderr)

# Session cache: stores last traversal so What-If doesn't re-run GraphRAG
_session_cache = {}


# ─── Pre-built demo personas ────────────────────────────────────────────────

DEMO_PERSONAS = {
    "maria": {
        "name": "Maria",
        "description": "Unbanked gig worker with credit card debt, no insurance, no savings",
        "args": {"q1": "no", "q2": "gig", "q3": "credit_card", "q4": "nothing", "q5": ["none"], "q6": None},
    },
    "james": {
        "name": "James",
        "description": "First-gen college student with student loans, has health insurance, minimal savings",
        "args": {"q1": "yes", "q2": "salary", "q3": "student", "q4": "under_500", "q5": ["health"], "q6": None},
    },
    "aisha": {
        "name": "Aisha",
        "description": "Recent immigrant, cash income, no debt, no savings, no insurance",
        "args": {"q1": "no", "q2": "cash", "q3": "none", "q4": "nothing", "q5": ["none"],
                 "q6": "I just moved to America and need help with money"},
    },
}

# ─── What-If Scenarios (per brief section 2.5) ──────────────────────────────
# Each scenario changes one profile variable. FOO re-runs, GraphRAG does NOT.

WHATIF_SCENARIOS = {
    "lose_job": {
        "label": "What if I lose my job?",
        "changes": {"income_type": "irregular", "savings_level": "nothing"},
    },
    "earn_more": {
        "label": "What if I earn $200 more per month?",
        "changes": {},  # income amount doesn't change FOO logic, but savings threshold may
    },
    "pay_off_debt": {
        "label": "What if I pay off my debt?",
        "changes": {"debt_type": "none"},
    },
    "get_renters_insurance": {
        "label": "What if I get renters insurance?",
        "changes": {"_add_insurance": "renters"},
    },
    "open_bank_account": {
        "label": "What if I open a bank account?",
        "changes": {"has_bank_account": "yes"},
    },
    "get_steady_job": {
        "label": "What if I get a steady job?",
        "changes": {"income_type": "salary"},
    },
}


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "graph_nodes": gq.G.number_of_nodes(),
        "graph_edges": gq.G.number_of_edges(),
        "communities": len(gq.communities),
        "has_embeddings": gq.has_embeddings,
    })


@app.route("/api/personas")
def personas():
    return jsonify({
        name: {"name": p["name"], "description": p["description"], "args": p["args"]}
        for name, p in DEMO_PERSONAS.items()
    })


def build_reasoning_path(traversal, steps):
    """Build a structured hop-by-hop reasoning path from graph traversal + FOO steps.
    Each query produces a unique path based on real graph edges."""
    entry_nodes = []
    reasoning_path = []
    hop = 0

    # Category mapping from node type
    cat_map = {
        "FINANCIAL_CONCEPT": "entry", "FINANCIAL_PRODUCT": "entry",
        "ACTION": "action", "RISK": "risk",
        "ORGANIZATION": "resource", "EVENT": "resource",
        "PERSON": "entry", "GEO": "entry",
    }

    # Build entry nodes from the first traversal layer
    for n in traversal.get("entry_nodes", []):
        ntype = n.get("type", "").upper().strip('"')
        entry_nodes.append({
            "id": n["name"], "label": n["name"],
            "category": cat_map.get(ntype, "entry"), "hop": 0,
        })

    # Build reasoning path from FOO steps
    prev_nodes = [n["name"] for n in traversal.get("entry_nodes", [])[:3]]
    seen = set(prev_nodes)

    for s in steps:
        step_nodes = s.get("source_nodes", [])
        risk_nodes = s.get("risk_nodes", [])

        # Find new nodes in this step (not yet seen)
        new_nodes = [n for n in step_nodes + risk_nodes if n not in seen]
        if not new_nodes and step_nodes:
            new_nodes = step_nodes[:1]  # At least show one node per step

        for target in new_nodes:
            # Find a source node from prev_nodes that connects via an edge
            source = prev_nodes[0] if prev_nodes else None

            # Look for actual graph edge between any prev node and target
            for e in traversal.get("edges", []):
                if (e["source"] in seen and e["target"] == target) or \
                   (e["target"] in seen and e["source"] == target):
                    source = e["source"] if e["target"] == target else e["target"]
                    break

            # Determine category
            cat = "action"
            if target in risk_nodes:
                cat = "risk"
            else:
                # Check from traversal node data
                for n in traversal.get("all_nodes", []):
                    if n["name"] == target:
                        ntype = n.get("type", "").upper().strip('"')
                        cat = cat_map.get(ntype, "action")
                        break

            # Determine edge label from step tier
            tier = s.get("tier", "")
            edge_labels = {
                "access": "enables",
                "stop_bleeding": "reduces_risk",
                "safety_net": "protects_against",
                "protection": "insures_against",
                "build": "builds_toward",
                "grow": "grows_into",
            }
            edge_label = edge_labels.get(tier, "leads_to")

            hop += 1
            reasoning_path.append({
                "hop": hop,
                "from": source,
                "to": target,
                "edge_label": edge_label,
                "node_id": target,
                "node_label": target,
                "node_category": cat,
                "step_number": s.get("step_number"),
                "step_action": s.get("action", ""),
            })
            seen.add(target)

        if new_nodes:
            prev_nodes = new_nodes

    # Build focused subgraph (only traversed nodes + edges)
    path_node_ids = set(n["id"] for n in entry_nodes)
    path_node_ids.update(r["node_id"] for r in reasoning_path)
    path_node_ids.update(r["from"] for r in reasoning_path if r["from"])

    focused_nodes = [n for n in traversal.get("all_nodes", []) if n["name"] in path_node_ids]
    focused_edges = [e for e in traversal.get("edges", [])
                     if e["source"] in path_node_ids and e["target"] in path_node_ids]

    return {
        "entry_nodes": entry_nodes,
        "reasoning_path": reasoning_path,
        "focused_subgraph": {
            "nodes": [{"id": n["name"], "label": n["name"][:25], "category": cat_map.get(n.get("type","").upper().strip('"'), "entry"),
                       "color": n.get("color","gray")} for n in focused_nodes],
            "links": [{"from": e["source"], "to": e["target"], "label": e.get("description","")[:40]}
                      for e in focused_edges],
        },
    }


@app.route("/api/query", methods=["POST"])
def query():
    """Full pipeline: intake -> graph traversal -> FOO -> personalization."""
    data = request.json

    if "persona" in data and data["persona"] in DEMO_PERSONAS:
        args = DEMO_PERSONAS[data["persona"]]["args"].copy()
    else:
        args = {
            "q1": data.get("q1", "yes"), "q2": data.get("q2", "salary"),
            "q3": data.get("q3", "none"), "q4": data.get("q4", "nothing"),
            "q5": data.get("q5", ["none"]), "q6": data.get("q6"),
            "persona_tags": data.get("persona_tags"),
        }

    # Step 1: Graph traversal (this is the only GraphRAG call)
    traversal = gq.traverse_from_profile(**args)

    # Cache traversal for What-If slider (per brief: "full context block stored in memory")
    session_id = data.get("session_id", "default")
    _session_cache[session_id] = {
        "traversal": traversal,
        "args": args,
    }

    # Step 2: FOO ordering (pass assets for insurance filtering)
    assets = data.get("assets", [])
    traversal["assets"] = assets
    steps = order_actions(traversal)

    # Step 3: Gemini personalization
    if not data.get("skip_personalize"):
        steps = personalize_steps(
            steps, persona=traversal["persona"],
            profile=traversal["profile"], q6_text=args.get("q6"),
        )

    # Step 4: Build response
    viz = gq.get_graph_for_visualization(traversal["all_nodes"], traversal["edges"])
    community_info = [gq.get_community_info(cid) for cid in traversal["communities_touched"]
                      if gq.get_community_info(cid)]

    # Step 5: Build structured reasoning path
    rpath = build_reasoning_path(traversal, steps)

    return jsonify({
        "session_id": session_id,
        "profile": traversal["profile"],
        "persona": traversal["persona"],
        "persona_tags": traversal.get("persona_tags", [traversal["persona"]]),
        "protection_gaps": traversal["protection_gaps"],
        "steps": steps,
        "graph": viz,
        "reasoning": rpath,
        "communities": community_info,
        "whatif_scenarios": {k: v["label"] for k, v in WHATIF_SCENARIOS.items()},
        "stats": {
            "total_nodes": len(traversal["all_nodes"]),
            "total_edges": len(traversal["edges"]),
            "total_steps": len(steps),
            "communities_touched": len(traversal["communities_touched"]),
        },
    })


@app.route("/api/whatif", methods=["POST"])
def whatif():
    """
    What-If slider: re-runs FOO only with one profile variable changed.
    Per brief: "GraphRAG runs exactly once. The slider only re-executes
    the JavaScript FOO rule engine with one profile variable changed.
    This takes ~50 milliseconds."

    Request: {"session_id": "...", "scenario": "lose_job"}
    """
    data = request.json
    session_id = data.get("session_id", "default")
    scenario_key = data.get("scenario")

    if session_id not in _session_cache:
        return jsonify({"error": "No active session. Call /api/query first."}), 400

    if scenario_key not in WHATIF_SCENARIOS:
        return jsonify({"error": f"Unknown scenario: {scenario_key}",
                        "available": list(WHATIF_SCENARIOS.keys())}), 400

    scenario = WHATIF_SCENARIOS[scenario_key]
    cached = _session_cache[session_id]
    traversal = cached["traversal"]

    # Deep copy the traversal and modify only the profile
    modified = copy.deepcopy(traversal)
    changes = scenario["changes"]

    # Apply profile changes
    for key, value in changes.items():
        if key == "_add_insurance":
            # Special: add insurance type to q5
            if value not in modified["profile"]["insurance_types"]:
                modified["profile"]["insurance_types"].append(value)
            if "none" in modified["profile"]["insurance_types"]:
                modified["profile"]["insurance_types"].remove("none")
            # Recalculate protection gaps
            all_types = {"renters", "health", "auto", "life"}
            has = set(modified["profile"]["insurance_types"]) - {"none"}
            modified["protection_gaps"] = sorted(all_types - has)
        elif key in modified["profile"]:
            modified["profile"][key] = value

    # Recalculate protection gaps if insurance changed
    if "has_bank_account" in changes or "debt_type" in changes or "income_type" in changes:
        pass  # protection_gaps don't change for these

    # Re-run FOO only (no GraphRAG, no Gemini) — this is ~50ms
    steps = order_actions(modified)

    return jsonify({
        "scenario": scenario_key,
        "scenario_label": scenario["label"],
        "profile": modified["profile"],
        "persona": modified["persona"],
        "protection_gaps": modified["protection_gaps"],
        "steps": steps,
        "stats": {"total_steps": len(steps)},
    })


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    results = gq.search(data.get("query", ""), top_k=data.get("top_k", 10))
    return jsonify({"query": data.get("query", ""), "results": results})


@app.route("/api/community/<int:cid>")
def community(cid):
    info = gq.get_community_info(cid)
    if info:
        return jsonify(info)
    return jsonify({"error": f"Community {cid} not found"}), 404


@app.route("/api/node/<path:name>")
def node(name):
    depth = request.args.get("depth", 1, type=int)
    result = gq.get_node_neighborhood(name.upper(), depth=min(depth, 3))
    if "error" in result:
        return jsonify(result), 404
    viz = gq.get_graph_for_visualization(result["all_nodes"], result["edges"])
    return jsonify(viz)


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  First Dollar API Server")
    print("  http://localhost:5000")
    print("  Endpoints:")
    print("    POST /api/query            — Full pipeline")
    print("    POST /api/whatif           — What-If slider (FOO only, ~50ms)")
    print("    POST /api/search           — Semantic search")
    print("    GET  /api/community/<id>   — Community info")
    print("    GET  /api/node/<name>      — Node neighborhood")
    print("    GET  /api/health           — Health check")
    print("    GET  /api/personas         — Demo personas\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
