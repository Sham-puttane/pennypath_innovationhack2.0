#!/usr/bin/env python3
"""
First Dollar — Backend API Server

Chains: query_graph -> foo_engine -> personalize -> JSON response

Endpoints:
    POST /api/query          — Full pipeline (intake form -> personalized priority queue + graph)
    POST /api/search         — Semantic search (Q6 free text)
    GET  /api/community/<id> — Community info with report
    GET  /api/node/<name>    — Node neighborhood (click-to-expand)
    GET  /api/health         — Health check
    GET  /api/personas       — Pre-built demo personas
"""

import sys
import os
import json
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load .env file for API keys
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


# ─── Pre-built demo personas ────────────────────────────────────────────────

DEMO_PERSONAS = {
    "maria": {
        "name": "Maria",
        "description": "Unbanked gig worker with credit card debt, no insurance, no savings",
        "args": {
            "q1": "no", "q2": "gig", "q3": "credit_card",
            "q4": "nothing", "q5": ["none"], "q6": None,
        },
    },
    "james": {
        "name": "James",
        "description": "College student with student loans, has health insurance, minimal savings",
        "args": {
            "q1": "yes", "q2": "salary", "q3": "student",
            "q4": "under_500", "q5": ["health"], "q6": None,
        },
    },
    "aisha": {
        "name": "Aisha",
        "description": "Recent immigrant, cash income, no debt, no savings, no insurance",
        "args": {
            "q1": "no", "q2": "cash", "q3": "none",
            "q4": "nothing", "q5": ["none"],
            "q6": "I just moved to America and need help with money",
        },
    },
}


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "graph_nodes": gq.G.number_of_nodes(),
        "graph_edges": gq.G.number_of_edges(),
        "communities": len(gq.communities),
        "has_embeddings": gq.has_embeddings,
    })


@app.route("/api/personas")
def personas():
    """Return available demo personas."""
    return jsonify({
        name: {"name": p["name"], "description": p["description"], "args": p["args"]}
        for name, p in DEMO_PERSONAS.items()
    })


@app.route("/api/query", methods=["POST"])
def query():
    """
    Full pipeline: intake form -> graph traversal -> FOO ordering -> personalization.

    Request body:
    {
        "q1": "no",
        "q2": "gig",
        "q3": "credit_card",
        "q4": "nothing",
        "q5": ["none"],
        "q6": "optional free text"   // optional
    }

    OR use a demo persona:
    {
        "persona": "maria"   // or "james" or "aisha"
    }
    """
    data = request.json

    # Check if using a demo persona
    if "persona" in data and data["persona"] in DEMO_PERSONAS:
        persona_data = DEMO_PERSONAS[data["persona"]]
        args = persona_data["args"].copy()
    else:
        args = {
            "q1": data.get("q1", "yes"),
            "q2": data.get("q2", "salary"),
            "q3": data.get("q3", "none"),
            "q4": data.get("q4", "nothing"),
            "q5": data.get("q5", ["none"]),
            "q6": data.get("q6"),
        }

    # Step 1: Graph traversal
    traversal = gq.traverse_from_profile(**args)

    # Step 2: FOO ordering
    steps = order_actions(traversal)

    # Step 3: Gemini personalization (optional — skip if "skip_personalize" flag)
    if not data.get("skip_personalize"):
        steps = personalize_steps(
            steps,
            persona=traversal["persona"],
            profile=traversal["profile"],
            q6_text=args.get("q6"),
        )

    # Step 4: Build visualization data
    viz = gq.get_graph_for_visualization(
        traversal["all_nodes"], traversal["edges"]
    )

    # Step 5: Get community reports for touched communities
    community_info = []
    for cid in traversal["communities_touched"]:
        info = gq.get_community_info(cid)
        if info:
            community_info.append(info)

    return jsonify({
        "profile": traversal["profile"],
        "persona": traversal["persona"],
        "protection_gaps": traversal["protection_gaps"],
        "steps": steps,
        "graph": viz,
        "communities": community_info,
        "stats": {
            "total_nodes": len(traversal["all_nodes"]),
            "total_edges": len(traversal["edges"]),
            "entry_nodes": len(traversal["entry_nodes"]),
            "action_nodes": len(traversal["action_nodes"]),
            "risk_nodes": len(traversal["risk_nodes"]),
            "resource_nodes": len(traversal["resource_nodes"]),
            "communities_touched": len(traversal["communities_touched"]),
            "total_steps": len(steps),
        },
    })


@app.route("/api/search", methods=["POST"])
def search():
    """Semantic search for Q6 free text."""
    data = request.json
    query_text = data.get("query", "")
    top_k = data.get("top_k", 10)
    results = gq.search(query_text, top_k=top_k)
    return jsonify({"query": query_text, "results": results})


@app.route("/api/community/<int:cid>")
def community(cid):
    """Get community info with report."""
    info = gq.get_community_info(cid)
    if info:
        return jsonify(info)
    return jsonify({"error": f"Community {cid} not found"}), 404


@app.route("/api/node/<path:name>")
def node(name):
    """Get node neighborhood (click-to-expand on graph)."""
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
    print("    POST /api/query          — Full pipeline")
    print("    POST /api/search         — Semantic search")
    print("    GET  /api/community/<id> — Community info")
    print("    GET  /api/node/<name>    — Node neighborhood")
    print("    GET  /api/health         — Health check")
    print("    GET  /api/personas       — Demo personas\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
