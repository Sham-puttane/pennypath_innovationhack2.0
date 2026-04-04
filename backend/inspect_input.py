#!/usr/bin/env python3
"""Inspect query_graph.py output for the 3 demo personas."""

import json
import sys
sys.stdout.reconfigure(encoding="utf-8")

from query_graph import GraphQuery
gq = GraphQuery()

PERSONAS = [
    {
        "name": "Maria — Unbanked gig worker, credit card debt, no insurance",
        "args": {"q1": "no", "q2": "gig", "q3": "credit_card", "q4": "nothing", "q5": ["none"]},
    },
    {
        "name": "James — Student, student loans, has health insurance",
        "args": {"q1": "yes", "q2": "salary", "q3": "student", "q4": "under_500", "q5": ["health"]},
    },
    {
        "name": "Aisha — Recent immigrant, cash income, no debt, no savings",
        "args": {"q1": "no", "q2": "cash", "q3": "none", "q4": "nothing", "q5": ["none"],
                 "q6": "I just moved to America and need help with money"},
    },
]

for p in PERSONAS:
    print("\n" + "=" * 70)
    print(f"  {p['name']}")
    print("=" * 70)

    result = gq.traverse_from_profile(**p["args"])

    print(f"\n  persona: {result['persona']}")
    print(f"  protection_gaps: {result['protection_gaps']}")
    print(f"  communities_touched: {result['communities_touched']}")

    print(f"\n  --- ENTRY NODES ({len(result['entry_nodes'])}) ---")
    for n in result["entry_nodes"]:
        print(f"  [{n['foo_category']:10}] {n['name']}")

    print(f"\n  --- ACTION NODES ({len(result['action_nodes'])}) ---")
    for n in result["action_nodes"]:
        print(f"  {n['name']}")
        print(f"    desc: {n['description'][:100]}")
        print(f"    community: {n['community_id']}, degree: {n['degree']}")

    print(f"\n  --- RISK NODES ({len(result['risk_nodes'])}) ---")
    for n in result["risk_nodes"]:
        print(f"  {n['name']}")
        print(f"    desc: {n['description'][:100]}")

    print(f"\n  --- RESOURCE NODES ({len(result['resource_nodes'])}) ---")
    for n in result["resource_nodes"][:10]:
        print(f"  {n['name']}: {n['description'][:80]}")

    print(f"\n  --- CONCEPT NODES ({len(result['concept_nodes'])}) (first 10) ---")
    for n in result["concept_nodes"][:10]:
        print(f"  [{n['type']:25}] {n['name']}")

    print(f"\n  TOTALS: {len(result['all_nodes'])} nodes, {len(result['edges'])} edges")

    # Save full JSON for reference
    with open(f"output/persona_{result['persona']}_full.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Full output saved to: output/persona_{result['persona']}_full.json")
