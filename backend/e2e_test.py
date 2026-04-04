#!/usr/bin/env python3
"""
First Dollar — End-to-End Test as a Real User
Simulates the full pipeline from intake form to final output.
No hardcoded answers — tests the actual system.
"""

import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from query_graph import GraphQuery
from foo_engine import order_actions
from personalize import personalize_steps


def run_user_session(q1, q2, q3, q4, q5, q6=None, label="User"):
    """Simulate a full user session through the pipeline."""
    print(f"\n{'='*70}")
    print(f"  USER SESSION: {label}")
    print(f"{'='*70}")

    # ── STEP 1: User fills intake form ───────────────────────���────────
    print(f"\n  [INTAKE FORM]")
    print(f"  Q1 (Bank account?):     {q1}")
    print(f"  Q2 (How paid?):         {q2}")
    print(f"  Q3 (Debt?):             {q3}")
    print(f"  Q4 (Savings?):          {q4}")
    print(f"  Q5 (Insurance?):        {q5}")
    print(f"  Q6 (Biggest fear?):     {q6 or '(skipped)'}")

    # ── STEP 2: Graph traversal ──────────────────────────────────��────
    print(f"\n  [GRAPH TRAVERSAL]")
    t0 = time.time()
    result = gq.traverse_from_profile(q1=q1, q2=q2, q3=q3, q4=q4, q5=q5, q6=q6)
    t1 = time.time()

    print(f"  Persona detected:       {result['persona']}")
    print(f"  Protection gaps:        {result['protection_gaps']}")
    print(f"  Entry nodes found:      {len(result['entry_nodes'])}")
    print(f"  Action nodes found:     {len(result['action_nodes'])}")
    print(f"  Risk nodes found:       {len(result['risk_nodes'])}")
    print(f"  Resource nodes found:   {len(result['resource_nodes'])}")
    print(f"  Communities touched:    {result['communities_touched']}")
    print(f"  Total graph nodes:      {len(result['all_nodes'])}")
    print(f"  Total graph edges:      {len(result['edges'])}")
    print(f"  Time: {(t1-t0)*1000:.0f}ms")

    # Verify: entry nodes should actually exist in the graph
    for n in result["entry_nodes"]:
        if n["name"] not in gq.G.nodes:
            print(f"  WARNING: Entry node '{n['name']}' NOT in graph!")

    # ── STEP 3: FOO ordering ──────────────────────────────────────────
    print(f"\n  [FOO ENGINE — Priority Queue]")
    t2 = time.time()
    steps = order_actions(result)
    t3 = time.time()

    for s in steps:
        sf = " ** STATE FARM **" if s["is_state_farm"] else ""
        print(f"  Step {s['step_number']:2d}. [{s['tier']:15s}] {s['action']}{sf}")
        # Verify: source_nodes should exist in graph
        for node_name in s["source_nodes"]:
            if node_name not in gq.G.nodes:
                print(f"        WARNING: source node '{node_name}' NOT in graph!")

    print(f"  Total steps: {len(steps)}")
    print(f"  Time: {(t3-t2)*1000:.0f}ms")

    # ── STEP 4: Gemini personalization ────────────────────────────────
    print(f"\n  [GEMINI PERSONALIZATION]")
    t4 = time.time()
    personalized = personalize_steps(steps, result["persona"], result["profile"], q6)
    t5 = time.time()

    # Check if personalization actually changed the text
    changed = sum(
        1 for s in personalized
        if s.get("personalized_action") != s["action"]
    )
    unchanged = len(personalized) - changed

    print(f"  Personalized: {changed}/{len(personalized)} steps rewritten")
    if unchanged > 0:
        print(f"  Unchanged: {unchanged} steps (LLM may have returned same text)")

    for s in personalized[:5]:
        sf = " [SF]" if s["is_state_farm"] else ""
        orig = s["action"]
        pers = s.get("personalized_action", orig)
        if pers != orig:
            print(f"\n  Step {s['step_number']}.{sf}")
            print(f"    BEFORE: {orig}")
            print(f"    AFTER:  {pers}")
            print(f"    REASON: {s.get('personalized_reason', s['reason'])[:100]}")

    print(f"  Time: {(t5-t4)*1000:.0f}ms")

    # ── STEP 5: Visualization data ─────────────────────────────────��──
    print(f"\n  [VISUALIZATION OUTPUT]")
    viz = gq.get_graph_for_visualization(result["all_nodes"], result["edges"])
    colors = {}
    for n in viz["nodes"]:
        colors[n["color"]] = colors.get(n["color"], 0) + 1

    print(f"  Vis.js nodes: {len(viz['nodes'])}")
    print(f"  Vis.js edges: {len(viz['edges'])}")
    print(f"  Node colors: {colors}")

    # ── STEP 6: Community context ─────────────────────────────────────
    print(f"\n  [COMMUNITY CONTEXT]")
    for cid in result["communities_touched"][:5]:
        info = gq.get_community_info(cid)
        if info and info.get("report"):
            print(f"  Community {cid}: {info['report'][:100]}...")

    # ── Summary ───────────────────────────────────────────────────────
    total_time = (t5 - t0) * 1000
    print(f"\n  [TOTAL PIPELINE TIME: {total_time:.0f}ms]")

    return personalized


# ═════════════════════════════════════════════��═════════════════════════════════
# MAIN — Run real user scenarios
# ═══════════════════════════════════════════════════════════════════════════════

print("Loading system...")
gq = GraphQuery()

# ── Test 1: A real user who has a bank account, earns salary, has CC debt,
#            some savings, and only has auto insurance.
run_user_session(
    q1="yes", q2="salary", q3="credit_card", q4="500_to_1000",
    q5=["auto"],
    q6=None,
    label="Real user: salaried, CC debt, $500-1k saved, has auto only"
)

# ── Test 2: Someone with irregular income, multiple debts, no bank account,
#            and they wrote something in Q6.
run_user_session(
    q1="no", q2="irregular", q3="multiple", q4="nothing",
    q5=["none"],
    q6="my car broke down and I can't get to work",
    label="Real user: irregular income, multiple debts, car broke down"
)

# ── Test 3: Someone who has everything mostly together but wants to optimize.
run_user_session(
    q1="yes", q2="salary", q3="none", q4="over_5000",
    q5=["health", "auto", "renters"],
    q6="I want to start investing but don't know where to begin",
    label="Real user: stable, $5k+ saved, wants to invest"
)

# ── Test 4: Edge case — Q6 with something the graph might NOT have.
run_user_session(
    q1="yes", q2="gig", q3="none", q4="under_500",
    q5=["none"],
    q6="crypto losses and NFT scams",
    label="Edge case: Q6 outside corpus (crypto/NFT)"
)
