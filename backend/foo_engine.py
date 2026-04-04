#!/usr/bin/env python3
"""
First Dollar — Financial Order of Operations (FOO) Rule Engine

Deterministic ordering of financial action steps based on mathematical
priority. Every ordering decision has a factual justification — not AI guessing.

Key principle: The order is MATH, not opinion.
  - 28% APR credit card debt compounds daily → pay before saving at 0.5% APY
  - $0 in emergency fund → one flat tire triggers debt spiral → fund before investing
  - No bank account → can't receive direct deposit → open account before anything

Usage:
    from foo_engine import order_actions
    steps = order_actions(traversal_result)
"""

from __future__ import annotations


# ─── FOO Priority Tiers ─────────────────────────────────────────────────────
# Lower number = higher priority. Each tier has a mathematical justification.
#
# Tier 1 (1-3):   ACCESS — Can't do anything without money access
# Tier 2 (4-6):   STOP BLEEDING — High-interest debt costs more than savings earn
# Tier 3 (7-9):   SAFETY NET — One emergency without a cushion triggers debt spiral
# Tier 4 (10-13): PROTECTION — Insurance is cheaper than the risk it covers
# Tier 5 (14-16): BUILD — Credit and long-term saving
# Tier 6 (17+):   GROW — Investing, advanced planning

TIER_ACCESS = 1
TIER_STOP_BLEEDING = 4
TIER_SAFETY_NET = 7
TIER_PROTECTION = 10
TIER_BUILD = 14
TIER_GROW = 17


def order_actions(result: dict) -> list[dict]:
    """
    Takes a traversal result from query_graph.py and returns an ordered
    priority queue of financial action steps.

    Args:
        result: Output of GraphQuery.traverse_from_profile()

    Returns:
        List of step dicts, ordered by FOO priority:
        [{
            "step_number": 1,
            "action": "Open a bank or prepaid card account",
            "reason": "...",
            "tier": "access",
            "priority": 1,
            "source_nodes": [...],
            "risk_nodes": [...],
            "resource_nodes": [...],
            "is_state_farm": False,
        }, ...]
    """
    profile = result["profile"]
    persona = result["persona"]
    protection_gaps = result["protection_gaps"]
    action_nodes = result["action_nodes"]
    risk_nodes = result["risk_nodes"]
    resource_nodes = result["resource_nodes"]
    entry_nodes = result["entry_nodes"]
    concept_nodes = result.get("concept_nodes", [])

    steps = []

    # ── TIER 1: ACCESS (priority 1-3) ────────────────────────────────────
    # Can't do anything without a place to put money

    if profile["has_bank_account"] == "no":
        steps.append(_step(
            priority=1,
            action="Open a bank account or get a prepaid card",
            reason="You need a safe place to keep your money. Without an account, "
                   "you lose money to check-cashing fees (1-5% per check) and "
                   "can't receive direct deposit.",
            tier="access",
            source_nodes=_match_nodes(action_nodes + entry_nodes,
                ["ACCOUNT", "PREPAID", "CHECKING", "BANK"]),
            resource_nodes=_match_nodes(resource_nodes,
                ["BANK", "CREDIT UNION"]),
        ))
    elif profile["has_bank_account"] == "have_but_dont_use":
        steps.append(_step(
            priority=1,
            action="Start using your bank account regularly",
            reason="You already have an account — using it saves you check-cashing fees "
                   "and builds a financial track record.",
            tier="access",
            source_nodes=_match_nodes(action_nodes,
                ["ACCOUNT", "CHECKING", "BALANCE"]),
        ))

    if profile["income_type"] in ("gig", "cash", "irregular"):
        steps.append(_step(
            priority=2,
            action="Set up a system to track your income",
            reason="With irregular income, knowing exactly what's coming in each "
                   "month is the foundation for every other financial decision.",
            tier="access",
            source_nodes=_match_nodes(action_nodes + entry_nodes,
                ["INCOME", "TRACK", "DEPOSIT"]),
        ))

    # Direct deposit setup (even for salary workers)
    if profile["has_bank_account"] != "no":
        steps.append(_step(
            priority=3,
            action="Set up direct deposit for your paycheck",
            reason="Direct deposit is faster, free, and eliminates check-cashing fees. "
                   "Many banks waive monthly fees with direct deposit.",
            tier="access",
            source_nodes=_match_nodes(action_nodes,
                ["DIRECT DEPOSIT", "PAYCHECK"]),
        ))

    # ── TIER 2: STOP BLEEDING (priority 4-6) ────────────────────────────
    # High-interest debt costs more per day than savings earn per month

    if profile["debt_type"] == "credit_card":
        steps.append(_step(
            priority=4,
            action="Pay down credit card debt aggressively",
            reason="Credit card APR (15-28%) compounds daily. Every dollar you "
                   "put toward this debt 'earns' 15-28% — no savings account "
                   "comes close. Pay minimums on everything else, maximums here.",
            tier="stop_bleeding",
            source_nodes=_match_nodes(action_nodes + entry_nodes,
                ["DEBT", "CREDIT CARD", "PAY"]),
            risk_nodes=_match_nodes(risk_nodes, ["DEBT", "CREDIT"]),
        ))
    elif profile["debt_type"] == "multiple":
        steps.append(_step(
            priority=4,
            action="List all debts and attack the highest-interest one first",
            reason="The avalanche method: pay minimums on all debts, then throw "
                   "every extra dollar at the highest APR. Saves the most money "
                   "mathematically.",
            tier="stop_bleeding",
            source_nodes=_match_nodes(action_nodes + entry_nodes,
                ["DEBT", "PAY", "COLLECTOR"]),
            risk_nodes=_match_nodes(risk_nodes, ["DEBT"]),
        ))
    elif profile["debt_type"] == "medical":
        steps.append(_step(
            priority=5,
            action="Negotiate medical debt and check for assistance programs",
            reason="Medical debt often has 0% interest and hospitals frequently "
                   "offer payment plans or charity care. Always negotiate before paying.",
            tier="stop_bleeding",
            source_nodes=_match_nodes(action_nodes + entry_nodes,
                ["MEDICAL", "DEBT", "NEGOTIATE"]),
        ))
    elif profile["debt_type"] == "student":
        # Student loans are lower priority — lower interest, tax deductible
        steps.append(_step(
            priority=13,
            action="Set up an income-driven repayment plan for student loans",
            reason="Federal student loans have lower rates (5-7%) and offer "
                   "income-driven plans that cap payments at 10-15% of income. "
                   "Don't overpay these before building emergency savings.",
            tier="build",
            source_nodes=_match_nodes(action_nodes + entry_nodes,
                ["STUDENT LOAN", "FEDERAL", "REPAYMENT"]),
        ))

    # Debt collector protection
    if profile["debt_type"] in ("credit_card", "medical", "multiple"):
        steps.append(_step(
            priority=6,
            action="Know your rights with debt collectors",
            reason="Debt collectors cannot threaten you, call before 8am or after 9pm, "
                   "or contact your employer. You can request debt validation in writing.",
            tier="stop_bleeding",
            source_nodes=_match_nodes(action_nodes + resource_nodes,
                ["COLLECTOR", "RIGHTS", "CFPB"]),
            risk_nodes=_match_nodes(risk_nodes, ["COLLECTOR", "THREAT"]),
        ))

    # ── TIER 3: SAFETY NET (priority 7-9) ────────────────────────────────
    # Without a cushion, one emergency restarts the debt cycle

    if profile["savings_level"] in ("nothing", "under_500"):
        steps.append(_step(
            priority=7,
            action="Build a $500 emergency fund",
            reason="$500 covers most common emergencies (car repair, urgent bill, "
                   "medical copay). Without it, one surprise expense forces you "
                   "into high-interest debt. Save this BEFORE paying extra on low-interest debt.",
            tier="safety_net",
            source_nodes=_match_nodes(action_nodes + entry_nodes,
                ["SAVING", "EMERGENCY", "FUND"]),
        ))

    if profile["savings_level"] in ("nothing", "under_500", "500_to_1000"):
        steps.append(_step(
            priority=9,
            action="Grow your emergency fund to one month's expenses",
            reason="After the first $500, keep building toward one month of expenses. "
                   "This prevents a job loss or illness from becoming a financial catastrophe.",
            tier="safety_net",
            source_nodes=_match_nodes(entry_nodes + concept_nodes,
                ["SAVING", "EMERGENCY", "MONTH"]),
        ))

    # Bill tracking
    steps.append(_step(
        priority=8,
        action="Track all your bills and due dates",
        reason="Late fees ($25-40 each) add up fast and can damage credit. "
               "A simple list of what's due when prevents this entirely.",
        tier="safety_net",
        source_nodes=_match_nodes(action_nodes,
            ["BILL", "PAY", "TRACK"]),
        risk_nodes=_match_nodes(risk_nodes, ["LATE", "FEE", "OVERDRAFT"]),
    ))

    # ── TIER 4: PROTECTION — State Farm cards inserted here ──────────────
    # Insurance is mathematically correct when: premium < (risk probability x cost)
    # These steps are where State Farm recommendations land

    for gap in protection_gaps:
        priority, action_text, reason_text = _insurance_step(gap, persona)
        steps.append(_step(
            priority=priority,
            action=action_text,
            reason=reason_text,
            tier="protection",
            source_nodes=_match_nodes(entry_nodes + action_nodes,
                [gap.upper(), "INSURANCE", "STATE FARM"]),
            resource_nodes=_match_nodes(resource_nodes + entry_nodes,
                ["STATE FARM"]),
            is_state_farm=True,
        ))

    # ── TIER 5: BUILD (priority 14-16) ───────────────────────────────────

    steps.append(_step(
        priority=14,
        action="Check your credit report for free",
        reason="Your credit report affects loan rates, apartment applications, and "
               "even job offers. Check for errors at annualcreditreport.com — it's free.",
        tier="build",
        source_nodes=_match_nodes(entry_nodes + concept_nodes,
            ["CREDIT REPORT", "CREDIT SCORE", "CREDIT HISTORY"]),
    ))

    if profile["savings_level"] in ("nothing", "under_500", "500_to_1000", "1000_to_5000"):
        steps.append(_step(
            priority=15,
            action="Start building credit with a secured card or credit-builder loan",
            reason="Good credit saves thousands on future loans and apartments. "
                   "A secured card (backed by your deposit) is the safest way to start.",
            tier="build",
            source_nodes=_match_nodes(concept_nodes,
                ["CREDIT", "SECURED", "BUILD"]),
        ))

    # ── TIER 6: GROW (priority 17+) ──────────────────────────────────────

    steps.append(_step(
        priority=17,
        action="Set a specific savings goal and automate it",
        reason="Automatic transfers remove willpower from the equation. "
               "Even $10/week builds to $520/year.",
        tier="grow",
        source_nodes=_match_nodes(action_nodes + entry_nodes,
            ["SAVING", "GOAL", "AUTOMATIC"]),
    ))

    # Protect against scams
    steps.append(_step(
        priority=16,
        action="Learn to recognize common financial scams",
        reason="Scammers target people who are new to the financial system. "
               "Never share your SSN, account numbers, or PINs over the phone.",
        tier="build",
        source_nodes=_match_nodes(risk_nodes + concept_nodes,
            ["SCAM", "FRAUD", "IDENTITY THEFT", "PERSONAL INFORMATION"]),
        risk_nodes=_match_nodes(risk_nodes, ["SCAM", "THEFT", "FRAUD"]),
    ))

    # ── Sort, deduplicate, and number ────────────────────────────────────

    # Remove duplicate priorities (keep first added)
    seen_priorities = set()
    unique_steps = []
    for s in steps:
        if s["priority"] not in seen_priorities:
            seen_priorities.add(s["priority"])
            unique_steps.append(s)

    # Sort by priority
    unique_steps.sort(key=lambda s: s["priority"])

    # Renumber 1, 2, 3...
    for i, s in enumerate(unique_steps):
        s["step_number"] = i + 1

    return unique_steps


# ─── Helper Functions ────────────────────────────────────────────────────────

def _step(
    priority: int,
    action: str,
    reason: str,
    tier: str,
    source_nodes: list[dict] | None = None,
    risk_nodes: list[dict] | None = None,
    resource_nodes: list[dict] | None = None,
    is_state_farm: bool = False,
) -> dict:
    """Create a step dict."""
    return {
        "step_number": 0,  # filled in at the end
        "action": action,
        "reason": reason,
        "tier": tier,
        "priority": priority,
        "source_nodes": [n["name"] for n in (source_nodes or [])],
        "risk_nodes": [n["name"] for n in (risk_nodes or [])],
        "resource_nodes": [n["name"] for n in (resource_nodes or [])],
        "is_state_farm": is_state_farm,
    }


def _match_nodes(nodes: list[dict], keywords: list[str]) -> list[dict]:
    """Filter nodes whose names contain any of the keywords."""
    matched = []
    for node in nodes:
        name = node["name"].upper()
        if any(kw.upper() in name for kw in keywords):
            matched.append(node)
    return matched


def _insurance_step(gap: str, persona: str) -> tuple[int, str, str]:
    """Return (priority, action, reason) for an insurance protection gap."""
    if gap == "renters":
        return (
            10,
            "Get renters insurance to protect your belongings",
            "Renters insurance costs $15-30/month and covers theft, fire, and "
            "liability. Without it, one break-in or kitchen fire could cost you "
            "everything. Many landlords require it anyway.",
        )
    elif gap == "auto":
        return (
            11,
            "Get proper auto insurance coverage",
            "Driving without adequate insurance risks a financial catastrophe. "
            "One at-fault accident without coverage could mean tens of thousands "
            "in liability. State minimums are often not enough.",
        )
    elif gap == "life":
        return (
            12,
            "Consider life insurance if others depend on your income",
            "If anyone relies on your income (children, spouse, parents), "
            "term life insurance is affordable ($20-40/month for $250k coverage) "
            "and protects them if something happens to you.",
        )
    elif gap == "health":
        return (
            10,
            "Get health insurance coverage",
            "One ER visit without insurance averages $2,200. Check healthcare.gov "
            "for subsidized plans, or your state's Medicaid program if your income "
            "qualifies. This is the #1 bankruptcy prevention step.",
        )
    else:
        return (
            12,
            f"Review your {gap} insurance needs",
            f"Consider whether {gap} insurance would protect you from financial risk.",
        )


# ─── CLI Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from query_graph import GraphQuery
    gq = GraphQuery()

    personas = [
        ("Maria — Unbanked gig worker, CC debt",
         {"q1": "no", "q2": "gig", "q3": "credit_card", "q4": "nothing", "q5": ["none"]}),
        ("James — Student, student loans, has health",
         {"q1": "yes", "q2": "salary", "q3": "student", "q4": "under_500", "q5": ["health"]}),
        ("Aisha — Immigrant, cash, no debt, no savings",
         {"q1": "no", "q2": "cash", "q3": "none", "q4": "nothing", "q5": ["none"],
          "q6": "I just moved to America and need help with money"}),
    ]

    for name, args in personas:
        result = gq.traverse_from_profile(**args)
        steps = order_actions(result)

        print("\n" + "=" * 70)
        print(f"  {name}")
        print(f"  Persona: {result['persona']} | Gaps: {result['protection_gaps']}")
        print("=" * 70)

        for s in steps:
            sf = " [STATE FARM]" if s["is_state_farm"] else ""
            print(f"\n  Step {s['step_number']}. [{s['tier'].upper():15}]{sf}")
            print(f"  {s['action']}")
            print(f"  Why: {s['reason'][:100]}...")
            if s["source_nodes"]:
                print(f"  Graph nodes: {', '.join(s['source_nodes'][:3])}")

        print(f"\n  Total steps: {len(steps)}")
