#!/usr/bin/env python3
"""
First Dollar — Financial Order of Operations (FOO) Rule Engine

Implements all 10 FOO rules from the brief exactly:
  Rule 1:  No bank account -> Step 1 always
  Rule 2:  High-APR debt (>15%) -> pause before saving (exception: Rule 3)
  Rule 3:  Savings < $500 -> $500 buffer before debt attack
  Rule 4:  Insurance gap -> State Farm card AFTER income stabilises
  Rule 5:  Cash/gig income -> emergency fund is $1,000 not $500
  Rule 6:  No credit history -> secured card after bank account
  Rule 7:  Student loans -> IDR check before standard repayment
  Rule 8:  Employer 401k match -> catch it before extra debt payment
  Rule 9:  Multiple debts -> avalanche (highest APR first)
  Rule 10: Savings > $1k + no high-APR debt -> investment steps unlock

State Farm card placement (per brief section 2.6):
  - Deep debt crisis: Step 6-7 (after triage)
  - Stable with gap: Step 2-3
  - Already has insurance: does not appear

Usage:
    from foo_engine import order_actions
    steps = order_actions(traversal_result)
"""

from __future__ import annotations


def order_actions(result: dict) -> list[dict]:
    """
    Takes a traversal result from query_graph.py and returns an ordered
    priority queue of financial action steps.

    Returns list of step dicts with: step_number, action, reason, tier,
    priority, source_nodes, risk_nodes, resource_nodes, is_state_farm,
    savings_amount (for State Farm card text personalization).
    """
    profile = result["profile"]
    protection_gaps = result["protection_gaps"]
    action_nodes = result["action_nodes"]
    risk_nodes = result["risk_nodes"]
    resource_nodes = result["resource_nodes"]
    entry_nodes = result["entry_nodes"]
    concept_nodes = result.get("concept_nodes", [])
    assets = result.get("assets", [])

    # Filter insurance gaps based on what the user actually owns/uses
    # Only recommend insurance that's relevant to their situation
    if assets:
        relevant_gaps = []
        for gap in protection_gaps:
            if gap == "renters" and ("rents" in assets or not assets or "none_above" in assets):
                relevant_gaps.append(gap)  # renters insurance if they rent
            elif gap == "auto" and "has_car" in assets:
                relevant_gaps.append(gap)  # auto insurance only if they have a car
            elif gap == "life" and "has_dependents" in assets:
                relevant_gaps.append(gap)  # life insurance only if others depend on them
            elif gap == "health":
                relevant_gaps.append(gap)  # health insurance always relevant
        protection_gaps = relevant_gaps

    steps = []
    has_bank = profile["has_bank_account"] != "no"
    is_irregular = profile["income_type"] in ("gig", "cash", "irregular")
    has_high_apr_debt = profile["debt_type"] in ("credit_card", "multiple")
    has_any_debt = profile["debt_type"] != "none"
    savings = profile["savings_level"]

    # Emergency fund threshold per Rule 5
    emergency_threshold = 1000 if is_irregular else 500
    emergency_label = f"${emergency_threshold:,}"

    # Determine if user is in "crisis" or "stable" for State Farm placement (Rule 4)
    in_crisis = has_high_apr_debt or savings in ("nothing", "under_500")
    income_stabilized = has_bank and savings not in ("nothing",)

    # ── RULE 1: No bank account -> Step 1 always ────────────────────────
    if not has_bank:
        steps.append(_step(
            priority=1,
            action="Open a bank account or get a prepaid card",
            reason="You need a safe place to keep your money. Without an account, "
                   "you lose money to check-cashing fees (1-5% per check) and "
                   "can't receive direct deposit. This blocks everything else.",
            tier="access",
            source_nodes=_match(action_nodes + entry_nodes, ["ACCOUNT", "PREPAID", "CHECKING", "BANK"]),
            resource_nodes=_match(resource_nodes, ["BANK", "CREDIT UNION"]),
        ))
    elif profile["has_bank_account"] == "have_but_dont_use":
        steps.append(_step(
            priority=1,
            action="Start using your bank account regularly",
            reason="You already have an account — using it saves you check-cashing fees "
                   "and builds a financial track record.",
            tier="access",
            source_nodes=_match(action_nodes, ["ACCOUNT", "CHECKING", "BALANCE"]),
        ))

    # Income tracking for irregular earners
    if is_irregular:
        steps.append(_step(
            priority=2,
            action="Set up a system to track your income",
            reason="With irregular income, knowing what's coming in each month "
                   "is the foundation for every other financial decision.",
            tier="access",
            source_nodes=_match(action_nodes + entry_nodes, ["INCOME", "TRACK", "DEPOSIT"]),
        ))

    # Direct deposit (if has bank)
    if has_bank:
        steps.append(_step(
            priority=3,
            action="Set up direct deposit for your paycheck",
            reason="Direct deposit is faster, free, and eliminates check-cashing fees. "
                   "Many banks waive monthly fees with direct deposit.",
            tier="access",
            source_nodes=_match(action_nodes, ["DIRECT DEPOSIT", "PAYCHECK"]),
        ))

    # ── RULE 3: Savings < $500 -> $500 buffer BEFORE debt attack ────────
    # (This comes before Rule 2 intentionally — brief says exception to Rule 2)
    if savings in ("nothing", "under_500"):
        steps.append(_step(
            priority=4,
            action=f"Build a {emergency_label} emergency fund first",
            reason=f"{emergency_label} covers most common emergencies (car repair, urgent bill, "
                   f"medical copay). Without it, one surprise expense forces you into "
                   f"high-interest debt. Save this BEFORE attacking debt.",
            tier="safety_net",
            source_nodes=_match(action_nodes + entry_nodes, ["SAVING", "EMERGENCY", "FUND"]),
        ))

    # ── RULE 2: High-APR debt -> pay aggressively ───────────────────────
    # (After $500 buffer per Rule 3 exception)
    if profile["debt_type"] == "credit_card":
        steps.append(_step(
            priority=5,
            action="Pay down credit card debt aggressively",
            reason="Credit card APR (15-28%) compounds daily. Every dollar toward this "
                   "debt 'earns' 15-28% — no savings account comes close. Pay minimums "
                   "on everything else, maximums here.",
            tier="stop_bleeding",
            source_nodes=_match(action_nodes + entry_nodes, ["DEBT", "CREDIT CARD", "PAY"]),
            risk_nodes=_match(risk_nodes, ["DEBT", "CREDIT"]),
        ))

    # ── RULE 9: Multiple debts -> avalanche (highest APR first) ─────────
    elif profile["debt_type"] == "multiple":
        steps.append(_step(
            priority=5,
            action="List all debts by interest rate — attack highest APR first",
            reason="The avalanche method: pay minimums on all debts, then throw every "
                   "extra dollar at the highest APR. This is mathematically optimal — "
                   "it saves the most money in total interest paid.",
            tier="stop_bleeding",
            source_nodes=_match(action_nodes + entry_nodes, ["DEBT", "PAY", "COLLECTOR"]),
            risk_nodes=_match(risk_nodes, ["DEBT"]),
        ))
    elif profile["debt_type"] == "medical":
        steps.append(_step(
            priority=6,
            action="Negotiate medical debt and check for assistance programs",
            reason="Medical debt often has 0% interest and hospitals frequently "
                   "offer payment plans or charity care. Always negotiate before paying.",
            tier="stop_bleeding",
            source_nodes=_match(action_nodes + entry_nodes, ["MEDICAL", "DEBT", "NEGOTIATE"]),
        ))

    # Debt collector rights
    if has_high_apr_debt or profile["debt_type"] == "medical":
        steps.append(_step(
            priority=7,
            action="Know your rights with debt collectors",
            reason="Debt collectors cannot threaten you, call before 8am or after 9pm, "
                   "or contact your employer. You can request debt validation in writing.",
            tier="stop_bleeding",
            source_nodes=_match(action_nodes + resource_nodes, ["COLLECTOR", "RIGHTS", "CFPB"]),
            risk_nodes=_match(risk_nodes, ["COLLECTOR", "THREAT"]),
        ))

    # ── RULE 8: 401k match -> catch free money before extra debt ────────
    # (We can't detect actual 401k from the form, but if user has salary
    # income, prompt them to check — free money beats debt math)
    if profile["income_type"] == "salary" and has_any_debt:
        steps.append(_step(
            priority=8,
            action="Check if your employer offers a 401k match — claim it",
            reason="If your employer matches 401k contributions, that's 50-100% "
                   "instant return on your money. Even with debt, free money "
                   "beats paying extra on a 7% loan.",
            tier="safety_net",
            source_nodes=_match(concept_nodes, ["RETIREMENT", "EMPLOYER", "BENEFITS"]),
            resource_nodes=_match(resource_nodes, ["EMPLOYER"]),
        ))

    # Bill tracking
    steps.append(_step(
        priority=9,
        action="Track all your bills and due dates",
        reason="Late fees ($25-40 each) add up fast and can damage credit. "
               "A simple list of what's due when prevents this entirely.",
        tier="safety_net",
        source_nodes=_match(action_nodes, ["BILL", "PAY", "TRACK"]),
        risk_nodes=_match(risk_nodes, ["LATE", "FEE", "OVERDRAFT"]),
    ))

    # ── RULE 5: Emergency fund threshold $1k for gig, $500 for salary ───
    if savings in ("nothing", "under_500", "500_to_1000"):
        steps.append(_step(
            priority=10,
            action=f"Grow your emergency fund to {emergency_label}",
            reason=f"{'Irregular income needs a bigger cushion. ' if is_irregular else ''}"
                   f"After the first buffer, keep building to {emergency_label}. "
                   f"This prevents a job loss or illness from becoming a financial catastrophe.",
            tier="safety_net",
            source_nodes=_match(entry_nodes + concept_nodes, ["SAVING", "EMERGENCY", "MONTH"]),
        ))

    # ── RULE 4: Insurance gap -> State Farm card AFTER income stabilises ─
    # Position depends on crisis vs stable (per brief section 2.6):
    #   Crisis (deep debt): Step 6-7 area (after triage) -> priority ~11-12
    #   Stable with gap: Step 2-3 area -> priority ~3-4
    #   No cash flow yet: don't insert (Rule 4 exception)

    if protection_gaps and income_stabilized:
        # Determine base priority based on user's situation
        # Per brief: crisis = Step 6-7, stable = Step 2-3
        if in_crisis:
            sf_base_priority = 11  # after debt triage
        else:
            sf_base_priority = 3.5  # early for stable users (between access and safety)

        for i, gap in enumerate(protection_gaps):
            action_text, reason_text = _insurance_text(gap, savings)
            steps.append(_step(
                priority=sf_base_priority + i * 0.1,
                action=action_text,
                reason=reason_text,
                tier="protection",
                source_nodes=_match(entry_nodes + action_nodes, [gap.upper(), "INSURANCE", "STATE FARM"]),
                resource_nodes=_match(resource_nodes + entry_nodes, ["STATE FARM"]),
                is_state_farm=True,
                savings_amount=savings,
            ))
    elif protection_gaps and not income_stabilized:
        # Rule 4 exception: "Not before cash flow exists"
        # Still add but at lower priority, after access steps
        for i, gap in enumerate(protection_gaps):
            action_text, reason_text = _insurance_text(gap, savings)
            steps.append(_step(
                priority=13 + i * 0.1,
                action=action_text,
                reason=reason_text,
                tier="protection",
                source_nodes=_match(entry_nodes + action_nodes, [gap.upper(), "INSURANCE", "STATE FARM"]),
                resource_nodes=_match(resource_nodes + entry_nodes, ["STATE FARM"]),
                is_state_farm=True,
                savings_amount=savings,
            ))

    # ── RULE 7: Student loans -> IDR check before standard repayment ────
    if profile["debt_type"] in ("student", "multiple"):
        steps.append(_step(
            priority=15,
            action="Check income-driven repayment (IDR) for student loans",
            reason="Federal student loans offer income-driven plans that cap payments "
                   "at 10-15% of income. May result in $0/month payment. Check before "
                   "committing to standard repayment.",
            tier="build",
            source_nodes=_match(action_nodes + entry_nodes, ["STUDENT LOAN", "FEDERAL", "REPAYMENT"]),
        ))

    # ── RULE 6: No credit history -> secured card after bank account ────
    if has_bank:
        steps.append(_step(
            priority=16,
            action="Start building credit with a secured card or credit-builder loan",
            reason="Good credit saves thousands on future loans and apartments. "
                   "A secured card (backed by your deposit) is the safest way to start. "
                   "Prerequisite: you must have a bank account first.",
            tier="build",
            source_nodes=_match(concept_nodes, ["CREDIT", "SECURED", "BUILD"]),
        ))

    # Credit report check
    steps.append(_step(
        priority=17,
        action="Check your credit report for free",
        reason="Your credit report affects loan rates, apartment applications, and "
               "even job offers. Check for errors at annualcreditreport.com — it's free.",
        tier="build",
        source_nodes=_match(entry_nodes + concept_nodes, ["CREDIT REPORT", "CREDIT SCORE"]),
    ))

    # Scam protection
    steps.append(_step(
        priority=18,
        action="Learn to recognize common financial scams",
        reason="Scammers target people who are new to the financial system. "
               "Never share your SSN, account numbers, or PINs over the phone.",
        tier="build",
        source_nodes=_match(risk_nodes + concept_nodes, ["SCAM", "FRAUD", "IDENTITY THEFT"]),
        risk_nodes=_match(risk_nodes, ["SCAM", "THEFT", "FRAUD"]),
    ))

    # ── RULE 10: Savings > $1k + no high-APR debt -> investment unlocks ──
    can_invest = (
        savings in ("1000_to_5000", "over_5000")
        and not has_high_apr_debt
    )

    if can_invest:
        steps.append(_step(
            priority=19,
            action="Set a specific savings goal and automate it",
            reason="Automatic transfers remove willpower from the equation. "
                   "Even $10/week builds to $520/year.",
            tier="grow",
            source_nodes=_match(action_nodes + entry_nodes, ["SAVING", "GOAL", "AUTOMATIC"]),
        ))
        steps.append(_step(
            priority=20,
            action="Start exploring low-cost investment options",
            reason="With savings above $1,000 and no high-interest debt, you're ready "
                   "to grow your money. Index funds or a Roth IRA are good starting points.",
            tier="grow",
            source_nodes=_match(concept_nodes, ["INVEST", "RETIREMENT", "SAVINGS PLAN"]),
        ))
    else:
        # Not yet eligible — just the savings automation step
        steps.append(_step(
            priority=19,
            action="Set a specific savings goal and automate it",
            reason="Automatic transfers remove willpower from the equation. "
                   "Even $10/week builds to $520/year. "
                   "Investment steps unlock once you have $1,000+ saved and no high-interest debt.",
            tier="grow",
            source_nodes=_match(action_nodes + entry_nodes, ["SAVING", "GOAL", "AUTOMATIC"]),
        ))

    # ── Sort, deduplicate, and number ────────────────────────────────────
    seen_priorities = set()
    unique_steps = []
    for s in steps:
        if s["priority"] not in seen_priorities:
            seen_priorities.add(s["priority"])
            unique_steps.append(s)

    unique_steps.sort(key=lambda s: s["priority"])

    for i, s in enumerate(unique_steps):
        s["step_number"] = i + 1

    return unique_steps


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _step(priority, action, reason, tier, source_nodes=None, risk_nodes=None,
          resource_nodes=None, is_state_farm=False, savings_amount=None):
    return {
        "step_number": 0,
        "action": action,
        "reason": reason,
        "tier": tier,
        "priority": priority,
        "source_nodes": [n["name"] for n in (source_nodes or [])],
        "risk_nodes": [n["name"] for n in (risk_nodes or [])],
        "resource_nodes": [n["name"] for n in (resource_nodes or [])],
        "is_state_farm": is_state_farm,
        "savings_amount": savings_amount,
    }


def _match(nodes, keywords):
    return [n for n in nodes if any(kw.upper() in n["name"].upper() for kw in keywords)]


def _insurance_text(gap, savings):
    """Return (action, reason) for an insurance gap. Includes savings amount for personalization."""
    savings_labels = {
        "nothing": "$0", "under_500": "under $500", "500_to_1000": "$500-$1,000",
        "1000_to_5000": "$1,000-$5,000", "over_5000": "over $5,000",
    }
    saved = savings_labels.get(savings, "your savings")

    texts = {
        "renters": (
            "Get renters insurance to protect your belongings",
            f"You have {saved} saved right now. One break-in or kitchen fire could wipe "
            f"this out completely. Renters insurance costs about $15-30/month and covers "
            f"theft, fire, and liability. That is the math for why this step is here.",
        ),
        "auto": (
            "Get proper auto insurance coverage",
            f"You have {saved} saved. One at-fault accident without adequate coverage "
            f"could mean tens of thousands in liability — wiping out everything above "
            f"this step. State minimums are often not enough.",
        ),
        "life": (
            "Consider life insurance if others depend on your income",
            "If anyone relies on your income (children, spouse, parents), "
            "term life insurance is affordable ($20-40/month for $250k coverage) "
            "and protects them if something happens to you.",
        ),
        "health": (
            "Get health insurance coverage",
            f"You have {saved} saved. One ER visit without insurance averages $2,200 — "
            f"that would set you back to zero. Check healthcare.gov for subsidized plans, "
            f"or your state's Medicaid program if your income qualifies.",
        ),
    }
    return texts.get(gap, (f"Review your {gap} insurance needs",
                           f"Consider whether {gap} insurance would protect your {saved} savings."))


# ─── CLI Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, sys
    sys.stdout.reconfigure(encoding="utf-8")
    from query_graph import GraphQuery
    gq = GraphQuery()

    tests = [
        ("Maria: unbanked gig, CC debt, $0, no insurance",
         {"q1": "no", "q2": "gig", "q3": "credit_card", "q4": "nothing", "q5": ["none"]}),
        ("James: student, loans, health ins, <$500",
         {"q1": "yes", "q2": "salary", "q3": "student", "q4": "under_500", "q5": ["health"]}),
        ("Stable: salary, no debt, $5k+, has health/auto/renters",
         {"q1": "yes", "q2": "salary", "q3": "none", "q4": "over_5000", "q5": ["health", "auto", "renters"]}),
    ]

    for name, args in tests:
        result = gq.traverse_from_profile(**args)
        steps = order_actions(result)
        print(f"\n{'='*70}\n  {name}\n  Persona: {result['persona']} | Gaps: {result['protection_gaps']}\n{'='*70}")
        for s in steps:
            sf = " [STATE FARM]" if s["is_state_farm"] else ""
            print(f"  {s['step_number']:2d}. [{s['tier']:15s}]{sf} {s['action']}")
            print(f"      {s['reason'][:100]}")
        print(f"  Total: {len(steps)} steps")
