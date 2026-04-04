#!/usr/bin/env python3
"""
First Dollar — Gemini Personalization Layer

Single LLM call per session. Takes FOO-ordered steps and rewrites
the action labels to be warm, personal, and specific to the user.

Rules:
  - Does NOT change the order
  - Does NOT add or remove steps
  - Only rewrites action labels and reasons
  - Echoes user's Q6 words if provided
"""

import json
import re
import requests

API_KEYS = [
    "AIzaSyDn1tTJ7cuiWyG5ksBH91iIfFKF46QYQ_0",
    "AIzaSyAeW5k5qiD3C6vy5ojppmpUpll0W6KLYMY",
    "AIzaSyAnsPL7uEz6i_PuJAU6xoq9z8nDM-BwXrk",
    "AIzaSyDillXU2EzEcHanFbPFY3AWrTIuchDL1W8",
    "AIzaSyA2vt4reAGSLWL7yKFdKnn2cEBvRNEdwMw",
    "AIzaSyCkIHCzbs0KES_A9nS505AMP7AN4jfzNC4",
]

MODEL = "gemini-2.5-flash"


def personalize_steps(
    steps: list[dict],
    persona: str,
    profile: dict,
    q6_text: str | None = None,
) -> list[dict]:
    """
    Personalize FOO-ordered steps using a single Gemini call.

    Args:
        steps: Output of foo_engine.order_actions()
        persona: e.g., "unbanked", "student", "gig_worker"
        profile: User's Q1-Q5 profile dict
        q6_text: Optional free text from Q6

    Returns:
        Same steps list with added 'personalized_action' and 'personalized_reason' fields.
        Falls back to original text if LLM call fails.
    """
    # Build the prompt
    steps_text = "\n".join(
        f"Step {s['step_number']}: {s['action']} | Reason: {s['reason']}"
        for s in steps
    )

    persona_label = persona.replace("_", " ")

    profile_text = (
        f"Bank account: {profile.get('has_bank_account', '?')}, "
        f"Income: {profile.get('income_type', '?')}, "
        f"Debt: {profile.get('debt_type', '?')}, "
        f"Savings: {profile.get('savings_level', '?')}, "
        f"Insurance: {', '.join(profile.get('insurance_types', []))}"
    )

    q6_line = f'\nThe user said in their own words: "{q6_text}"' if q6_text else ""

    prompt = f"""You are a warm, supportive financial coach writing for someone who is a {persona_label}.

User profile: {profile_text}{q6_line}

Below are their personalized financial priority steps, already in the correct order.
Rewrite ONLY the action text and reason to be:
- Warm and encouraging (like a friend who happens to be great with money)
- Specific to their situation (reference their profile details)
- Simple language (no jargon, 8th grade reading level)
- Short (action: max 15 words, reason: max 2 sentences)
{f'- Echo their own words from Q6 where relevant' if q6_text else ''}

Do NOT change the order, add steps, or remove steps. Keep the same step numbers.

Steps to personalize:
{steps_text}

Return ONLY a JSON array, no other text. Format:
[{{"step": 1, "action": "...", "reason": "..."}}, ...]"""

    # Call Gemini with key rotation
    response_text = _call_gemini(prompt)

    if response_text:
        personalized = _parse_response(response_text)
        if personalized and len(personalized) == len(steps):
            for step, p in zip(steps, personalized):
                step["personalized_action"] = p.get("action", step["action"])
                step["personalized_reason"] = p.get("reason", step["reason"])
            return steps

    # Fallback: use original text
    for step in steps:
        step["personalized_action"] = step["action"]
        step["personalized_reason"] = step["reason"]
    return steps


def _call_gemini(prompt: str) -> str | None:
    """Call Gemini with API key rotation."""
    for i, key in enumerate(API_KEYS):
        try:
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "maxOutputTokens": 4000,
                        "temperature": 0.7,
                    },
                },
                timeout=30,
            )
            if resp.status_code == 200:
                text = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
                return text.strip()
            # Rate limited — try next key
        except Exception:
            continue
    return None


def _parse_response(text: str) -> list[dict] | None:
    """Parse JSON array from LLM response."""
    try:
        # Try direct parse
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ─── CLI Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    from query_graph import GraphQuery
    from foo_engine import order_actions

    gq = GraphQuery()

    # Test with Aisha (immigrant persona with Q6)
    print("=" * 60)
    print("  Testing Gemini Personalization — Aisha (immigrant)")
    print("=" * 60)

    result = gq.traverse_from_profile(
        q1="no", q2="cash", q3="none", q4="nothing", q5=["none"],
        q6="I just moved to America and need help with money"
    )
    steps = order_actions(result)
    personalized = personalize_steps(
        steps,
        persona=result["persona"],
        profile=result["profile"],
        q6_text="I just moved to America and need help with money",
    )

    for s in personalized:
        sf = " [STATE FARM]" if s["is_state_farm"] else ""
        print(f"\n  Step {s['step_number']}.{sf}")
        print(f"  ORIGINAL:      {s['action']}")
        print(f"  PERSONALIZED:  {s['personalized_action']}")
        print(f"  REASON:        {s['personalized_reason']}")

    # Test with Maria (unbanked gig worker)
    print("\n" + "=" * 60)
    print("  Testing Gemini Personalization — Maria (unbanked gig worker)")
    print("=" * 60)

    result2 = gq.traverse_from_profile(
        q1="no", q2="gig", q3="credit_card", q4="nothing", q5=["none"]
    )
    steps2 = order_actions(result2)
    personalized2 = personalize_steps(
        steps2,
        persona=result2["persona"],
        profile=result2["profile"],
    )

    for s in personalized2:
        sf = " [STATE FARM]" if s["is_state_farm"] else ""
        print(f"\n  Step {s['step_number']}.{sf}")
        print(f"  ORIGINAL:      {s['action']}")
        print(f"  PERSONALIZED:  {s['personalized_action']}")
        print(f"  REASON:        {s['personalized_reason']}")
