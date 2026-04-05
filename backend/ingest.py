#!/usr/bin/env python3
"""
PennyPath — Data Ingestion Pipeline (Person 2)
Downloads 6 verified financial data sources, extracts text, saves to input/ for GraphRAG indexing.

Sources:
  1. CFPB Your Money Your Goals Toolkit (split into 9 modules)
  2. CFPB Behind on Bills booklet
  3. CFPB Debt Getting in Your Way booklet
  4. FEMA Emergency Financial First Aid Kit (EFFAK)
  5. State Farm Simple Insights (12 articles, saved individually)
  6. CFPB Newcomer's Guides (2 web pages)
"""

import os
import sys
import time
import json
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import quote as urlquote

# ─── Configuration ───────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent
INPUT_DIR = ROOT_DIR / "input"
DOWNLOAD_DIR = ROOT_DIR / "downloads"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ─── PDF Sources ─────────────────────────────────────────────────────────────

PDF_SOURCES = [
    {
        "name": "cfpb_ymyg_toolkit",
        "url": "https://files.consumerfinance.gov/f/documents/cfpb_your-money-your-goals_financial-empowerment_toolkit.pdf",
        "description": "CFPB Your Money Your Goals - Full Toolkit",
        "split_modules": True,
    },
    {
        "name": "cfpb_behind_on_bills",
        "url": "https://files.consumerfinance.gov/f/documents/cfpb_ymyg_behind-on-bills_print.pdf",
        "description": "CFPB Behind on Bills - Crisis Guide",
        "split_modules": False,
    },
    {
        "name": "cfpb_debt",
        "url": "https://files.consumerfinance.gov/f/documents/bcfp_your-money-goals_debt_booklet_print.pdf",
        "description": "CFPB Debt Getting in Your Way - Student Loans and Debt Management",
        "split_modules": False,
    },
    {
        "name": "fema_effak",
        "url": "https://www.ready.gov/sites/default/files/2020-03/ready_emergency-financial-first-aid-toolkit.pdf",
        "description": "FEMA Emergency Financial First Aid Kit",
        "split_modules": False,
    },
]

# ─── State Farm Simple Insights (12 verified URLs) ──────────────────────────

STATEFARM_PAGES = [
    ("statefarm_types_of_insurance", "https://www.statefarm.com/simple-insights/residence/what-are-all-the-different-types-of-insurance"),
    ("statefarm_renters_how_much", "https://www.statefarm.com/simple-insights/residence/how-much-renters-insurance-do-i-need"),
    ("statefarm_renters_hotel", "https://www.statefarm.com/simple-insights/residence/does-renters-insurance-cover-hotel-stay"),
    ("statefarm_life_what_is", "https://www.statefarm.com/simple-insights/life-insurance/what-is-life-insurance"),
    ("statefarm_life_how_much", "https://www.statefarm.com/simple-insights/life-insurance/how-much-life-insurance-do-i-need"),
    ("statefarm_life_types", "https://www.statefarm.com/simple-insights/life-insurance/types-of-life-insurance"),
    ("statefarm_auto_deductibles", "https://www.statefarm.com/simple-insights/auto-and-vehicles/car-insurance-deductibles-and-coverages-choosing-well"),
    ("statefarm_auto_how_much", "https://www.statefarm.com/simple-insights/auto-and-vehicles/how-much-car-insurance-do-i-need"),
    ("statefarm_auto_premiums", "https://www.statefarm.com/simple-insights/auto-and-vehicles/what-affects-car-insurance-premiums"),
    ("statefarm_auto_full_coverage", "https://www.statefarm.com/simple-insights/auto-and-vehicles/what-is-full-coverage-auto-insurance"),
    ("statefarm_bundle", "https://www.statefarm.com/simple-insights/auto-and-vehicles/bundle-insurance"),
    ("statefarm_insurance_review", "https://www.statefarm.com/simple-insights/financial/moments-when-you-should-check-your-insurance"),
]

# ─── CFPB Newcomer's Guides ─────────────────────────────────────────────────

NEWCOMER_PAGES = [
    ("cfpb_newcomer_guides", "https://www.consumerfinance.gov/consumer-tools/educator-tools/your-money-your-goals/companion-guides/"),
    ("cfpb_newcomer_blog", "https://www.consumerfinance.gov/about-us/blog/the-newcomers-guides-to-managing-money/"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD & EXTRACTION FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def verify_url(url: str) -> bool:
    """HEAD-request a URL to check if it's alive."""
    try:
        resp = requests.head(url, headers=HEADERS, timeout=15, allow_redirects=True)
        return resp.status_code < 400
    except Exception:
        # Some servers block HEAD, try GET with stream
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15, stream=True)
            resp.close()
            return resp.status_code < 400
        except Exception:
            return False


def download_pdf(url: str, name: str, retries: int = 3) -> Path | None:
    """Download a PDF with retry logic. Returns local path or None."""
    dest = DOWNLOAD_DIR / f"{name}.pdf"
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"    [cached] {dest.name} ({dest.stat().st_size / 1_048_576:.1f} MB)")
        return dest

    for attempt in range(1, retries + 1):
        try:
            print(f"    downloading (attempt {attempt}/{retries})...")
            resp = requests.get(url, headers=HEADERS, timeout=120)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"    saved {dest.name} ({len(resp.content) / 1_048_576:.1f} MB)")
            return dest
        except Exception as e:
            print(f"    attempt {attempt} failed: {e}")
            if attempt < retries:
                wait = 2 ** attempt
                print(f"    retrying in {wait}s...")
                time.sleep(wait)

    print(f"    FAILED all {retries} attempts for {name}")
    return None


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract all text from a PDF using PyMuPDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        if text.strip():
            pages.append(f"\n--- Page {i + 1} ---\n{text}")
    doc.close()
    return "\n".join(pages)


def split_ymyg_modules(full_text: str) -> dict[str, str]:
    """Split the YMYG toolkit into module-based chunks.

    Uses the actual section headers (MODULE N  Title) with double-space
    pattern to find real content sections, not TOC references.
    """
    import re

    module_names = {
        "1": "module_1_goals",
        "2": "module_2_saving",
        "3": "module_3_income",
        "4": "module_4_bills",
        "5": "module_5_monthly",
        "6": "module_6_debt",
        "7": "module_7_credit",
        "8": "module_8_products",
        "9": "module_9_protecting",
    }

    # Find actual module headers: "MODULE N  Title" (double space before title)
    pattern = r'MODULE\s+(\d)\s{2,}\S'
    positions = []
    for match in re.finditer(pattern, full_text):
        mod_num = match.group(1)
        positions.append((mod_num, match.start()))

    modules = {}

    if not positions:
        print("      could not split by modules, saving as single file")
        modules["full_toolkit"] = full_text
        return modules

    # Capture intro (everything before first module)
    first_pos = positions[0][1]
    if first_pos > 500:
        modules["intro"] = full_text[:first_pos].strip()
        print(f"      intro: {len(modules['intro']):,} chars")

    # Extract each module
    for i, (mod_num, start) in enumerate(positions):
        end = positions[i + 1][1] if i + 1 < len(positions) else len(full_text)
        chunk = full_text[start:end].strip()
        name = module_names.get(mod_num, f"module_{mod_num}")
        if len(chunk) > 200:
            modules[name] = chunk
            print(f"      {name}: {len(chunk):,} chars")

    return modules


# ═══════════════════════════════════════════════════════════════════════════════
# WEB SCRAPING — 3-tier: BS4 -> Playwright -> Wayback Machine
# ═══════════════════════════════════════════════════════════════════════════════

def scrape_bs4(url: str) -> str | None:
    """Scrape with requests + BeautifulSoup."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise
        for tag in soup(["nav", "footer", "script", "style", "header", "aside", "noscript"]):
            tag.decompose()

        # Find main content
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", {"role": "main"})
            or soup.find("div", class_="content")
        )
        target = main if main else soup.body or soup

        text = target.get_text(separator="\n", strip=True)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        text = "\n".join(lines)

        return text if len(text) > 200 else None
    except Exception as e:
        print(f"      bs4 error: {e}")
        return None


def scrape_playwright(url: str) -> str | None:
    """Scrape with Playwright headless browser (JS-rendered pages)."""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2000)  # let JS render

            # Remove noise elements
            page.evaluate("""
                document.querySelectorAll('nav, footer, header, aside, script, style, noscript')
                    .forEach(el => el.remove());
            """)

            # Get main content
            main = page.query_selector("main") or page.query_selector("article") or page.query_selector("body")
            text = main.inner_text() if main else page.inner_text()
            browser.close()

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)
            return text if len(text) > 200 else None
    except Exception as e:
        print(f"      playwright error: {e}")
        return None


def scrape_wayback(url: str) -> str | None:
    """Try fetching from the Wayback Machine as last resort."""
    try:
        wb_url = f"https://web.archive.org/web/{urlquote(url, safe='')}"
        print(f"      trying wayback: {wb_url[:80]}...")
        return scrape_bs4(wb_url)
    except Exception as e:
        print(f"      wayback error: {e}")
        return None


def scrape_page(url: str) -> str | None:
    """Orchestrate scraping: BS4 -> Playwright -> Wayback."""
    text = scrape_bs4(url)
    if text and len(text) > 200:
        return text

    print(f"      bs4 insufficient, trying playwright...")
    text = scrape_playwright(url)
    if text and len(text) > 200:
        return text

    print(f"      playwright failed, trying wayback...")
    return scrape_wayback(url)


# ═══════════════════════════════════════════════════════════════════════════════
# FILE OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def save_text(filename: str, text: str, description: str = "") -> Path:
    """Save text to input/ directory for GraphRAG indexing."""
    path = INPUT_DIR / f"{filename}.txt"
    header = f"# {description}\n# Source: {filename}\n\n" if description else ""
    path.write_text(header + text, encoding="utf-8")
    size_kb = len(text) / 1024
    print(f"    saved: {path.name} ({size_kb:.0f} KB, {len(text):,} chars)")
    return path


def validate_text(text: str, source_name: str) -> bool:
    """Check that extracted text is valid (not error pages, not too short)."""
    if not text or len(text) < 200:
        print(f"    WARN: {source_name} — text too short ({len(text) if text else 0} chars)")
        return False
    bad_markers = ["403 forbidden", "access denied", "page not found", "404 error"]
    text_lower = text[:500].lower()
    for marker in bad_markers:
        if marker in text_lower:
            print(f"    WARN: {source_name} — looks like an error page ('{marker}')")
            return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  PennyPath — Data Ingestion Pipeline")
    print("=" * 70)

    INPUT_DIR.mkdir(exist_ok=True)
    DOWNLOAD_DIR.mkdir(exist_ok=True)

    total_chars = 0
    file_count = 0
    failures = []

    # ── Step 0: Verify all URLs ──────────────────────────────────────────
    print("\n[Step 0] Verifying URLs...")
    all_urls = (
        [(s["name"], s["url"]) for s in PDF_SOURCES]
        + [(name, url) for name, url in STATEFARM_PAGES]
        + [(name, url) for name, url in NEWCOMER_PAGES]
    )
    for name, url in all_urls:
        alive = verify_url(url)
        status = "OK" if alive else "DEAD"
        print(f"  [{status}] {name}")
        if not alive:
            failures.append(f"URL dead: {name}")

    # ── Step 1: Download and extract PDFs ────────────────────────────────
    print("\n[Step 1] Downloading and extracting PDFs...")

    for source in PDF_SOURCES:
        name = source["name"]
        print(f"\n  [{name}] {source['description']}")

        pdf_path = download_pdf(source["url"], name)
        if not pdf_path:
            failures.append(f"PDF download failed: {name}")
            continue

        print(f"    extracting text...")
        text = extract_pdf_text(pdf_path)

        if not validate_text(text, name):
            failures.append(f"PDF extraction failed: {name}")
            continue

        print(f"    extracted {len(text):,} characters")

        if source.get("split_modules"):
            print(f"    splitting into modules...")
            modules = split_ymyg_modules(text)
            for mod_name, mod_text in modules.items():
                save_text(
                    f"cfpb_ymyg_{mod_name}",
                    mod_text,
                    f"CFPB YMYG Toolkit - {mod_name.replace('_', ' ').title()}"
                )
                total_chars += len(mod_text)
                file_count += 1
        else:
            save_text(name, text, source["description"])
            total_chars += len(text)
            file_count += 1

    # ── Step 2: Scrape State Farm articles (individually) ────────────────
    print("\n\n[Step 2] Scraping State Farm Simple Insights...")

    for i, (name, url) in enumerate(STATEFARM_PAGES, 1):
        print(f"\n  [{i}/{len(STATEFARM_PAGES)}] {name}")
        text = scrape_page(url)

        if text and validate_text(text, name):
            # Add source URL as metadata for GraphRAG
            header_text = f"Source URL: {url}\n\n{text}"
            save_text(name, header_text, f"State Farm Simple Insights - {name.replace('statefarm_', '').replace('_', ' ').title()}")
            total_chars += len(header_text)
            file_count += 1
        else:
            failures.append(f"Scrape failed: {name}")

        time.sleep(1)  # polite delay

    # ── Step 3: Scrape CFPB Newcomer's Guides ────────────────────────────
    print("\n\n[Step 3] Scraping CFPB Newcomer's Guides...")

    newcomer_texts = []
    for name, url in NEWCOMER_PAGES:
        print(f"\n  [{name}]")
        text = scrape_page(url)
        if text and validate_text(text, name):
            newcomer_texts.append(f"Source URL: {url}\n\n{text}")
            print(f"    got {len(text):,} chars")
        else:
            failures.append(f"Scrape failed: {name}")
        time.sleep(1)

    if newcomer_texts:
        combined = "\n\n---\n\n".join(newcomer_texts)
        save_text(
            "cfpb_new_americans",
            combined,
            "CFPB Newcomer's Guides to Managing Money - For Immigrants and International Students"
        )
        total_chars += len(combined)
        file_count += 1

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  INGESTION COMPLETE")
    print(f"  Files created: {file_count}")
    print(f"  Total text:    {total_chars:,} chars ({total_chars / 1024:.0f} KB)")
    print(f"  Output dir:    {INPUT_DIR}")

    if failures:
        print(f"\n  WARNINGS ({len(failures)}):")
        for f in failures:
            print(f"    - {f}")

    print("=" * 70)

    # List files
    print("\n  Files ready for GraphRAG indexing:")
    for f in sorted(INPUT_DIR.glob("*.txt")):
        size = f.stat().st_size / 1024
        print(f"    {f.name:50s} {size:>8.1f} KB")

    print(f"\n  Next: run 'python -m graphrag index --root .' from backend/")

    # Save manifest for debugging
    manifest = {
        "files": file_count,
        "total_chars": total_chars,
        "failures": failures,
        "output_dir": str(INPUT_DIR),
    }
    (ROOT_DIR / "ingest_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
