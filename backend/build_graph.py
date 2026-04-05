#!/usr/bin/env python3
"""
PennyPath — Build Knowledge Graph from Cached Extractions
Bypasses LLM summarization by directly constructing the graph
from GraphRAG's cached entity extraction results.

Outputs:
  - output/knowledge_graph.graphml  (for visualization + NetworkX queries)
  - output/entities.json            (deduplicated entity list)
  - output/relationships.json       (relationship list)
  - output/communities.json         (Louvain community clusters)
  - output/graph_stats.json         (summary statistics)
"""

import json
import re
from pathlib import Path
from collections import defaultdict

try:
    import networkx as nx
    from networkx.algorithms.community import louvain_communities
except ImportError:
    print("Installing networkx...")
    import subprocess
    subprocess.check_call(["pip", "install", "networkx"])
    import networkx as nx
    from networkx.algorithms.community import louvain_communities

ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache" / "extract_graph"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def parse_extractions() -> tuple[list[dict], list[dict]]:
    """Parse all cached extraction results into entities and relationships."""
    entities = []
    relationships = []

    for f in sorted(CACHE_DIR.iterdir()):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            content = data["result"]["response"]["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError):
            continue

        for line in content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.strip("()").split("<|>")
            if len(parts) < 4:
                continue

            record_type = parts[0].strip().strip('"').lower()

            if record_type == "entity" and len(parts) >= 4:
                entities.append({
                    "name": parts[1].strip().upper(),  # normalize case
                    "type": parts[2].strip().upper(),
                    "description": parts[3].strip().rstrip('")'),
                })
            elif record_type == "relationship" and len(parts) >= 5:
                weight_str = parts[4].strip().rstrip('")')
                try:
                    weight = float(re.search(r"[\d.]+", weight_str).group())
                except (AttributeError, ValueError):
                    weight = 1.0
                relationships.append({
                    "source": parts[1].strip().upper(),
                    "target": parts[2].strip().upper(),
                    "description": parts[3].strip(),
                    "weight": weight,
                })

    return entities, relationships


def deduplicate_entities(entities: list[dict]) -> dict[str, dict]:
    """Deduplicate entities by name, merging descriptions."""
    merged = {}
    for e in entities:
        name = e["name"]
        if name in merged:
            # Keep longest description, prefer more specific type
            existing = merged[name]
            if len(e["description"]) > len(existing["description"]):
                existing["description"] = e["description"]
            # Track all types seen
            existing["types"].add(e["type"])
            existing["count"] += 1
        else:
            merged[name] = {
                "name": name,
                "type": e["type"],
                "types": {e["type"]},
                "description": e["description"],
                "count": 1,
            }

    # Finalize: pick most common type
    for name, ent in merged.items():
        ent["types"] = list(ent["types"])
        # Keep primary type as the one originally set
    return merged


def build_graph(
    entities: dict[str, dict], relationships: list[dict]
) -> nx.Graph:
    """Build a NetworkX graph from entities and relationships."""
    G = nx.Graph()

    # Add entity nodes
    for name, ent in entities.items():
        G.add_node(
            name,
            type=ent["type"],
            description=ent["description"][:500],
            mention_count=ent["count"],
        )

    # Add relationship edges
    for rel in relationships:
        src, tgt = rel["source"], rel["target"]
        if src in entities and tgt in entities:
            if G.has_edge(src, tgt):
                # Merge: increase weight, append description
                G[src][tgt]["weight"] += rel["weight"]
                existing_desc = G[src][tgt]["description"]
                if len(existing_desc) < 500:
                    G[src][tgt]["description"] = (
                        existing_desc + " | " + rel["description"]
                    )[:500]
            else:
                G.add_edge(
                    src,
                    tgt,
                    weight=rel["weight"],
                    description=rel["description"][:500],
                )

    return G


def detect_communities(G: nx.Graph) -> list[dict]:
    """Run Louvain community detection on the graph."""
    # Only use the largest connected component for community detection
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest = max(components, key=len)
        subG = G.subgraph(largest).copy()
    else:
        subG = G

    communities = louvain_communities(subG, resolution=1.0, seed=42)

    community_data = []
    for i, community in enumerate(sorted(communities, key=len, reverse=True)):
        members = list(community)
        # Get most connected nodes as "leaders"
        subgraph = G.subgraph(members)
        degrees = sorted(
            subgraph.degree(), key=lambda x: x[1], reverse=True
        )
        leaders = [n for n, d in degrees[:5]]

        # Determine dominant types in community
        type_counts = defaultdict(int)
        for node in members:
            if node in G.nodes:
                type_counts[G.nodes[node].get("type", "UNKNOWN")] += 1

        community_data.append({
            "id": i,
            "size": len(members),
            "leaders": leaders,
            "dominant_types": dict(
                sorted(type_counts.items(), key=lambda x: -x[1])[:5]
            ),
            "members": members[:50],  # cap for JSON readability
        })

    return community_data


def main():
    print("=" * 60)
    print("  PennyPath — Building Knowledge Graph")
    print("=" * 60)

    # Step 1: Parse cache
    print("\n[1] Parsing cached extractions...")
    raw_entities, raw_relationships = parse_extractions()
    print(f"    Raw entities: {len(raw_entities)}")
    print(f"    Raw relationships: {len(raw_relationships)}")

    # Step 2: Deduplicate
    print("\n[2] Deduplicating entities...")
    entities = deduplicate_entities(raw_entities)
    print(f"    Unique entities: {len(entities)}")

    # Step 3: Build graph
    print("\n[3] Building graph...")
    G = build_graph(entities, raw_relationships)
    print(f"    Nodes: {G.number_of_nodes()}")
    print(f"    Edges: {G.number_of_edges()}")
    print(f"    Connected components: {nx.number_connected_components(G)}")

    # Step 4: Community detection
    print("\n[4] Detecting communities (Louvain)...")
    communities = detect_communities(G)
    print(f"    Communities found: {len(communities)}")
    for c in communities[:10]:
        leaders_str = ", ".join(c["leaders"][:3])
        print(f"    Community {c['id']}: {c['size']} nodes — {leaders_str}")

    # Step 5: Save outputs
    print("\n[5] Saving outputs...")

    # GraphML (for visualization tools like Gephi, yEd, or frontend)
    nx.write_graphml(G, str(OUTPUT_DIR / "knowledge_graph.graphml"))
    print(f"    knowledge_graph.graphml ({(OUTPUT_DIR / 'knowledge_graph.graphml').stat().st_size / 1024:.0f} KB)")

    # Entities JSON
    entities_list = [
        {k: v for k, v in ent.items() if k != "types"}
        for ent in sorted(entities.values(), key=lambda x: -x["count"])
    ]
    (OUTPUT_DIR / "entities.json").write_text(
        json.dumps(entities_list, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"    entities.json ({len(entities_list)} entities)")

    # Relationships JSON
    valid_rels = [
        r for r in raw_relationships
        if r["source"] in entities and r["target"] in entities
    ]
    (OUTPUT_DIR / "relationships.json").write_text(
        json.dumps(valid_rels, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"    relationships.json ({len(valid_rels)} relationships)")

    # Communities JSON
    (OUTPUT_DIR / "communities.json").write_text(
        json.dumps(communities, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"    communities.json ({len(communities)} communities)")

    # Stats
    type_dist = defaultdict(int)
    for ent in entities.values():
        type_dist[ent["type"]] += 1

    stats = {
        "total_entities": len(entities),
        "total_relationships": len(valid_rels),
        "total_communities": len(communities),
        "graph_nodes": G.number_of_nodes(),
        "graph_edges": G.number_of_edges(),
        "connected_components": nx.number_connected_components(G),
        "entity_type_distribution": dict(
            sorted(type_dist.items(), key=lambda x: -x[1])
        ),
        "top_entities_by_degree": [
            {"name": n, "degree": d, "type": G.nodes[n].get("type", "?")}
            for n, d in sorted(G.degree(), key=lambda x: -x[1])[:20]
        ],
    }
    (OUTPUT_DIR / "graph_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"    graph_stats.json")

    # Summary
    print("\n" + "=" * 60)
    print("  GRAPH BUILD COMPLETE")
    print(f"  {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  {len(communities)} communities detected")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 60)

    # Print top entities by connectivity
    print("\n  Top 15 entities by connections:")
    for n, d in sorted(G.degree(), key=lambda x: -x[1])[:15]:
        etype = G.nodes[n].get("type", "?")
        print(f"    [{etype:20s}] {n:40s} ({d} connections)")


if __name__ == "__main__":
    main()
