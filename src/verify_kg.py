"""
verify_kg.py  –  Verification suite for the Manufacturing ECO Knowledge Graph
=============================================================================
Usage:  python src/verify_kg.py

Reads credentials from .env  (NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD).

Checks performed
----------------
1. Node counts per label  (expects exact numbers for core labels)
2. Relationship counts per type  (GT_IMPACTS must equal #is_affected=1 rows)
3. Orphan check: Components missing Material, Process, or Subsystem  (expect 0)
4. 2-hop neighbourhood of FRAME-TOP-PLATE
5. Material swap query: components using MAT-AL-6061
6. MACHINE_REPLACEMENT ECOs whose FROM_MACHINE cannot serve the target process (expect ~113)

NOTE: GT_IMPACTS is the evaluation ANSWER KEY – agent traversal must NEVER use it.
"""

import os
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from neo4j import GraphDatabase

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme123")

PASS = "✅"
FAIL = "❌"
WARN = "⚠️ "

all_ok = True


def connect():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        driver.verify_connectivity()
        print(f"Connected to Neo4j at {URI}\n")
    except Exception as e:
        print(f"[ERROR] Cannot connect: {e}")
        sys.exit(1)
    return driver


def banner(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print('─'*60)


# ─────────────────────────────────────────────────────────────
# CHECK 1 – Node counts
# ─────────────────────────────────────────────────────────────

EXPECTED_COUNTS = {
    "Component": 50,
    "Material":  11,
    "Process":    8,
    "Machine":   24,
    "Supplier":  25,
    "SOP":       12,
    "ECO":     1500,
    "Subsystem":  6,
    "Product":    1,
}

def check_node_counts(driver):
    global all_ok
    banner("CHECK 1 – Node counts per label")
    counts = {}
    with driver.session() as s:
        for label in EXPECTED_COUNTS:
            r = s.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
            counts[label] = r.single()["cnt"]

    rows = []
    for label, expected in EXPECTED_COUNTS.items():
        actual = counts.get(label, 0)
        ok = actual == expected
        if not ok:
            all_ok = False
        rows.append({
            "Label": label,
            "Expected": expected,
            "Actual": actual,
            "Status": PASS if ok else FAIL,
        })
    print(pd.DataFrame(rows).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# CHECK 2 – Relationship counts
# ─────────────────────────────────────────────────────────────

EXPECTED_GT_IMPACTS = 6000  # rows where is_affected == 1

def check_rel_counts(driver):
    global all_ok
    banner("CHECK 2 – Relationship counts per type")

    rel_types = [
        "CONTAINS", "MADE_OF", "MANUFACTURED_BY", "CAPABLE_OF",
        "GOVERNED_BY", "SUPPLIED_BY", "CHANGED",
        "FROM_MATERIAL", "TO_MATERIAL", "FROM_MACHINE", "TO_MACHINE",
        "GT_IMPACTS",
    ]
    rows = []
    with driver.session() as s:
        for rt in rel_types:
            r = s.run(f"MATCH ()-[r:{rt}]->() RETURN count(r) AS cnt")
            cnt = r.single()["cnt"]
            note = ""
            status = PASS
            if rt == "GT_IMPACTS":
                ok = cnt == EXPECTED_GT_IMPACTS
                note = f"(expected {EXPECTED_GT_IMPACTS})"
                if not ok:
                    all_ok = False
                    status = FAIL
            rows.append({"Type": rt, "Count": cnt, "Note": note, "Status": status})
    print(pd.DataFrame(rows).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# CHECK 3 – Orphan components
# ─────────────────────────────────────────────────────────────

def check_orphans(driver):
    global all_ok
    banner("CHECK 3 – Orphan components (missing Material / Process / Subsystem parent)")

    queries = {
        "no Material":           "MATCH (c:Component) WHERE NOT (c)-[:MADE_OF]->(:Material) RETURN c.id AS id",
        "no Process":            "MATCH (c:Component) WHERE NOT (c)-[:MANUFACTURED_BY]->(:Process) RETURN c.id AS id",
        "no Subsystem parent":   "MATCH (c:Component) WHERE NOT ()-[:CONTAINS]->(c) RETURN c.id AS id",
    }

    for desc, q in queries.items():
        with driver.session() as s:
            ids = [r["id"] for r in s.run(q)]
        ok = len(ids) == 0
        if not ok:
            all_ok = False
        status = PASS if ok else FAIL
        print(f"  {status}  Components with {desc}: {len(ids)}")
        if ids:
            print(f"      → {ids[:10]} {'…' if len(ids) > 10 else ''}")


# ─────────────────────────────────────────────────────────────
# CHECK 4 – 2-hop neighbourhood of FRAME-TOP-PLATE
# ─────────────────────────────────────────────────────────────

def check_neighbourhood(driver):
    banner("CHECK 4 – 2-hop neighbourhood of FRAME-TOP-PLATE")

    query = """
        MATCH (c:Component {id: 'FRAME-TOP-PLATE'})
        OPTIONAL MATCH (c)-[r1]->(n1)
        OPTIONAL MATCH (n1)-[r2]->(n2)
        RETURN
            type(r1)            AS rel1,
            labels(n1)[0]       AS label1,
            n1.id               AS node1,
            type(r2)            AS rel2,
            labels(n2)[0]       AS label2,
            n2.id               AS node2
        LIMIT 30
    """
    with driver.session() as s:
        rows = [dict(r) for r in s.run(query)]

    if not rows:
        print(f"  {FAIL}  FRAME-TOP-PLATE not found in graph!")
        global all_ok
        all_ok = False
        return

    print(f"  {PASS}  FRAME-TOP-PLATE found — showing first 30 2-hop paths:\n")
    print(pd.DataFrame(rows).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# CHECK 5 – Material swap query for MAT-AL-6061
# ─────────────────────────────────────────────────────────────

def check_material_swap(driver):
    banner("CHECK 5 – Components using MAT-AL-6061, their processes & capable machines")

    query = """
        MATCH (c:Component)-[:MADE_OF]->(m:Material {id: 'MAT-AL-6061'})
        OPTIONAL MATCH (c)-[:MANUFACTURED_BY]->(p:Process)
        OPTIONAL MATCH (mach:Machine)-[:CAPABLE_OF]->(p)
        RETURN
            c.id         AS component,
            p.id         AS process,
            mach.id      AS machine,
            mach.cpk     AS cpk
        ORDER BY c.id, mach.id
    """
    with driver.session() as s:
        rows = [dict(r) for r in s.run(query)]

    if not rows:
        print(f"  {WARN}  No components found using MAT-AL-6061")
        return

    print(f"  Found {len(rows)} rows.\n")
    print(pd.DataFrame(rows).to_string(index=False))


# ─────────────────────────────────────────────────────────────
# CHECK 6 – MACHINE_REPLACEMENT mismatch count
# ─────────────────────────────────────────────────────────────

def check_machine_replacement_mismatches(driver):
    banner("CHECK 6 – MACHINE_REPLACEMENT ECOs where FROM_MACHINE can't serve the component's process")

    query = """
        MATCH (e:ECO {change_type: 'MACHINE_REPLACEMENT'})
              -[:CHANGED]->(c:Component)
              -[:MANUFACTURED_BY]->(p:Process)
        MATCH (e)-[:FROM_MACHINE]->(fm:Machine)
        WHERE NOT (fm)-[:CAPABLE_OF]->(p)
        RETURN count(e) AS mismatch_count
    """
    with driver.session() as s:
        result = s.run(query).single()
        cnt = result["mismatch_count"] if result else 0

    print(f"\n  {WARN}  DATA QUALITY WARNING")
    print(f"  MACHINE_REPLACEMENT ECOs where FROM_MACHINE cannot perform")
    print(f"  the component's manufacturing process: {cnt}")
    print(f"  (Expected ~113 — synthetic data known issue; rows retained as-is)")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    driver = connect()

    check_node_counts(driver)
    check_rel_counts(driver)
    check_orphans(driver)
    check_neighbourhood(driver)
    check_material_swap(driver)
    check_machine_replacement_mismatches(driver)

    driver.close()

    print("\n" + "═"*60)
    if all_ok:
        print(f"  {PASS}  ALL CHECKS PASSED")
    else:
        print(f"  {FAIL}  ONE OR MORE CHECKS FAILED — review output above")
    print("═"*60)
    print("\nBrowse the graph at: http://localhost:7474")
    print("Login: neo4j / changeme123")
    print("\nREMINDER: GT_IMPACTS is the evaluation answer key.")
    print("          Agent traversal queries must NEVER use GT_IMPACTS.\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
