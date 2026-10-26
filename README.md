# Manufacturing Engineering Change Impact Agent — Step 2: Neo4j Knowledge Graph

## Overview

A **Neo4j 5** knowledge graph for a synthetic industrial quadcopter drone, built to support LLM-based ECO (Engineering Change Order) impact analysis, RAG retrieval, and LangGraph agents (later steps).

- **Dataset**: fully synthetic; 11 CSVs, 1 500 ECOs, 6 000 ground-truth impact labels, 50 000 production records.
- **Neo4j Browser**: http://localhost:7474  |  Login: `neo4j / changeme123`

---

## Setup

```bash
# 1. Create virtual environment and install deps
python3 -m venv venv
source venv/bin/activate        # or: venv\Scripts\activate on Windows
pip install -r requirements.txt

# 2. Start Neo4j (Docker)
docker run -d --name neo4j-eco \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/changeme123 \
  neo4j:5

# 3. Copy and edit credentials (optional — defaults match above)
cp .env.example .env
```

---

## Running

```bash
# Full wipe + load (idempotent, safe to re-run)
python src/build_kg.py --wipe

# Verify the graph (all checks must pass before proceeding to Step 3)
python src/verify_kg.py
```

---

## Graph Schema

| Node label | Key ID column | Notable properties |
|---|---|---|
| `Product` | `id` | name |
| `Subsystem` | `id` | name |
| `Component` | `id` | name, subsystem, nominal_mass_g, prod_defect_rate, prod_ops_count … |
| `Material` | `id` | density_g_cm3, yield_strength_mpa, machinability_index, raw_cost_index, co2_kg_per_kg |
| `Process` | `id` | cycle_mean_min, power_kw, governing_sop |
| `Machine` | `id` | cpk, age_years, maintenance_status, prod_defect_rate … |
| `Supplier` | `id` | quality_rating, lead_time_days, qualification_status, certified_materials (list) |
| `SOP` | `id` | revision, scope, critical_tolerances, risk |
| `ECO` | `id` | family, change_type, current_value, proposed_value, intent, direct_severity, impact_dimension, description, status |

| Relationship | From → To | Edge properties |
|---|---|---|
| `CONTAINS` | Product/Subsystem → Subsystem/Component | quantity, fastener_required |
| `MADE_OF` | Component → Material | — |
| `MANUFACTURED_BY` | Component → Process | — |
| `CAPABLE_OF` | Machine → Process | cpk |
| `GOVERNED_BY` | Process/Component → SOP | — |
| `SUPPLIED_BY` | Material → Supplier | lead_time_days |
| `CHANGED` | ECO → Component | — |
| `FROM_MATERIAL` / `TO_MATERIAL` | ECO → Material | — |
| `FROM_MACHINE` / `TO_MACHINE` | ECO → Machine | — |
| `GT_IMPACTS` ⚠️ | ECO → Component/Process/Machine/SOP | severity, dimension, confidence, rationale |

> ⚠️ **`GT_IMPACTS` is the evaluation ANSWER KEY** (ground truth for impact assessment and GNN training).  
> **Agent traversal queries must NEVER use `GT_IMPACTS`.**

---

## Data Quirks Handled

| Quirk | Fix |
|---|---|
| UTF-8 BOM on all CSVs | `encoding="utf-8-sig"` throughout |
| `engineering_changes.csv`: unquoted commas in `technical_description` (col 10) | `csv.reader` + surplus-field merge |
| `components.csv`: unquoted commas in `cad_dimensions` (col 8) | Same repair; 16 columns expected |
| `graph_edges.csv`: mangled `properties_json` (`{\quantity\":1}"`) | Regex fallback `([A-Za-z_]+)\\*"?:\s*([\d.]+)` |
| `production_records.csv` (50 000 rows) | **Not loaded as nodes** — aggregated per `component_id` / `machine_id` and stored as `prod_*` properties |
| ~113 MACHINE_REPLACEMENT ECOs with incompatible FROM_MACHINE | Retained as-is; reported as data-quality warning in `verify_kg.py` |
| Node type and label columns use uppercase (`PRODUCT`, `COMPONENT`, …) | Mapped to Neo4j title-case labels in code |

---

## Verification Results (all passed ✅)

```
CHECK 1 – Node counts:  Component 50 ✅  Material 11 ✅  Process 8 ✅  Machine 24 ✅
          Supplier 25 ✅  SOP 12 ✅  ECO 1500 ✅  Subsystem 6 ✅  Product 1 ✅

CHECK 2 – Relationship counts: GT_IMPACTS = 6000 ✅

CHECK 3 – Orphan check: 0 components missing Material, Process, or Subsystem ✅

CHECK 4 – FRAME-TOP-PLATE 2-hop neighbourhood: valid paths found ✅

CHECK 5 – MAT-AL-6061 material swap: 32 rows (8 components × 4 CNC machines) ✅

CHECK 6 – MACHINE_REPLACEMENT mismatches: 113 ⚠️  (synthetic data known issue, retained)
```

---

## Project Layout

```
.
├── data/                       # 11 source CSVs (do not modify)
├── src/
│   ├── build_kg.py             # Graph builder (run with --wipe for clean load)
│   └── verify_kg.py            # Verification suite
├── .env                        # Neo4j credentials (gitignored)
├── .env.example                # Template
├── requirements.txt            # neo4j, pandas, python-dotenv
└── README.md
```

---

## Note on Synthetic Data

All data is **fully synthetic** and represents a fictional industrial quadcopter drone. No real engineering, materials, or supplier data are included.

## ignore this

// Schema overview
CALL db.schema.visualization();

// Neighbourhood of one component. This excludes GT_IMPACTS and ECO nodes, which would make it a hairball.
MATCH p=(c:Component {id:'FRAME-TOP-PLATE'})-[*1..2]-(n)
WHERE none(r IN relationships(p) WHERE type(r)='GT_IMPACTS')
  AND none(x IN nodes(p) WHERE x:ECO)
RETURN p LIMIT 100;

// The whole product structure
MATCH p=(:Product)-[:CONTAINS*1..2]->(:Component) RETURN p;

// Spot-check the repaired rows
MATCH (c:Component {id:'FASTENER-M3-TI'}) RETURN c.cad_dimensions;   // should be "M3x8, M3x12 hex"
MATCH (e:ECO {id:'ECO-000002'}) RETURN e.description;                // should end "...vendor qualifications."

// Ground truth for one ECO (for you to read, never for the agent)
MATCH (e:ECO {id:'ECO-000002'})-[g:GT_IMPACTS]->(n)
RETURN labels(n)[0] AS type, n.id, g.severity, g.rationale;