"""
agent.py  –  Step 5: Multi-agent orchestration (LangGraph) for ECO impact analysis
==================================================================================
Pipeline (nodes + conditional edges):

  START → Change Parser ──(component not found)──► END
              │
              ▼
        Graph Traversal           (Neo4j: direct dependencies + 2-hop neighbourhood)
              │
              ▼   conditional fan-out (only the relevant sub-agents run)
   ┌──────────┼───────────────┬──────────────────┐
   ▼          ▼               ▼                  ▼
 Material   Process        Historical         SOP / Compliance
 Comparison Compatibility  Precedent (RAG)    (always)
 (if material) (if material/machine/   (always)
               process/dimension/tolerance)
   └──────────┴───────────────┴──────────────────┘
              ▼
        Synthesis / Reasoning     (ranked High/Medium/Low impacts + rationale + confidence)
              ▼
        Report Generator          (Markdown → reports/)

Usage:
    python src/agent.py --request "Change FRAME-TOP-PLATE material from Al 6061 to Al 7075"
    python src/agent.py --eco ECO-000002            # use an existing ECO's free-text description
    python src/agent.py --eco ECO-000002 --eval     # ALSO score against the GT_IMPACTS answer key

Needs in .env:  ANTHROPIC_API_KEY, NEO4J_*   (and the vector index from `python src/rag.py index`)

IMPORTANT: the agent never reads GT_IMPACTS. It is read ONLY by evaluate(), after the run,
           when --eval is given, to score the result.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import anthropic
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph
from neo4j import GraphDatabase

sys.path.insert(0, str(Path(__file__).parent))
import rag  # Step 4: graph_rag(), hybrid(), doc_rag()

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

URI      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
USER     = os.getenv("NEO4J_USER",     "neo4j")
PASSWORD = os.getenv("NEO4J_PASSWORD", "changeme123")
MODEL    = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

DRIVER = None   # set in main()
CLIENT = None   # set in main()

IMPACT_TYPES = ["Component", "Process", "Machine", "SOP",
                "Material", "Supplier", "QualityRequirement"]
SEVERITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
MATERIAL_KEYS = ["density_g_cm3", "yield_strength_mpa", "ultimate_strength_mpa",
                 "thermal_conductivity", "machinability_index",
                 "raw_cost_index", "co2_kg_per_kg"]


def J(obj) -> str:
    return json.dumps(obj, indent=1, default=str)


# ══════════════════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════════════════

class State(TypedDict, total=False):
    request: str            # free-text change request
    eco_id: str             # optional: hides this ECO from its own precedent search
    parsed: dict            # structured intent from the Change Parser
    error: str
    direct: dict            # direct dependencies of the component
    blast: dict             # other components sharing material / process / SOP
    valid_ids: list         # node IDs the synthesis step is allowed to cite
    material_report: str
    material_deltas: dict
    process_report: str
    precedent_report: str
    precedents: list
    sop_report: str
    assessment: dict
    dropped: int
    report_md: str
    report_path: str


# ══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def llm_text(system: str, user: str, max_tokens: int = 900) -> str:
    resp = CLIENT.messages.create(model=MODEL, max_tokens=max_tokens, system=system,
                                  messages=[{"role": "user", "content": user}])
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def llm_tool(system: str, user: str, tool: dict, max_tokens: int = 1500) -> dict:
    resp = CLIENT.messages.create(
        model=MODEL, max_tokens=max_tokens, system=system, tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=[{"role": "user", "content": user}])
    for b in resp.content:
        if b.type == "tool_use":
            return b.input
    return {}


def catalogue(session) -> str:
    rows = session.run(
        "MATCH (n) WHERE (n:Component OR n:Material OR n:Machine) AND n.id IS NOT NULL "
        "RETURN labels(n)[0] AS l, n.id AS id, n.name AS name ORDER BY l, id").data()
    return "\n".join(f"{r['l']}: {r['id']} ({r['name']})" for r in rows)


# ══════════════════════════════════════════════════════════════════════════════
# NODE 1 – CHANGE PARSER  (LLM → structured intent)
# ══════════════════════════════════════════════════════════════════════════════

PARSE_TOOL = {
    "name": "record_change",
    "description": "Record the structured intent of the engineering change request.",
    "input_schema": {
        "type": "object",
        "properties": {
            "component_id": {"type": "string",
                             "description": "Exact Component ID from the catalogue"},
            "attribute": {"type": "string",
                          "enum": ["material", "machine", "process", "dimension",
                                   "tolerance", "supplier", "other"]},
            "old_value": {"type": "string", "description": "Old value as written in the request"},
            "new_value": {"type": "string", "description": "New value as written in the request"},
            "old_id": {"type": "string",
                       "description": "Catalogue ID of the old material/machine, or empty"},
            "new_id": {"type": "string",
                       "description": "Catalogue ID of the new material/machine, or empty"},
            "summary": {"type": "string", "description": "One-sentence summary of the change"},
        },
        "required": ["component_id", "attribute", "summary"],
    },
}


def parse_change(state: State) -> dict:
    with DRIVER.session() as s:
        cat = catalogue(s)
    system = ("You parse manufacturing engineering change requests into structured intent. "
              "Map every mentioned part, material or machine to its exact catalogue ID "
              "(leave an ID empty if it is not in the catalogue). Call record_change once.\n\n"
              "CATALOGUE:\n" + cat)
    parsed = llm_tool(system, state["request"], PARSE_TOOL, max_tokens=800)
    cid = parsed.get("component_id")
    with DRIVER.session() as s:
        found = s.run("MATCH (c:Component {id:$id}) RETURN c.id AS id", id=cid).data() if cid else []
    if not found:
        return {"parsed": parsed, "error": f"Component '{cid}' was not found in the graph."}
    return {"parsed": parsed}


def after_parse(state: State):
    return END if state.get("error") else "graph_traversal"


# ══════════════════════════════════════════════════════════════════════════════
# NODE 2 – GRAPH TRAVERSAL  (Neo4j; never touches GT_IMPACTS)
# ══════════════════════════════════════════════════════════════════════════════

def direct_context(session, cid: str) -> dict:
    rows = session.run("""
        MATCH (c:Component {id: $cid})
        OPTIONAL MATCH (c)-[:MADE_OF]->(m:Material)
        OPTIONAL MATCH (c)-[:MANUFACTURED_BY]->(p:Process)
        OPTIONAL MATCH (c)-[:GOVERNED_BY]->(s:SOP)
        OPTIONAL MATCH (c)-[:MUST_MEET]->(q:QualityRequirement)
        OPTIONAL MATCH (c)-[:SUPPLIED_BY]->(sp:Supplier)
        OPTIONAL MATCH (c)-[:PART_OF]->(sub:Subsystem)
        OPTIONAL MATCH (p)-[:REQUIRES]->(mc:Machine)
        RETURN properties(c) AS component, properties(m) AS material, properties(p) AS process,
               collect(DISTINCT properties(mc)) AS machines,
               collect(DISTINCT properties(s))  AS sops,
               collect(DISTINCT properties(q))  AS qrs,
               collect(DISTINCT properties(sp)) AS suppliers,
               collect(DISTINCT sub.id)         AS assemblies
    """, cid=cid).data()
    return rows[0] if rows else {}


def blast_radius(session, cid: str) -> dict:
    rows = session.run("""
        MATCH (c:Component {id: $cid})
        OPTIONAL MATCH (c)-[:MADE_OF]->(:Material)<-[:MADE_OF]-(a:Component) WHERE a.id <> $cid
        WITH c, collect(DISTINCT a.id) AS same_material
        OPTIONAL MATCH (c)-[:MANUFACTURED_BY]->(:Process)<-[:MANUFACTURED_BY]-(b:Component)
               WHERE b.id <> $cid
        WITH c, same_material, collect(DISTINCT b.id) AS same_process
        OPTIONAL MATCH (c)-[:GOVERNED_BY]->(:SOP)<-[:GOVERNED_BY]-(d:Component) WHERE d.id <> $cid
        RETURN same_material, same_process, collect(DISTINCT d.id) AS same_sop
    """, cid=cid).data()
    return rows[0] if rows else {}


def graph_traversal(state: State) -> dict:
    cid = state["parsed"]["component_id"]
    with DRIVER.session() as s:
        direct = direct_context(s, cid)
        blast = blast_radius(s, cid)
        sub = rag.graph_rag(s, cid, hops=2)          # excludes ECO nodes and GT_IMPACTS
    valid = {i for ids in sub.get("ids", {}).values() for i in ids}
    for k in ("old_id", "new_id"):
        if state["parsed"].get(k):
            valid.add(state["parsed"][k])
    print(f"  [graph_traversal] {len(valid)} candidate nodes in the 2-hop neighbourhood of {cid}")
    return {"direct": direct, "blast": blast, "valid_ids": sorted(valid)}


# ══════════════════════════════════════════════════════════════════════════════
# CONDITIONAL BRANCH
# ══════════════════════════════════════════════════════════════════════════════

def route_agents(state: State) -> list:
    attr = state["parsed"].get("attribute")
    nxt = ["historical_precedent", "sop_compliance"]                   # always
    if attr == "material":
        nxt.append("material_comparison")
    if attr in ("material", "machine", "process", "dimension", "tolerance"):
        nxt.append("process_compatibility")
    print(f"  [router] attribute='{attr}' → running: {', '.join(nxt)}")
    return nxt


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT A – MATERIAL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

MATERIAL_SYS = (
    "You are a materials engineer. Given old vs new material properties (with computed "
    "percentage changes), assess the consequences for a CNC-machined drone part: strength/"
    "weight, machinability, thermal behaviour, cost and CO2. Flag every significant difference "
    "and say whether it raises or lowers risk. Use only the data provided. Be concise "
    "(max ~150 words, bullet points).")


def material_comparison(state: State) -> dict:
    p, ctx = state["parsed"], state["direct"]
    old_id = p.get("old_id") or (ctx.get("material") or {}).get("id")
    new_id = p.get("new_id")
    if not (old_id and new_id):
        return {"material_report": "Material comparison skipped: old/new material IDs not resolved."}
    with DRIVER.session() as s:
        mats = {r["id"]: r["props"] for r in s.run(
            "MATCH (m:Material) WHERE m.id IN $ids RETURN m.id AS id, properties(m) AS props",
            ids=[old_id, new_id])}
    old, new = mats.get(old_id), mats.get(new_id)
    if not (old and new):
        return {"material_report": f"Material comparison skipped: {old_id} or {new_id} not in graph."}
    deltas = {}
    for k in MATERIAL_KEYS:
        o, n = old.get(k), new.get(k)
        if isinstance(o, (int, float)) and isinstance(n, (int, float)):
            deltas[k] = {"old": o, "new": n,
                         "pct_change": round(100 * (n - o) / o, 1) if o else None}
    text = llm_text(MATERIAL_SYS, J({"component": p["component_id"], "old_material": old_id,
                                     "new_material": new_id, "property_deltas": deltas,
                                     "other_components_made_of_old_material":
                                         state["blast"].get("same_material", [])}))
    return {"material_report": text, "material_deltas": deltas}


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT B – PROCESS COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════════════

PROCESS_SYS = (
    "You are a manufacturing process engineer. Judge whether the existing process and machines "
    "can handle the proposed change, given machine capability (Cpk; below 1.33 is marginal), "
    "maintenance status, the component's tolerance and, if relevant, machinability of the new "
    "material or the capability of a replacement machine. Name specific machine IDs that are at "
    "risk. Use only the data provided. Be concise (max ~150 words, bullet points).")


def process_compatibility(state: State) -> dict:
    p, ctx = state["parsed"], state["direct"]
    machines = ctx.get("machines", [])
    flags = []
    for m in machines:
        if isinstance(m.get("cpk"), (int, float)) and m["cpk"] < 1.33:
            flags.append(f"{m['id']}: Cpk {m['cpk']} is below 1.33")
        if m.get("maintenance_status") in ("OVERDUE", "DUE_SOON"):
            flags.append(f"{m['id']}: maintenance {m['maintenance_status']}")

    extra = {}
    with DRIVER.session() as s:
        if p.get("attribute") == "material" and p.get("new_id"):
            extra["new_material"] = (s.run("MATCH (m:Material {id:$id}) RETURN properties(m) AS p",
                                           id=p["new_id"]).data() or [{}])[0].get("p")
        if p.get("attribute") == "machine" and p.get("new_id"):
            extra["replacement_machine"] = (s.run("MATCH (m:Machine {id:$id}) RETURN properties(m) AS p",
                                                  id=p["new_id"]).data() or [{}])[0].get("p")
    comp = ctx.get("component", {})
    text = llm_text(PROCESS_SYS, J({
        "change": p, "process": ctx.get("process"), "machines": machines,
        "machine_risk_flags": flags,
        "component_tolerance_mm": comp.get("nominal_tolerance_mm"),
        "quality_requirements": ctx.get("qrs"),
        "current_material_machinability": (ctx.get("material") or {}).get("machinability_index"),
        **extra}))
    return {"process_report": text}


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT C – HISTORICAL PRECEDENT  (hybrid RAG over past ECOs)
# ══════════════════════════════════════════════════════════════════════════════

PRECEDENT_SYS = (
    "You are a change-control analyst. Using ONLY the past ECOs provided, say what happened in "
    "similar changes: change type, recorded severity and impact dimension. State how similar "
    "each precedent really is and what it suggests about this change. If nothing relevant was "
    "found, say so plainly. Be concise (max ~150 words, bullet points).")


def historical_precedent(state: State) -> dict:
    p = state["parsed"]
    query = f"{p.get('summary', '')} {state['request']}"[:500]
    try:
        with DRIVER.session() as s:
            res = rag.hybrid(s, p["component_id"], query, hops=2, k=12,
                             exclude_eco=state.get("eco_id"))
            hits = [h for h in res["passages"] if h["doc_type"] == "ECO"][:5]
            meta = {}
            if hits:
                meta = {r["id"]: r for r in s.run(
                    "MATCH (e:ECO) WHERE e.id IN $ids RETURN e.id AS id, e.change_type AS change_type, "
                    "e.direct_severity AS direct_severity, e.impact_dimension AS impact_dimension, "
                    "e.status AS status", ids=[h["node_id"] for h in hits]).data()}
    except Exception as ex:
        return {"precedent_report": f"Precedent search unavailable ({ex}). "
                                    f"Run `python src/rag.py index` first.", "precedents": []}
    precedents = [{"id": h["node_id"], "text": h["text"][:600],
                   **{k: v for k, v in meta.get(h["node_id"], {}).items() if k != "id"}}
                  for h in hits]
    if not precedents:
        return {"precedent_report": "No similar past ECOs found for this component's neighbourhood.",
                "precedents": []}
    text = llm_text(PRECEDENT_SYS, J({"current_change": p, "past_ecos": precedents}))
    return {"precedent_report": text, "precedents": precedents}


# ══════════════════════════════════════════════════════════════════════════════
# SUB-AGENT D – SOP / COMPLIANCE
# ══════════════════════════════════════════════════════════════════════════════

SOP_SYS = (
    "You are a quality & compliance engineer. Using ONLY the governing SOPs, quality "
    "requirements and SOP passages provided, decide whether the change could violate an SOP or "
    "quality requirement, or whether any SOP needs revision/re-validation. Cite SOP and QR IDs "
    "and their risk class/revision. If nothing is affected, say so. Be concise (max ~150 words, "
    "bullet points).")


def sop_compliance(state: State) -> dict:
    p, ctx = state["parsed"], state["direct"]
    sops, qrs = ctx.get("sops", []), ctx.get("qrs", [])
    passages = []
    if sops:
        try:
            query = f"{p.get('summary', '')} {state['request']}"[:500]
            passages = [{"sop": h["node_id"], "text": h["text"][:600]} for h in rag.doc_rag(
                query, k=4, where={"node_id": {"$in": [s["id"] for s in sops]}})]
        except Exception:
            passages = []
    text = llm_text(SOP_SYS, J({"change": p, "governing_sops": sops,
                                "quality_requirements": qrs, "sop_passages": passages,
                                "other_components_under_same_sop":
                                    state["blast"].get("same_sop", [])}))
    return {"sop_report": text}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 4 – SYNTHESIS / REASONING  (ranked impact assessment)
# ══════════════════════════════════════════════════════════════════════════════

ASSESS_TOOL = {
    "name": "record_assessment",
    "description": "Record the ranked impact assessment for the change.",
    "input_schema": {
        "type": "object",
        "properties": {
            "overall_risk": {"type": "string", "enum": ["High", "Medium", "Low"]},
            "overall_confidence": {"type": "number", "description": "0 to 1"},
            "summary": {"type": "string"},
            "impacts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "node_type": {"type": "string", "enum": IMPACT_TYPES},
                        "node_id": {"type": "string"},
                        "severity": {"type": "string", "enum": ["High", "Medium", "Low"]},
                        "confidence": {"type": "number", "description": "0 to 1"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["node_type", "node_id", "severity", "confidence", "rationale"],
                },
            },
            "recommended_actions": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["overall_risk", "overall_confidence", "summary", "impacts",
                     "recommended_actions"],
    },
}

SYNTH_SYS = (
    "You are the lead change-impact engineer. Combine the sub-agent findings and graph facts "
    "into ONE ranked impact assessment. Rules:\n"
    "- Only cite node IDs from VALID_IDS (exact spelling). Never invent IDs.\n"
    "- Include the changed component itself, plus every directly affected process, machine, SOP, "
    "material, supplier or quality requirement; add other components only when the evidence shows "
    "they are really affected (shared material/process/SOP).\n"
    "- Severity High/Medium/Low must follow the evidence; confidence (0-1) reflects how well "
    "the evidence supports it. Each rationale: 1-2 sentences grounded in the findings.\n"
    "- Do not claim impacts the evidence does not support. Call record_assessment once.")


def synthesis(state: State) -> dict:
    payload = {
        "VALID_IDS": state["valid_ids"],
        "change": state["parsed"],
        "component_context": state["direct"],
        "blast_radius": state["blast"],
        "material_findings": state.get("material_report", "Not run (material not changed)."),
        "process_findings": state.get("process_report", "Not run."),
        "precedent_findings": state.get("precedent_report", "Not run."),
        "sop_findings": state.get("sop_report", "Not run."),
    }
    result = llm_tool(SYNTH_SYS, J(payload), ASSESS_TOOL, max_tokens=3000)
    valid = set(state["valid_ids"])
    kept, dropped = [], 0
    for i in result.get("impacts", []):
        if i.get("node_id") in valid and i.get("node_type") in IMPACT_TYPES:
            kept.append(i)
        else:
            dropped += 1            # hallucination guard: unknown IDs are discarded
    kept.sort(key=lambda i: (SEVERITY_ORDER.get(i["severity"], 3), -float(i.get("confidence", 0))))
    result["impacts"] = kept
    return {"assessment": result, "dropped": dropped}


# ══════════════════════════════════════════════════════════════════════════════
# NODE 5 – REPORT GENERATOR  (Markdown)
# ══════════════════════════════════════════════════════════════════════════════

def report_node(state: State) -> dict:
    a, p = state["assessment"], state["parsed"]
    esc = lambda t: str(t).replace("|", "/").replace("\n", " ")

    md = [f"# Engineering Change Impact Report",
          f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", "",
          "## Change request", f"> {state['request']}", "",
          "## Parsed intent",
          f"- **Component:** `{p.get('component_id')}`",
          f"- **Attribute:** {p.get('attribute')}",
          f"- **Change:** {p.get('old_value', '?')} → {p.get('new_value', '?')}",
          f"- **Summary:** {p.get('summary')}", "",
          "## Overall assessment",
          f"- **Overall risk:** {a.get('overall_risk')}  ·  **Confidence:** "
          f"{float(a.get('overall_confidence', 0)):.2f}",
          f"- {a.get('summary')}", "",
          "## Ranked impacts", "",
          "| # | Severity | Type | Node | Confidence | Rationale |",
          "|---|----------|------|------|-----------|-----------|"]
    for n, i in enumerate(a.get("impacts", []), 1):
        md.append(f"| {n} | {i['severity']} | {i['node_type']} | `{i['node_id']}` | "
                  f"{float(i.get('confidence', 0)):.2f} | {esc(i['rationale'])} |")
    md += ["", "## Recommended actions"]
    md += [f"{n}. {act}" for n, act in enumerate(a.get("recommended_actions", []), 1)] or ["_None._"]

    md += ["", "## Sub-agent findings"]
    for title, key in [("Material comparison", "material_report"),
                       ("Process compatibility", "process_report"),
                       ("Historical precedent", "precedent_report"),
                       ("SOP / compliance", "sop_report")]:
        md += [f"### {title}", state.get(key, "_Not run for this change (conditional branch skipped)._"), ""]

    if state.get("material_deltas"):
        md += ["### Material property deltas", "", "| Property | Old | New | Change % |", "|---|---|---|---|"]
        for k, d in state["material_deltas"].items():
            md.append(f"| {k} | {d['old']} | {d['new']} | {d['pct_change']} |")
        md.append("")

    md += ["## Provenance",
           f"- Graph candidate nodes considered: {len(state['valid_ids'])}",
           f"- Impacts discarded as not present in the graph: {state.get('dropped', 0)}",
           f"- Past ECOs retrieved: {len(state.get('precedents', []))}",
           "- Data sources: Neo4j knowledge graph (GT_IMPACTS answer key not used), vector index"]
    text = "\n".join(md)

    out_dir = ROOT / "reports"
    out_dir.mkdir(exist_ok=True)
    name = state.get("eco_id") or "adhoc"
    path = out_dir / f"impact_{name}_{datetime.now():%Y%m%d_%H%M%S}.md"
    path.write_text(text, encoding="utf-8")
    return {"report_md": text, "report_path": str(path)}


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    g = StateGraph(State)
    g.add_node("parse_change", parse_change)
    g.add_node("graph_traversal", graph_traversal)
    g.add_node("material_comparison", material_comparison)
    g.add_node("process_compatibility", process_compatibility)
    g.add_node("historical_precedent", historical_precedent)
    g.add_node("sop_compliance", sop_compliance)
    g.add_node("synthesis", synthesis)
    g.add_node("report", report_node)

    g.add_edge(START, "parse_change")
    g.add_conditional_edges("parse_change", after_parse,
                            {"graph_traversal": "graph_traversal", END: END})
    g.add_conditional_edges("graph_traversal", route_agents,
                            ["material_comparison", "process_compatibility",
                             "historical_precedent", "sop_compliance"])
    # fan-in: every branch that ran feeds the synthesis node
    for n in ("material_comparison", "process_compatibility",
              "historical_precedent", "sop_compliance"):
        g.add_edge(n, "synthesis")
    g.add_edge("synthesis", "report")
    g.add_edge("report", END)
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION  (the ONLY place GT_IMPACTS is read — after the agent has finished)
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(eco_id: str, impacts: list):
    with DRIVER.session() as s:
        gt = {(r["t"], r["id"]): r["sev"] for r in s.run(
            "MATCH (e:ECO {id:$id})-[g:GT_IMPACTS]->(n) "
            "RETURN labels(n)[0] AS t, n.id AS id, g.severity AS sev", id=eco_id)}
    scored_types = {"Component", "Process", "Machine", "SOP"}   # types the answer key covers
    pred = {(i["node_type"], i["node_id"]) for i in impacts if i["node_type"] in scored_types}
    truth = set(gt)
    tp = pred & truth
    prec = len(tp) / len(pred) if pred else 0.0
    rec = len(tp) / len(truth) if truth else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    print("\n── EVALUATION vs GT_IMPACTS (answer key) ──")
    print(f"  Ground-truth impacted nodes : {len(truth)}")
    print(f"  Agent-predicted nodes       : {len(pred)}")
    print(f"  Correct                     : {len(tp)}")
    print(f"  Precision {prec:.2f} · Recall {rec:.2f} · F1 {f1:.2f}")
    missed = sorted(truth - pred)
    extra = sorted(pred - truth)
    if missed: print(f"  Missed : {missed[:10]}{' …' if len(missed) > 10 else ''}")
    if extra:  print(f"  Extra  : {extra[:10]}{' …' if len(extra) > 10 else ''}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    global DRIVER, CLIENT
    ap = argparse.ArgumentParser(description="Multi-agent ECO impact analysis (LangGraph)")
    ap.add_argument("--request", help="free-text change request")
    ap.add_argument("--eco", help="use an existing ECO's free-text description, e.g. ECO-000002")
    ap.add_argument("--eval", action="store_true", help="score against GT_IMPACTS after the run (needs --eco)")
    args = ap.parse_args()
    if not (args.request or args.eco):
        ap.error("provide --request or --eco")
    if args.eval and not args.eco:
        ap.error("--eval needs --eco")
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("[ERROR] ANTHROPIC_API_KEY missing — add it to .env")

    DRIVER = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        DRIVER.verify_connectivity()
    except Exception as e:
        sys.exit(f"[ERROR] Cannot connect to Neo4j: {e}")
    CLIENT = anthropic.Anthropic()

    request = args.request
    if args.eco:
        with DRIVER.session() as s:
            rows = s.run("MATCH (e:ECO {id:$id}) RETURN e.description AS d", id=args.eco).data()
        if not rows or not rows[0]["d"]:
            sys.exit(f"ECO {args.eco} not found or has no description.")
        request = rows[0]["d"]
        print(f"ECO {args.eco}: {request}\n")

    print("Running the agent graph …")
    final = build_app().invoke({"request": request, "eco_id": args.eco or ""})

    if final.get("error"):
        print(f"\n[STOPPED] {final['error']}")
    else:
        print("\n" + final["report_md"])
        print(f"\nReport saved to {final['report_path']}")
        if args.eval:
            evaluate(args.eco, final["assessment"].get("impacts", []))

    DRIVER.close()


if __name__ == "__main__":
    main()
