"""
agents.py
Agent 1 (Product Manager -> Claude 3.5 Sonnet) and
Agent 2 (Solution Architect -> GPT-4o).

Both agents are forced into strict JSON output. Agent 2 NEVER sees the
client's raw text - it only ever receives the validated
ProcessKnowledgePackage JSON. That single design choice is your
"context-preserving handoff" in code, not just in a slide.
"""

import json
import os
from schemas import ProcessKnowledgePackage, ManufacturingDesign

PM_SYSTEM_PROMPT = """You are the Product Manager Agent in a pharmaceutical technology-transfer system.
You read messy R&D notes, lab reports, or plain-language client requests describing a process being
transferred from pilot/lab scale to mass manufacturing.

Extract ONE critical process parameter as a Process Knowledge Package. Respond with ONLY valid JSON,
no markdown fences, no commentary, matching exactly this shape:

{
  "parameter": string,
  "target_value": number or null,
  "unit": string or null,
  "validated_range_low": number or null,
  "validated_range_high": number or null,
  "criticality": "High" | "Medium" | "Low" or null,
  "quality_impact": string or null,
  "scale_sensitivity": "High" | "Medium" | "Low" or null,
  "evidence": string or null,
  "evidence_scale": "pilot" | "commercial" or null,
  "uncertainty": string or null,
  "risk_cap": number or null,
  "dependencies": array of strings,
  "last_updated": ISO date string or null,
  "safety_rationale": string or null
}

Rules:
- If the source text does not state a field, output null (or [] for dependencies). NEVER invent a value.
- Do not resolve conflicts in the source text yourself - report both values as given
  (e.g. put the risk-assessment cap in risk_cap even if it disagrees with the stated target).
- Preserve the WHY behind numbers whenever the source gives one, in safety_rationale.
"""

SA_SYSTEM_PROMPT = """You are the Solution Architect Agent in a pharmaceutical technology-transfer system.
You receive ONLY a validated Process Knowledge Package as JSON - never raw notes, never client text.
Treat it as ground truth exactly as written. Do not add information that is not implied by the input.
Do not creatively fill gaps. If a field is null, state that explicitly as an open item rather than guessing.

Respond with ONLY valid JSON, no markdown fences, no commentary, matching exactly this shape:

{
  "control_instruction": string,
  "monitoring": string,
  "validation_requirement": string,
  "deviation_handling": string,
  "risk_assessment": string,
  "traceability": string,
  "open_risk_flags": array of strings
}
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def run_pm_agent(client_text: str, mock: bool = False) -> dict:
    """Agent 1: Claude 3.5 Sonnet turns messy text into a Process Knowledge Package."""
    if mock:
        return _mock_pm_output(client_text)

    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1000,
        system=PM_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": client_text}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    return json.loads(_strip_fences(raw))


def run_sa_agent(pkg_json: dict, mock: bool = False) -> dict:
    """Agent 2: GPT-4o turns a validated Process Knowledge Package into a manufacturing design."""
    if mock:
        return _mock_sa_output(pkg_json)

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    resp = client.chat.completions.create(
        model="gpt-4o",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SA_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(pkg_json)},
        ],
    )
    return json.loads(resp.choices[0].message.content)


# ---------------------------------------------------------------------------
# Mock mode: deterministic, no network calls. Used if WiFi/API keys fail live.
# ---------------------------------------------------------------------------

def _mock_pm_output(client_text: str) -> dict:
    """Very small heuristic 'mock LLM' so the demo still runs offline."""
    text = client_text.lower()
    pkg = {
        "parameter": "Temperature",
        "target_value": 52,
        "unit": "°C",
        "validated_range_low": 51,
        "validated_range_high": 52,
        "criticality": "High",
        "quality_impact": "Product quality",
        "scale_sensitivity": "High",
        "evidence": "Pilot validation data",
        "evidence_scale": "pilot",
        "uncertainty": "Industrial-scale behaviour unvalidated",
        "risk_cap": None,
        "dependencies": ["Dissolution rate"],
        "last_updated": "2026-01-15",
        "safety_rationale": "Above 52°C the coating layer degrades before compression.",
    }
    if "missing criticality" in text:
        pkg["criticality"] = None
        pkg["scale_sensitivity"] = None
    if "conflict" in text:
        pkg["target_value"] = 52
        pkg["risk_cap"] = 45
    if "stale" in text:
        pkg["last_updated"] = "2019-03-01"
    if "no rationale" in text:
        pkg["safety_rationale"] = None
        pkg["uncertainty"] = None
    return pkg


def _mock_sa_output(pkg_json: dict) -> dict:
    low = pkg_json.get("validated_range_low")
    high = pkg_json.get("validated_range_high")
    unit = pkg_json.get("unit") or ""
    flags = []
    if pkg_json.get("evidence_scale") == "pilot":
        flags.append("Commercial-scale behaviour unverified - pilot evidence only.")
    if pkg_json.get("dependencies"):
        flags.append(f"Linked quality attributes at risk: {', '.join(pkg_json['dependencies'])}.")
    return {
        "control_instruction": f"Maintain {pkg_json.get('parameter', 'parameter')} within {low}-{high}{unit}.",
        "monitoring": "Continuous monitoring with automated deviation alerts.",
        "validation_requirement": "Full-scale validation run required before production release.",
        "deviation_handling": "Flag any excursion outside validated range for immediate quality review.",
        "risk_assessment": f"Criticality: {pkg_json.get('criticality', 'Unknown')}. Quality impact: {pkg_json.get('quality_impact', 'Unknown')}.",
        "traceability": f"Linked to source evidence: {pkg_json.get('evidence', 'Unknown')}.",
        "open_risk_flags": flags,
    }
