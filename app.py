"""
app.py
PharmaBridge - Context-Preserving Technology Transfer demo.
Run with: streamlit run app.py
"""

import json
import streamlit as st
from pydantic import ValidationError

from schemas import ProcessKnowledgePackage
from validation import validate_package
from agents import run_pm_agent, run_sa_agent

st.set_page_config(page_title="PharmaBridge", page_icon="🧪", layout="wide")

PRESETS = {
    "✅ Clean handoff (should pass)": "Transfer our pilot-scale tablet manufacturing process to mass "
        "production. Temperature target 52°C, validated pilot range 51-52°C, high criticality, "
        "affects product quality and dissolution rate, high scale sensitivity, evidenced by pilot "
        "validation data, updated 2026-01-15.",
    "🚫 Missing criticality (should BLOCK)": "Transfer our tablet process to mass production. missing criticality "
        "Temperature target 52°C, range 51-52°C.",
    "🚫 Conflicting values (should BLOCK)": "Transfer our tablet process. conflict Temperature target 52°C but "
        "risk assessment caps it at 45°C, range 51-52°C, high criticality.",
    "⚠️ Pilot-only evidence (warns, doesn't block)": "Transfer our tablet process to mass production. "
        "Temperature 52°C, range 51-52°C, high criticality, evidenced only by small-scale pilot blender runs.",
    "⚠️ Stale data (warns, doesn't block)": "Transfer our tablet process. stale Temperature 52°C range 51-52°C "
        "high criticality, formula last validated years ago.",
    "⚠️ Looks complete but no rationale (warns)": "Transfer our tablet process. no rationale Temperature 52°C "
        "range 51-52°C high criticality, no explanation given for why.",
}

st.title("🧪 PharmaBridge")
st.caption("Context-Preserving Technology Transfer — R&D → Manufacturing")

with st.sidebar:
    st.header("Demo controls")
    mock_mode = st.toggle("Offline / Mock mode (no API calls)", value=True,
                           help="Use this if venue WiFi is unreliable. Turn off to hit real Claude + GPT-4o.")
    if not mock_mode:
        st.info("Live mode needs ANTHROPIC_API_KEY and OPENAI_API_KEY set as environment variables.")
    st.divider()
    st.header("Preset scenarios")
    preset_choice = st.selectbox("Load a scenario", ["(custom input)"] + list(PRESETS.keys()))

default_text = PRESETS.get(preset_choice, "") if preset_choice != "(custom input)" else ""

client_text = st.text_area(
    "Client requirement / messy R&D notes",
    value=default_text,
    height=120,
    placeholder="e.g. 'Transfer our lab-scale tablet manufacturing process to mass production...'",
)

run = st.button("▶ Run handoff pipeline", type="primary", use_container_width=True)

if run:
    if not client_text.strip():
        st.warning("Enter or select a client requirement first.")
        st.stop()

    # ---------------- Agent 1 ----------------
    st.subheader("① Agent 1 — Product Manager (Claude 3.5 Sonnet)")
    with st.spinner("Reading source text and extracting Process Knowledge Package..."):
        try:
            pm_raw = run_pm_agent(client_text, mock=mock_mode)
        except Exception as e:
            st.error(f"Agent 1 call failed: {e}")
            st.stop()

    st.json(pm_raw)

    try:
        pkg = ProcessKnowledgePackage(**pm_raw)
    except ValidationError as e:
        st.error("Agent 1 output did not match the required schema. Handoff rejected before validation even runs.")
        st.code(str(e))
        st.stop()

    # ---------------- Validation layer ----------------
    st.subheader("② Deterministic Validation Layer (pure Python — no LLM)")
    result = validate_package(pkg)

    if result.blocking_issues:
        for issue in result.blocking_issues:
            st.error(f"🚫 **{issue.code}** — {issue.message}")

    if result.warnings:
        for issue in result.warnings:
            st.warning(f"⚠️ **{issue.code}** — {issue.message}")

    if not result.blocking_issues and not result.warnings:
        st.success("✅ No issues detected. Package is clean.")

    if not result.passed:
        st.error("### HANDOFF BLOCKED — Agent 2 will not run.")
        st.info("This is the point of the system: incomplete context doesn't become a wrong "
                 "decision downstream, it becomes a controlled stop here.")
        st.stop()
    else:
        st.success("Validation passed" + (" with flagged risks." if result.warnings else "."))

    # ---------------- Agent 2 ----------------
    st.subheader("③ Agent 2 — Solution Architect (GPT-4o)")
    st.caption("Receives ONLY the validated JSON below — never the original raw text.")
    with st.spinner("Transforming knowledge package into manufacturing controls..."):
        try:
            sa_raw = run_sa_agent(pkg.model_dump(), mock=mock_mode)
        except Exception as e:
            st.error(f"Agent 2 call failed: {e}")
            st.stop()

    st.json(sa_raw)

    st.subheader("④ Production Blueprint")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Control:** {sa_raw.get('control_instruction')}")
        st.markdown(f"**Monitoring:** {sa_raw.get('monitoring')}")
        st.markdown(f"**Validation:** {sa_raw.get('validation_requirement')}")
    with c2:
        st.markdown(f"**Deviation handling:** {sa_raw.get('deviation_handling')}")
        st.markdown(f"**Risk assessment:** {sa_raw.get('risk_assessment')}")
        st.markdown(f"**Traceability:** {sa_raw.get('traceability')}")

    if sa_raw.get("open_risk_flags"):
        st.warning("**Open risk flags carried into the blueprint:**\n" +
                    "\n".join(f"- {f}" for f in sa_raw["open_risk_flags"]))

    st.success("✅ Ready for implementation.")
