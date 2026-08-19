# PharmaBridge — Round 2 Demo

Context-preserving R&D → Manufacturing handoff. Two real LLM agents plus a
deterministic Python validation layer that can block a bad handoff.

## 1. Setup (do this before the day, at home)

```bash
cd pharmabridge_demo
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get API keys:
- Anthropic: https://console.anthropic.com (for Agent 1 — Claude 3.5 Sonnet)
- OpenAI: https://platform.openai.com (for Agent 2 — GPT-4o)

Set them as environment variables before running live mode:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

(Windows PowerShell: `$env:ANTHROPIC_API_KEY="sk-ant-..."`)

## 2. Run it

```bash
streamlit run app.py
```

Opens at http://localhost:8501

## 3. Two modes — use both strategically

- **Mock/offline mode (default, toggle in sidebar ON):** No network calls,
  runs instantly, deterministic. Use this if venue WiFi is bad — it still
  proves the *pipeline logic* end-to-end, including the blocking behaviour.
- **Live mode (toggle OFF):** Real Claude + real GPT-4o calls. Use this once,
  early in your demo, to prove it's genuinely AI-integrated — then switch
  back to mock for the rest of the walkthrough so you're not at the mercy of
  API latency in front of judges.

**Recommended sequence for judges:** run one scenario live (proves it's
real AI), then flip to mock and run the edge-case presets (proves the
innovation — the blocking logic — reliably, every time, in seconds).

## 4. Preset scenarios (sidebar dropdown)

| Preset | What it proves |
|---|---|
| Clean handoff | Full pipeline runs end-to-end, produces a blueprint |
| Missing criticality | Validation layer BLOCKS before Agent 2 runs |
| Conflicting values | Validation layer catches PM target vs risk-assessment cap disagreeing |
| Pilot-only evidence | Warns but doesn't block — scale gap surfaces as an open risk flag in the final blueprint |
| Stale data | Warns that source info is outdated, doesn't silently treat it as current |
| Looks complete but no rationale | Catches the "perfect-looking but incomplete" case from your deck |

## 5. 5-minute pitch script

1. **(30s) Problem:** "1 in 4 batches fail at scale-up not because the process
   was wrong, but because the *reasoning behind it* got lost in handoff."
2. **(60s) Run the clean scenario live** (real API mode). Show Agent 1's JSON,
   then Agent 2's manufacturing blueprint. Point out: "Agent 2 never sees the
   client's original messy text — only this structured package. That's the
   context-preserving handoff."
3. **(90s) Switch to mock mode. Run "Missing criticality."** Show it BLOCKS.
   Say: "This is the part that makes this more than an Agent 1 → Agent 2
   chain — Agent 2 is designed to know when Agent 1 shouldn't be trusted yet."
4. **(60s) Run "Conflicting values."** Show the explicit contradiction being
   caught instead of silently resolved by an LLM guessing.
5. **(30s) Close on limitations slide honestly:** "This doesn't replace GMP
   review or lab validation — it guarantees nothing gets there without the
   reasoning attached."

## 6. If something breaks on stage

- API key issue / no WiFi → flip mock mode ON, keep going, nobody needs to know.
- A preset doesn't demo the point clearly → type the scenario manually using
  the keywords `missing criticality`, `conflict`, `stale`, `no rationale` in
  your input text (the mock agent looks for these keywords — see `agents.py`
  `_mock_pm_output`).
