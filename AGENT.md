# AGENT.MD - Operational Manual

## CRITICAL DIRECTIVES (READ FIRST)
1. **NO HALLUCINATION:** Do NOT assume, guess, or invent any API interfaces, dataset schemas, class methods, or library behaviors. 
2. **READ BEFORE WRITE:** You must read documentation, source code, or actual data schemas BEFORE writing implementation code.
3. **FAITHFUL ADAPTATION:** When integrating external repositories, preserve the original architecture. Refactor only what is strictly necessary to meet our project's requirements.
4. **PHASE LOCK:** Do not jump ahead. Complete the current phase, verify outputs, and update this file before moving to the next phase.

---

## Project Overview
**Repository:** `llm-vs-rag-bench`
**Goal:** Build a research CLI tool that benchmarks an Agentic Retrieval architecture (LLM-Wiki-Agent) against a traditional RAG pipeline. The system must allow dynamic selection of N documents and M questions, run both architectures, benchmark results, and export the agent's trajectories for future Supervised Fine-Tuning (SFT).

---

## LLM Configuration
All LLM calls (Agent, RAG generation, LLM-as-Judge) MUST use the following custom OpenAI-compatible endpoint:
- `OPENAI_BASE_URL=http://az.gptplus5.com/v1`
- `MODEL_NAME=gemini-3-flash-preview`
- `EMBEDDING_MODEL_NAME=text-embedding-3-small` (or compatible via the same base URL)

---

## External Dependencies & Anti-Hallucination Rules

### 1. LLM-Wiki-Agent (Agentic Architecture)
- **Source:** `https://github.com/SamurAIGPT/llm-wiki-agent`
- **Rule:** This must be an EXACT architectural clone. Do NOT rewrite the agent from scratch.
- **Action Required:** 
  1. Clone the repo.
  2. Read and analyze its file structure, agent loop, tool definitions, and prompt templates.
  3. Refactor ONLY the LLM client initialization to use our custom endpoint.
  4. Replace its default knowledge source (Wikipedia) with our document corpus, matching its existing retrieval interface.

### 2. UniDoc-Bench (Dataset)
- **Source:** `https://huggingface.co/datasets/Salesforce/UniDoc-Bench` and `https://github.com/SalesforceAIResearch/UniDoc-Bench`
- **Rule:** Do NOT assume the dataset schema (column names, data types, splits).
- **Action Required:**
  1. Write a script to load the dataset using the `datasets` library.
  2. Print and inspect the schema, splits, and 2-3 sample rows.
  3. Identify exactly how the "healthcare" subset is filtered.
  4. Identify exactly which columns represent: Context/Document, Question, and Ground Truth Answer.
  5. Build `src/data/` based STRICTLY on the observed schema.

### 3. RAG Pipeline
- **Rule:** Design the chunking and retrieval strategy based on the ACTUAL size and format of the documents observed in the UniDoc-Bench dataset. Do not assume document length.

---

## Target Directory Structure
```text
llm-vs-rag-bench/
├── AGENT.md           
├── requirements.txt
├── .env.example
├── main.py             <-- CLI entrypoint (Phase 7)
├── docs/               <-- Research & Analysis (CRITICAL: Read these before coding)
│   ├── wiki_agent_analysis.md
│   ├── dataset_analysis.md
│   └── rag_design.md
├── scripts/            <-- Standalone test/inspection scripts
├── src/
│   ├── config.py       <-- Env vars and settings
│   ├── llm_client.py   <-- Custom OpenAI wrapper
│   ├── llm-wiki/       <-- Refactored LLM-Wiki-Agent code
│   ├── rag/            # RAG pipeline code
│   ├── data/           # Dataset loading and processing
│   └── evaluation/     # Benchmarking and metrics logic
├── results/            # Where CSV reports are saved
└── trajectories/       # Where JSONL training data is saved
```

---

## Phased Execution Plan

### Phase 0: Research & Scaffolding
**Objective:** Read all external sources. Output NO implementation code except directory structure, `requirements.txt`, `.env.example`, and inspection scripts. 
**Deliverables:**
- `docs/wiki_agent_analysis.md` (Classes, methods, agent loop logic, LLM call patterns)
- `docs/dataset_analysis.md` (Schema, column names, healthcare filter, sample data)
- `docs/rag_design.md` (Chunking and vector store choices based on actual data size)
- `scripts/inspect_dataset.py`

### Phase 1: Core Infrastructure
**Objective:** Build `src/config.py` and `src/llm_client.py` based on the LLM call patterns discovered in Phase 0.
**Requirement:** The LLM client must be a drop-in replacement (or wrapper) for how the original llm-wiki-agent calls the LLM.

### Phase 2: Dataset Module
**Objective:** Build `src/data/dataset_loader.py` and `src/data/models.py` based on the ACTUAL schema discovered in Phase 0.
**Requirement:** All field names and types must match the HuggingFace dataset exactly.

### Phase 3: LLM-Wiki-Agent Adaptation
**Objective:** Clone and adapt `llm-wiki-agent` into `src/llm-wiki/`.
**Requirement:** Preserve original logic. Change only the LLM client and the knowledge source interface.

### Phase 4: RAG Pipeline
**Objective:** Build `src/rag/` based on the design doc in Phase 0.
**Requirement:** Must handle the actual document size/format and use our custom LLM/Embedding endpoint.

### Phase 5: Evaluation Module [COMPLETED]
**Objective:** Build LLM-as-Judge and metrics calculator in `src/evaluation/`.
**Requirement:** Judge prompt must score 1-5. Metrics must compute mean/median score, latency, token usage, retrieval count.
**Deliverables:**
- `src/evaluation/judge.py` — LLMJudge class with rigorous prompt, parses response reliably
- `src/evaluation/metrics.py` — MetricsCalculator and ArchitectureMetrics dataclass
- `src/evaluation/report.py` — ReportGenerator creates DataFrame, saves CSV, prints to console
- `scripts/test_evaluation.py` — Test script with mock BenchmarkResults

### Phase 6: Trajectory Export [COMPLETED]
**Objective:** Build JSONL exporter for SFT in `src/trajectory/`.
**Requirement:** Format must strictly follow: `{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
**Deliverables:**
- `src/trajectory/exporter.py` — TrajectoryExporter class that converts Trajectory dataclass to OpenAI JSONL format
- `src/trajectory/__init__.py` — Module exports
- `scripts/test_trajectory_export.py` — Test script validating export against actual agent output patterns

### Phase 7: CLI Integration [COMPLETED]
**Objective:** Wire everything together in `main.py` using `typer`.
**Requirement:** Prompt user for N docs and M questions. Run both pipelines. Output comparative report and save trajectories. Include `--dry-run` and `--verbose` flags.
**Deliverables:**
- `main.py` — CLI entrypoint with typer commands: benchmark, inspect-dataset, test-llm
- Updated `requirements.txt` with typer dependency

### Phase 8: [NEXT]
**Objective:** TBD

---

## How to Proceed
If you are starting a new session, read this file first. 
1. Check the "Phased Execution Plan" to see the current phase.
2. Read the `docs/` directory to load the context from previous phases.
3. Execute ONLY the tasks for the current phase.
4. Do NOT move to the next phase until the human approves the output.

***

### How to use this effectively:

1. **Update the Phase Tracker:** When you finish Phase 0, change `[CURRENT]` to Phase 1, and mark Phase 0 as `[COMPLETED]`. This tells the AI exactly where to pick up.
