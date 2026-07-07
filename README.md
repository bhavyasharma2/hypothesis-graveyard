# 🪦 Hypothesis Graveyard

**Four AI agents that surface what your team already knows before you waste weeks rediscovering it.**

> Built for quant finance researchers, ML engineers, data scientists, and software engineering teams.

---

## The Problem

Knowledge-intensive teams have a memory problem.

A hypothesis gets tested. It fails. The lessons get buried in a closed ticket, a forgotten notebook, a Slack thread nobody scrolls back to. The person who ran the experiment moves on. The knowledge moves with them.

Six months later, someone new proposes the same idea. The team has no way to know it was already tried. No way to know why it failed. No way to surface the two weeks of work that already answered this exact question.

This cycle repeats constantly across quant finance teams, ML research labs, data science orgs, and software engineering teams everywhere. The cost is not just time. It is the compounding loss of every lesson a team ever learned, silently draining the organization of its own hard-won experience.

The tools that exist today do not solve this. Wikis require someone to maintain them. Search bars find keywords not meaning. Postmortem docs sit unread until after the mistake is already made.

**Hypothesis Graveyard fixes that.**

---

## The Solution

Hypothesis Graveyard is a multi-agent system that maintains semantic memory of every hypothesis a team has ever tested, and surfaces that memory at the exact moment someone is about to repeat a mistake.

It does not wait to be asked. The moment you propose an idea, four specialized agents analyze it, search your team's history, build the strongest case for it, and deliver a calibrated verdict before a single hour of work is wasted.

---

## Architecture

```
User Input
    │
    ▼
🕵️ Capture Agent ──── Validates input, blocks prompt injection
    │
    ▼
🏛️ Archaeology Agent ── Searches ChromaDB for semantically similar past attempts
    │                          ▲
    ▼                          │
🛡️ Steelman Agent ──── Builds strongest version using google_search MCP tool
    │
    ▼
⚖️ Verdict Agent ───── Delivers conviction score (0-100) anchored in history
    │
    ▼
Conviction Score + Verdict
    │
    ▼ (if buried by user)
ChromaDB Vector Store ◄──── Graveyard grows, system gets smarter
```

---

## Key Features

### Deja Vu Alert
Fires instantly when a new hypothesis similarity exceeds 73% of a past attempt. Shows who tested it, when, and what happened — before the full analysis even completes.

### Semantic Paraphrase Detection
The system catches ideas, not keywords. During testing, two completely differently worded hypotheses about the same concept matched at 96% similarity with zero shared words.

### Calibrated Verdicts
The conviction score is anchored in historical precedent. The same system returned conviction 90 for a reworded successful hypothesis and conviction 45 for a reworded failed one. The score shifts based on what actually happened, not how compelling the argument sounds.

### Team Mode
Every buried hypothesis shows contributor name and date. Institutional memory becomes human and attributable.

### Graveyard Map
A Plotly network graph with nodes colored by outcome and edges connecting semantically similar entries. Uses the Wong color blind safe palette for full accessibility.

### Auto-expanding Domains
The Capture Agent automatically recognizes and creates new domain categories. The system self-expanded to include E-commerce/Recommender Systems when a new hypothesis type was submitted.

### Security Guardrails
Three layers: input length validation, gibberish detection, and prompt injection detection. Attempts like "ignore previous instructions" are blocked with a clear error message.

### Auto-seeding
Pre-loaded with 20 rich historical hypotheses across Quant Finance, Software Engineering, and ML/Data Science on first launch. No manual database setup required.

### Accessibility
Wong color blind safe palette on all visualizations. Symbol plus color outcome indicators (✕ Failed, ✓ Success, ○ Proposed, — Abandoned) used throughout.

---

## Tech Stack

| Component | Technology |
|---|---|
| Multi-agent orchestration | Google ADK 1.3.0 |
| Agent reasoning | Gemini 2.5 Flash |
| Semantic search | Gemini Embedding 001 |
| Vector store | ChromaDB |
| Live web evidence | google_search MCP tool |
| Deployment infrastructure | Cloud Run MCP via Antigravity |
| Network visualization | Plotly + NetworkX |
| Web interface | Streamlit |
| Built with | Antigravity IDE (vibe coding) |

---

## Project Demo

A comprehensive end-to-end walkthrough demonstrating every feature, workflow, and capability of the project in detail: https://drive.google.com/file/d/1ALg2CegzKpAhV6pipoxmCmuMPkopb9Yg/view?usp=sharing

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- Free Gemini API key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Step 1: Clone the repository
```bash
git clone https://github.com/bhavyasharma2/hypothesis-graveyard.git
cd hypothesis-graveyard
```

### Step 2: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Add your Gemini API key
Create a `.env` file in the project root:
```
GEMINI_API_KEY=your_key_here
```

### Step 4: Run the app
```bash
streamlit run app.py
```

The graveyard auto-seeds with 20 historical hypotheses on first launch. Open your browser at `http://localhost:8501`.

---

## Demo Scenarios

Try these inputs to see the system in action:

**Failure catch (quant):**
```
We want to build a deep learning model that reads real-time Level 2 order book snapshots to predict where NASDAQ stock prices will move in the next 15 seconds
```

**Deja Vu Alert trigger:**
```
What if we combined drone footage of shopping mall parking lots with aggregated debit card spending patterns to forecast which retail companies will surprise on earnings this quarter?
```

**Failure catch (engineering):**
```
We should migrate our monolithic auth service into microservices to improve scalability and team autonomy
```

**Novel hypothesis (no precedent):**
```
Using federated learning across hospital networks to predict sepsis onset without sharing patient data
```

---

## Course Concepts Demonstrated

| Concept | Where | Implementation |
|---|---|---|
| Multi-agent system (ADK) | Code | 4 specialized agents in sequential pipeline |
| MCP Server | Code | google_search MCP tool in Steelman Agent |
| Antigravity | Video | Entire project built via vibe coding in Antigravity IDE |
| Security features | Code | Input validation, prompt injection detection, domain relevance checking |
| Deployability | Video | Cloud Run MCP configured in Antigravity |

---

## Project Structure

```
hypothesis-graveyard/
├── app.py                    # Main Streamlit application
├── graveyard.py              # ChromaDB vector store operations
├── main.py                   # CLI pipeline orchestration
├── seed.py                   # Database seeding with historical hypotheses
├── requirements.txt          # Project dependencies
├── .env                      # API key (not committed)
├── agents/
│   ├── capture_agent.py      # Input validation and structuring
│   ├── archaeology_agent.py  # Semantic similarity search
│   ├── steelman_agent.py     # Hypothesis strengthening with web search
│   └── verdict_agent.py      # Conviction score synthesis
└── chroma_db/                # Persistent vector store (not committed)
```

---

## Submission

**Track:** Freestyle — AI Agents: Intensive Vibe Coding Capstone Project

**Competition:** [5-Day AI Agents: Intensive Vibe Coding Course With Google](https://www.kaggle.com/competitions/vibecoding-agents-capstone-project)

---

## Author

**Bhavya Sharma**
