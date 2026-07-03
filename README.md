# 🪦 Hypothesis Graveyard

Institutional memory for quant finance and engineering teams. Never repeat a failed hypothesis again.

## The Problem

Teams waste weeks testing ideas that were already tested and failed. That knowledge disappears when people leave, projects end, or notebooks get buried. Six months later someone tests the exact same thing and fails again.

## The Solution

Hypothesis Graveyard is a multi-agent system that maintains semantic memory of every hypothesis a team has ever tested — and surfaces that memory the moment someone is about to repeat a mistake.

## How It Works

When you propose a hypothesis, four specialized agents analyze it in sequence:

* **🕵️ Capture Agent** — Structures and validates your hypothesis
* **🏛️ Archaeology Agent** — Searches institutional memory for similar past attempts
* **🛡️ Steelman Agent** — Builds the strongest possible version using current evidence
* **⚖️ Verdict Agent** — Delivers a conviction score (0-100) with actionable verdict

## Key Features

* **Déjà Vu Alert** — Instantly flags when a hypothesis closely matches a past attempt
* **Semantic Search** — Finds similar hypotheses even when worded completely differently
* **Team Mode** — Contributor attribution shows who tested what and when
* **Graveyard Map** — Visual network showing relationships between hypotheses
* **Security Guardrails** — Prompt injection detection and input sanitization
* **Auto-seeding** — Pre-loaded with 20 realistic historical hypotheses across Quant Finance, Software Engineering, and ML/Data Science
* **Auto-expanding Domains** — The Capture Agent automatically recognises and creates new domain categories beyond the defaults. The system has self-expanded to include domains like E-commerce/Recommender Systems when new hypothesis types are submitted.

## Tech Stack

* **Google ADK** — Multi-agent orchestration
* **Gemini 2.5 Flash** — Agent reasoning
* **Gemini Embedding 001** — Semantic similarity search
* **ChromaDB** — Persistent vector store
* **Plotly + NetworkX** — Graveyard Map visualization
* **Streamlit** — Web interface
* *Built entirely in Antigravity IDE*

## Architecture

```
User Input → Capture Agent → Archaeology Agent → Steelman Agent → Verdict Agent → Conviction Score
                                     ↑                                    ↓
                              ChromaDB Vector Store ←←← Write Back (if buried)
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/bhavyasharma2/hypothesis-graveyard.git
   cd hypothesis-graveyard
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a free Gemini API key** at [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

4. **Add your Gemini API key:**
   Either paste it directly in the app sidebar under 'Enter your Gemini API Key', or create a `.env` file with `GEMINI_API_KEY=your_key_here` for persistent use.

5. **Run the app:**
   ```bash
   streamlit run app.py
   ```
   *The graveyard auto-seeds with 20 historical hypotheses on first run — no manual setup needed.*

## Demo Scenarios

* **Quant Finance:** *"Stocks that gap up more than 2% on Monday open outperform the market over the next 5 days"*
* **Software Engineering:** *"We should migrate our monolithic auth service into microservices to improve scalability"*
* **Semantic Match Test:** *"What if we used parking lot satellite images and credit card data to predict retail earnings?"*

## Course Concepts Demonstrated

| Concept | Implementation |
| :--- | :--- |
| **Multi-agent system (ADK)** | 4 specialized agents in sequential pipeline |
| **MCP Server** | Cloud Run MCP + google_search tool integration |
| **Antigravity** | Entire project built in Antigravity IDE |
| **Security features** | Input sanitization, prompt injection detection |
| **Deployability** | GCP Cloud Run ready |

---

**Track:** Freestyle — AI Agents: Intensive Vibe Coding Capstone Project  
**Author:** Bhavya Sharma
