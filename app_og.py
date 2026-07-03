import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import networkx as nx
import plotly.graph_objects as go
from st_keyup import st_keyup

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import graveyard
from agents.capture_agent import capture_hypothesis
from agents.archaeology_agent import analyze_history
from agents.steelman_agent import steelman_hypothesis
from agents.verdict_agent import evaluate_verdict

# Page configuration
st.set_page_config(
    page_title="Hypothesis Graveyard",
    page_icon="🪦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Key resolution logic
if "gemini_api_key" in st.session_state and st.session_state.gemini_api_key:
    os.environ["GEMINI_API_KEY"] = st.session_state.gemini_api_key
else:
    load_dotenv()
    env_key = os.getenv("GEMINI_API_KEY", "")
    if env_key and env_key != "your_key_here":
        st.session_state.gemini_api_key = env_key
        os.environ["GEMINI_API_KEY"] = env_key

# Custom CSS for premium dark-mode styling with purple/blue gradients
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0c0e14 !important;
        color: #f3f4f6 !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Padding to avoid scrolling behind top Deploy bar */
    .main .block-container {
        padding-top: 0.1rem !important;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #11131e !important;
        border-right: 1px solid #21253a !important;
    }
    
    /* Card Styles */
    .hypothesis-card {
        background-color: #161825;
        border: 1px solid #21253a;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
        transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    }
    .hypothesis-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(139, 92, 246, 0.12);
        border-color: #7c3aed;
    }
    
    /* Card Left Border Styles matching outcome color */
    .card-failed {
        border-left: 4px solid #ef4444 !important;
    }
    .card-success {
        border-left: 4px solid #10b981 !important;
    }
    .card-proposed {
        border-left: 4px solid #f59e0b !important;
    }
    .card-abandoned {
        border-left: 4px solid #6b7280 !important;
    }
    
    /* Badge Styles */
    .badge {
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .badge-quant {
        background-color: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .badge-swe {
        background-color: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }
    .badge-success {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-failed {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    .badge-proposed {
        background-color: rgba(245, 158, 11, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-abandoned {
        background-color: rgba(107, 114, 128, 0.15);
        color: #9ca3af;
        border: 1px solid rgba(107, 114, 128, 0.3);
    }
    
    /* Title styling */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 3rem;
        margin-top: 0rem !important;
        margin-bottom: 0.2rem;
    }
    .main-title-text {
        background: linear-gradient(90deg, #60a5fa 0%, #a78bfa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* General buttons subtle gradient highlight */
    .stButton > button {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.08) 0%, rgba(37, 99, 235, 0.08) 100%) !important;
        color: #e5e7eb !important;
        border: 1px solid rgba(124, 58, 237, 0.3) !important;
        border-radius: 6px !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.2) 0%, rgba(37, 99, 235, 0.2) 100%) !important;
        border-color: #a78bfa !important;
        color: #ffffff !important;
        box-shadow: 0 0 10px rgba(167, 139, 250, 0.2) !important;
    }
    
    /* Prominent gradient for Analyze button */
    .prominent-btn button {
        background: linear-gradient(90deg, #8b5cf6 0%, #3b82f6 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        border: none !important;
        padding: 0.8rem 2rem !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.35) !important;
        transition: transform 0.1s ease, box-shadow 0.2s ease !important;
    }
    .prominent-btn button:hover {
        box-shadow: 0 6px 22px rgba(139, 92, 246, 0.55) !important;
        transform: translateY(-1px) !important;
    }
    .prominent-btn button:active {
        background: #1d4ed8 !important;
        box-shadow: 0 2px 10px rgba(37, 99, 235, 0.4) !important;
        transform: translateY(1px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- AUTO-SEEDING STARTUP CHECK -----------------
try:
    db_count = graveyard.collection.count()
except Exception:
    db_count = 0

if db_count == 0:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        st.warning("⚠️ Graveyard database is empty. Please configure a valid Gemini API Key in the sidebar expander to automatically seed the database with 20 historical hypotheses.")
        
        # Render clean sidebar with expanded key config so they see it immediately
        with st.sidebar:
            st.markdown("## ⚙️ Configuration")
            api_key_input = st.text_input("Enter your Gemini API Key", type="password", help="Needed to run embeddings and agents.")
            if api_key_input:
                st.session_state.gemini_api_key = api_key_input
                os.environ["GEMINI_API_KEY"] = api_key_input
                st.rerun()
        st.stop()
    else:
        # Show seeding spinner
        with st.spinner("🌱 Initializing Hypothesis Graveyard with historical data..."):
            try:
                from seed import SEED_HYPOTHESES
                for hyp in SEED_HYPOTHESES:
                    graveyard.store_hypothesis(
                        text=hyp["text"],
                        domain=hyp["domain"],
                        outcome=hyp["outcome"],
                        conviction_score=hyp["conviction_score"],
                        notes=hyp["notes"],
                        contributor=hyp.get("contributor", "Anonymous"),
                        date=hyp.get("date")
                    )
                st.success("🌱 Graveyard initialized successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed to seed database: {e}")
                st.stop()

# ----------------- SIDEBAR LAYOUT (Configuration Expander & Stats) -----------------
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    current_key = st.session_state.get("gemini_api_key", "")
    
    if current_key and current_key != "your_key_here":
        st.success("✅ API Key accepted")
        new_key = st.text_input("Enter your Gemini API Key", value=current_key, type="password", help="Needed to run embeddings and agents.")
        if new_key != current_key:
            st.session_state.gemini_api_key = new_key
            os.environ["GEMINI_API_KEY"] = new_key
            st.rerun()
            
        if st.button("Clear / Reset Key"):
            st.session_state.gemini_api_key = ""
            os.environ["GEMINI_API_KEY"] = ""
            st.rerun()
    else:
        api_key_input = st.text_input("Enter your Gemini API Key", type="password", help="Needed to run embeddings and agents.")
        if api_key_input:
            st.session_state.gemini_api_key = api_key_input
            os.environ["GEMINI_API_KEY"] = api_key_input
            st.rerun()
                
    st.markdown("---")
    st.markdown("## 📊 Graveyard Statistics")
    
    try:
        count = graveyard.collection.count()
        if count > 0:
            all_data = graveyard.collection.get()
            df_stats = pd.DataFrame([meta for meta in all_data['metadatas']])
            
            st.metric("Buried Hypotheses", f"{count}")
            
            # Domain breakdown
            if 'domain' in df_stats.columns:
                domain_counts = df_stats['domain'].value_counts()
                st.markdown("**Domains**")
                for dom, cnt in domain_counts.items():
                    st.caption(f"- {dom}: {cnt}")
                    
            # Outcome breakdown
            if 'outcome' in df_stats.columns:
                outcome_counts = df_stats['outcome'].value_counts()
                st.markdown("**Outcomes**")
                for out, cnt in outcome_counts.items():
                    out_l = out.lower()
                    color = "#9ca3af"
                    if out_l == 'success':
                        color = "#10b981"
                    elif out_l == 'failed':
                        color = "#ef4444"
                    elif out_l == 'proposed':
                        color = "#f59e0b"
                    st.markdown(f"<div style='font-size: 0.85rem; padding: 2px 0;'><span style='color: {color}; margin-right: 6px;'>●</span>{out}: {cnt}</div>", unsafe_allow_html=True)
                    
            # Avg conviction
            if 'conviction_score' in df_stats.columns:
                avg_conv = df_stats['conviction_score'].mean()
                st.metric("Avg Conviction Score", f"{avg_conv:.1f}/100")
        else:
            st.info("Graveyard is empty.")
    except Exception as e:
        st.sidebar.error(f"Error loading stats: {e}")

# ----------------- MAIN TITLE (Move to top of main area, with tombstone emoji) -----------------
st.markdown('<h1 class="main-title">🪦 <span class="main-title-text">Hypothesis Graveyard</span></h1>', unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 2rem;'>Maintain institutional memory of failed and tested ideas. Avoid repeating past mistakes.</p>", unsafe_allow_html=True)

# ----------------- TABBED ROUTING -----------------
tab_sandbox, tab_browser, tab_map = st.tabs(["🔬 Analysis Sandbox", "🔍 Browse Graveyard", "🗺️ Graveyard Map"])

# PAGE 1: ANALYSIS SANDBOX
with tab_sandbox:
    col_input, col_results = st.columns([1, 1])
    
    with col_results:
        st.markdown("### Analysis Results")
        dejavu_placeholder = st.empty()
        
    with col_input:
        st.markdown("### Propose New Hypothesis")
        
        # Agent descriptions expander
        with st.expander("ℹ️ How it works", expanded=False):
            st.markdown("""
            * **🕵️ Capture Agent** — Understands and structures your hypothesis.
            * **🏛️ Archaeology Agent** — Searches institutional memory for similar past attempts and their outcomes.
            * **🛡️ Steelman Agent** — Builds the strongest possible version of your hypothesis using current evidence.
            * **⚖️ Verdict Agent** — Delivers a conviction score (0-100) and tells you whether to proceed, revise, or abandon.
            """)
            
        # Proposal Form: Domain text input and suggested chips removed completely. Auto-detects domain context.
        with st.form("hypothesis_form"):
            raw_text = st.text_area(
                "Describe your hypothesis:",
                placeholder="e.g., We should use a multi-head attention mechanism on 1-minute order book imbalances because...",
                height=180
            )
            
            # Prominent gradient analyze button wrapper
            st.markdown("<div class='prominent-btn'>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Analyse Hypothesis", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        if submitted:
            if not raw_text.strip():
                st.error("Please enter a hypothesis description.")
            elif not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_key_here":
                st.error("Please set a valid GEMINI_API_KEY in the sidebar or .env file.")
            else:
                st.session_state.raw_text = raw_text
                st.session_state.analysis_completed = False
                st.session_state.buried = False
                if "dejavu_alert" in st.session_state:
                    del st.session_state["dejavu_alert"]
                
                try:
                    with st.status("🔬 Executing Agents Pipeline...", expanded=True) as status:
                        status.write("🕵️ Capture Agent: Parsing structured details and auto-detecting domain...")
                        captured_hyp = capture_hypothesis(raw_text)
                        st.session_state.captured_hyp = captured_hyp
                        
                        status.write("🏛️ Archaeology Agent: Scanning precedents...")
                        archaeology_rep = analyze_history(captured_hyp)
                        st.session_state.archaeology_rep = archaeology_rep
                        
                        # Deja Vu Alert check (similarity > 85%)
                        query_text = f"{captured_hyp.core_idea} {captured_hyp.rationale}"
                        similar_cases = graveyard.search_similar(query_text, n=3)
                        dejavu_found = False
                        highest_sim = 0.0
                        matched_case = None
                        for case in similar_cases:
                            dist = case.get('distance', 2.0)
                            if dist is None: dist = 2.0
                            sim = (1.0 - (dist / 2.0)) * 100
                            if sim > 85.0 and sim > highest_sim:
                                highest_sim = sim
                                matched_case = case
                                dejavu_found = True
                        if dejavu_found:
                            st.session_state.dejavu_alert = {
                                "title": matched_case['text'],
                                "outcome": matched_case['metadata'].get('outcome', 'Unknown'),
                                "similarity": highest_sim,
                                "contributor": matched_case['metadata'].get('contributor', 'Anonymous'),
                                "date": matched_case['metadata'].get('date', 'Unknown Date')
                            }
                            
                        status.write("🛡️ Steelman Agent: Optimizing testing protocol...")
                        steelman_rep = steelman_hypothesis(captured_hyp, archaeology_rep)
                        st.session_state.steelman_rep = steelman_rep
                        
                        status.write("⚖️ Verdict Agent: Synthesizing final conviction...")
                        verdict_rep = evaluate_verdict(captured_hyp, archaeology_rep, steelman_rep)
                        st.session_state.verdict_rep = verdict_rep
                        
                        status.update(label="✅ Agents pipeline complete.", state="complete")
                        
                    st.session_state.analysis_completed = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error running pipeline: {e}")
                    
    with col_results:
        # Display Deja Vu Alert
        if "dejavu_alert" in st.session_state:
            alert = st.session_state.dejavu_alert
            st.markdown(f"""
            <div style='background-color: rgba(239, 68, 68, 0.12); border: 2px solid #ef4444; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center;'>
                <div style='color: #f87171; font-size: 1.2rem; font-weight: 800; margin-bottom: 0.5rem;'>⚠️ DÉJÀ VU DETECTED — {alert['contributor']} already tested this on {alert['date']}</div>
                <div style='color: #fca5a5; font-size: 0.9rem; font-weight: 500;'>Matched Hypothesis:</div>
                <div style='color: #fca5a5; font-size: 1.05rem; font-weight: 600; margin-top: 0.5rem; line-height: 1.4;'>"{alert['title']}"</div>
                <div style='font-size: 1.15rem; color: #ef4444; font-weight: 800; margin-top: 0.5rem;'>Outcome: {alert['outcome']} | Similarity: {alert['similarity']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        if st.session_state.get("analysis_completed", False):
            chyp = st.session_state.captured_hyp
            arch = st.session_state.archaeology_rep
            steel = st.session_state.steelman_rep
            verd = st.session_state.verdict_rep
            
            # Conviction Score Display (Larger & Color-coded: red < 40, amber 40-70, green > 70)
            score = verd.conviction_score
            if score < 40:
                score_color = "#ef4444" # red
            elif score <= 70:
                score_color = "#f59e0b" # amber
            else:
                score_color = "#10b981" # green
                
            st.markdown(f"""
            <div style='text-align: center; margin: 1.5rem auto; padding: 1.8rem; border: 2.5px solid {score_color}; border-radius: 12px; background-color: #141724;'>
                <div style='font-size: 5.5rem; font-weight: 900; color: {score_color}; line-height: 1;'>{score:.0f}</div>
                <div style='font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; margin-top: 10px; font-weight: 700;'>Conviction Score</div>
                <div style='font-size: 1.3rem; font-weight: 700; color: #f3f4f6; margin-top: 15px;'>Verdict: {verd.verdict_category}</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.write(verd.executive_summary)
            st.markdown("---")
            
            res_tabs = st.tabs(["⚖️ Verdict & Milestones", "🛡️ Steelman", "🏛️ History", "🕵️ Structured Idea"])
            
            with res_tabs[0]:
                st.markdown("##### Pros & Strengths")
                for pro in verd.pros:
                    st.markdown(f"- ✅ {pro}")
                st.markdown("##### Cons & Risks")
                for con in verd.cons:
                    st.markdown(f"- ❌ {con}")
                st.markdown("##### Critical Validation Milestones")
                for ms in verd.critical_milestones:
                    st.markdown(f"- 📍 {ms}")
                    
            with res_tabs[1]:
                st.info(f"**Refined Hypothesis Statement:**\n\n{steel.steelmanned_statement}")
                st.markdown("##### Key Refinements")
                for imp in steel.key_improvements:
                    st.markdown(f"- ⚡ {imp}")
                st.markdown("##### Robust Testing Protocol")
                st.markdown(steel.robust_testing_protocol)
                st.markdown("##### Mitigation Strategies")
                for mit in steel.mitigation_strategies:
                    st.markdown(f"- 🛡️ {mit}")
                    
            with res_tabs[2]:
                st.markdown(arch.historical_precedents_summary)
                if arch.similar_cases_found:
                    st.markdown("##### Similar Past Cases Analyzed")
                    for case in arch.similar_cases_found:
                        out_badge = f'<span class="badge badge-success">{case.outcome}</span>' if case.outcome.lower() == 'success' else f'<span class="badge badge-failed">{case.outcome}</span>'
                        st.markdown(f"""
                        <div class="hypothesis-card">
                            <div>{out_badge} <strong>Score: {case.conviction_score:.0f}/100</strong></div>
                            <p style='margin-top: 0.5rem; font-style: italic;'>"{case.past_hypothesis_text}"</p>
                            <strong>Overlap Analysis:</strong> {case.overlap_analysis}<br/>
                            <strong>Lessons:</strong> {case.lessons_learned}<br/>
                            <strong>Distinction:</strong> {case.distinction}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.caption("No similar historical cases found in the graveyard.")
                    
            with res_tabs[3]:
                st.markdown(f"**Title:** {chyp.title}")
                st.markdown(f"**Domain:** {chyp.domain}")
                st.markdown(f"**Core Idea:** {chyp.core_idea}")
                st.markdown(f"**Rationale:** {chyp.rationale}")
                st.markdown("##### Key Assumptions")
                for ass in chyp.assumptions:
                    st.markdown(f"- {ass}")
                st.markdown(f"##### Original Proposed Test")
                st.markdown(chyp.proposed_test)
                
            st.markdown("---")
            
            # Burial Confirmation handling
            if st.session_state.get("buried", False):
                st.success("✅ Hypothesis successfully archived in the graveyard. The institutional database has been updated.")
            else:
                st.markdown("### 🪦 Bury in the Graveyard")
                st.caption("Document this analysis in the graveyard database to maintain institutional memory.")
                
                with st.form("bury_form"):
                    col_b1, col_b2, col_b3 = st.columns(3)
                    with col_b1:
                        outcome_opt = st.selectbox("Outcome:", ["Proposed", "Failed", "Success", "Abandoned"])
                    with col_b2:
                        final_score = st.slider("Adjust Conviction Score:", 0, 100, int(score))
                    with col_b3:
                        contributor_opt = st.text_input("Your name:", placeholder="Anonymous")
                        
                    notes_opt = st.text_area("Burying Notes / Key Lessons:", value=verd.executive_summary)
                    
                    confirm_bury = st.form_submit_button("Bury in Graveyard", use_container_width=True)
                    
                    if confirm_bury:
                        try:
                            doc_id = graveyard.store_hypothesis(
                                text=st.session_state.raw_text,
                                domain=chyp.domain,
                                outcome=outcome_opt,
                                conviction_score=float(final_score),
                                notes=notes_opt,
                                contributor=contributor_opt.strip() if contributor_opt.strip() else "Anonymous"
                            )
                            st.session_state.buried = True
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to bury: {e}")
        else:
            st.info("Submit a hypothesis on the left to see the multi-agent analysis results.")

# PAGE 2: BROWSE GRAVEYARD
with tab_browser:
    if "search_query" not in st.session_state:
        st.session_state.search_query = ""
        
    col_search, col_filter_domain, col_filter_outcome, col_sort = st.columns([2, 1, 1, 1])
    
    with col_search:
        search_query = st_keyup("Search Graveyard:", placeholder="Search similar ideas...", debounce=500, key="search_input")
        
    if search_query != st.session_state.search_query:
        st.session_state.search_query = search_query
        st.rerun()
        
    try:
        all_data = graveyard.collection.get()
        if all_data and all_data['documents']:
            records = []
            for i in range(len(all_data['documents'])):
                meta = all_data['metadatas'][i]
                records.append({
                    "id": all_data['ids'][i],
                    "text": all_data['documents'][i],
                    "domain": meta.get("domain", "Unknown"),
                    "outcome": meta.get("outcome", "Unknown"),
                    "conviction_score": meta.get("conviction_score", 0.0),
                    "notes": meta.get("notes", ""),
                    "contributor": meta.get("contributor", "Anonymous"),
                    "date": meta.get("date", "Unknown Date")
                })
            df = pd.DataFrame(records)
            
            domains = ["All"] + sorted(df['domain'].unique().tolist())
            outcomes = ["All"] + sorted(df['outcome'].unique().tolist())
            
            with col_filter_domain:
                selected_domain = st.selectbox("Filter Domain:", domains)
            with col_filter_outcome:
                selected_outcome = st.selectbox("Filter Outcome:", outcomes)
            with col_sort:
                selected_sort = st.selectbox("Sort by:", ["Most Recent", "Oldest First", "Highest Score", "Lowest Score"])
                
            if search_query.strip():
                if not os.getenv("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY") == "your_key_here":
                    filtered_df = df[df['text'].str.contains(search_query, case=False) | df['notes'].str.contains(search_query, case=False)].copy()
                else:
                    similar_results = graveyard.search_similar(search_query, n=15)
                    matching_ids = []
                    for res in similar_results:
                        dist = res.get('distance', 2.0)
                        if dist is None: dist = 2.0
                        sim = (1.0 - (dist / 2.0)) * 100
                        if sim >= 70.0:
                            matching_ids.append(res['id'])
                    filtered_df = df[df['id'].isin(matching_ids)].copy()
                    if not filtered_df.empty:
                        filtered_df['sort_order'] = filtered_df['id'].apply(lambda x: matching_ids.index(x))
                        filtered_df = filtered_df.sort_values('sort_order')
            else:
                filtered_df = df.copy()
                
            if selected_domain != "All":
                filtered_df = filtered_df[filtered_df['domain'] == selected_domain]
            if selected_outcome != "All":
                filtered_df = filtered_df[filtered_df['outcome'] == selected_outcome]
                
            if selected_sort == "Most Recent":
                filtered_df = filtered_df.sort_values(by="date", ascending=False)
            elif selected_sort == "Oldest First":
                filtered_df = filtered_df.sort_values(by="date", ascending=True)
            elif selected_sort == "Highest Score":
                filtered_df = filtered_df.sort_values(by="conviction_score", ascending=False)
            elif selected_sort == "Lowest Score":
                filtered_df = filtered_df.sort_values(by="conviction_score", ascending=True)
                
            if not filtered_df.empty:
                for _, row in filtered_df.iterrows():
                    dom_class = "badge-quant" if row['domain'].lower() == 'quant finance' else "badge-swe"
                    
                    out_lower = row['outcome'].lower()
                    if out_lower == 'success':
                        out_class = "badge-success"
                        border_class = "card-success"
                    elif out_lower == 'failed':
                        out_class = "badge-failed"
                        border_class = "card-failed"
                    elif out_lower == 'proposed':
                        out_class = "badge-proposed"
                        border_class = "card-proposed"
                    else:
                        out_class = "badge-abandoned"
                        border_class = "card-abandoned"
                        
                    # Color-code conviction scores
                    c_score = row['conviction_score']
                    if c_score < 40:
                        score_text_color = "#ef4444"
                    elif c_score <= 70:
                        score_text_color = "#f59e0b"
                    else:
                        score_text_color = "#10b981"
                        
                    # Row columns: Left for text, right for prominent large score
                    col_card_text, col_card_score = st.columns([8, 2])
                    
                    with col_card_text:
                        st.markdown(f"""
                        <div class="hypothesis-card {border_class}" style="margin-bottom: 0;">
                            <div>
                                <span class="badge {dom_class}">{row['domain']}</span>
                                <span class="badge {out_class}">{row['outcome']}</span>
                            </div>
                            <p style="font-size: 1.15rem; font-weight: 600; margin-top: 0.5rem; margin-bottom: 0.5rem; color: #f3f4f6;">"{row['text']}"</p>
                            <div style="font-size: 0.8rem; color: #9ca3af; margin-bottom: 0.8rem;">
                                Buried by <strong>{row['contributor']}</strong> on {row['date']}
                            </div>
                            <div style="background-color: #0c0e14; padding: 1rem; border-radius: 8px; border-left: 3px solid #4b5563;">
                                <strong style="color: #9ca3af; font-size: 0.8rem; text-transform: uppercase;">Institutional Notes & Lessons:</strong>
                                <p style="margin: 0.3rem 0 0 0; font-size: 0.95rem; line-height: 1.5; color: #d1d5db;">{row['notes']}</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_card_score:
                        st.markdown(f"""
                        <div style='background-color: #161825; border: 2px solid {score_text_color}; border-radius: 12px; padding: 1.5rem; text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center;'>
                            <div style='font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px;'>Conviction</div>
                            <div style='font-size: 3.2rem; font-weight: 900; color: {score_text_color}; line-height: 1; margin-top: 5px;'>{c_score:.0f}</div>
                            <div style='font-size: 0.65rem; color: #9ca3af; text-transform: uppercase; margin-top: 4px;'>Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    st.markdown("<div style='height: 0.8rem;'></div>", unsafe_allow_html=True)
            else:
                st.info("No hypotheses match your search criteria.")
        else:
            st.info("The graveyard is currently empty.")
    except Exception as e:
        st.error(f"Error loading graveyard browser: {e}")

# PAGE 3: GRAVEYARD MAP
with tab_map:
    st.markdown("<p style='color: #94a3b8; font-size: 1rem; margin-bottom: 1.5rem;'>Visualizing relationships between hypotheses. Nodes represent hypotheses, and edges represent embedding similarity > 70%.</p>", unsafe_allow_html=True)
    
    try:
        data = graveyard.collection.get(include=['embeddings', 'documents', 'metadatas'])
        if data and data['documents']:
            num_nodes = len(data['documents'])
            
            selected_node = st.session_state.get("selected_node")
            
            G = nx.Graph()
            id_to_record = {}
            node_ids = []
            
            for i in range(num_nodes):
                node_id = data['ids'][i]
                text = data['documents'][i]
                meta = data['metadatas'][i]
                
                outcome = meta.get('outcome', 'Unknown')
                conviction = float(meta.get('conviction_score', 50.0))
                
                G.add_node(node_id)
                node_ids.append(node_id)
                id_to_record[node_id] = {
                    "text": text,
                    "domain": meta.get('domain', 'Unknown'),
                    "outcome": outcome,
                    "conviction": conviction,
                    "contributor": meta.get('contributor', 'Anonymous'),
                    "date": meta.get('date', 'Unknown Date'),
                    "notes": meta.get('notes', 'No notes provided.')
                }
                
            # Embeddings similarities
            embeddings = np.array(data['embeddings'])
            sim_matrix = np.dot(embeddings, embeddings.T)
            
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    sim = float(sim_matrix[i, j])
                    if sim > 0.70:
                        G.add_edge(data['ids'][i], data['ids'][j])
            
            # Compute layouts
            pos = nx.spring_layout(G, seed=42)
            
            # Build edges trace
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1.5, color='#4b5563'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Build nodes trace categorized by outcome to support fixed color strings per trace
            traces_data = {
                'failed': {'x': [], 'y': [], 'size': [], 'text': [], 'border_color': [], 'border_width': [], 'opacity': [], 'ids': [], 'color': '#E74C3C', 'name': 'Failed'},
                'success': {'x': [], 'y': [], 'size': [], 'text': [], 'border_color': [], 'border_width': [], 'opacity': [], 'ids': [], 'color': '#2ECC71', 'name': 'Success'},
                'proposed': {'x': [], 'y': [], 'size': [], 'text': [], 'border_color': [], 'border_width': [], 'opacity': [], 'ids': [], 'color': '#F39C12', 'name': 'Proposed'},
                'other': {'x': [], 'y': [], 'size': [], 'text': [], 'border_color': [], 'border_width': [], 'opacity': [], 'ids': [], 'color': '#7F8C8D', 'name': 'Other'}
            }
            
            for node in G.nodes():
                x, y = pos[node]
                
                rec = id_to_record[node]
                outcome = rec['outcome'].lower()
                conviction = rec['conviction']
                
                if outcome == 'failed':
                    cat = 'failed'
                elif outcome == 'success':
                    cat = 'success'
                elif outcome == 'proposed':
                    cat = 'proposed'
                else:
                    cat = 'other'
                    
                # Sized by conviction_score scaled between 15-40px
                size = 15.0 + (conviction / 100.0) * 25.0
                
                if selected_node:
                    if node == selected_node:
                        node_size = size * 1.25
                        border_color = '#FFFFFF'
                        border_width = 3
                        opacity = 1.0
                    else:
                        node_size = size
                        border_color = '#374151'
                        border_width = 1
                        opacity = 0.6
                else:
                    node_size = size
                    border_color = '#374151'
                    border_width = 1
                    opacity = 1.0
                    
                traces_data[cat]['x'].append(x)
                traces_data[cat]['y'].append(y)
                traces_data[cat]['size'].append(node_size)
                traces_data[cat]['text'].append(rec['text'])
                traces_data[cat]['border_color'].append(border_color)
                traces_data[cat]['border_width'].append(border_width)
                traces_data[cat]['opacity'].append(opacity)
                traces_data[cat]['ids'].append(node)
                
            fig = go.Figure()
            fig.add_trace(edge_trace)
            
            for cat, trace in traces_data.items():
                if len(trace['x']) > 0:
                    fig.add_trace(go.Scatter(
                        x=trace['x'], y=trace['y'],
                        mode='markers',
                        hoverinfo='text',
                        text=trace['text'],
                        marker=dict(
                            showscale=False,
                            color=trace['color'], # Fixed color string per trace
                            size=trace['size'],
                            opacity=trace['opacity'], # Opacities list
                            line=dict(
                                color=trace['border_color'],
                                width=trace['border_width']
                            )
                        ),
                        customdata=trace['ids'],
                        name=trace['name']
                    ))
            
            fig.update_layout(
                plot_bgcolor='#0E1117',
                paper_bgcolor='#0E1117',
                showlegend=False,
                margin=dict(b=0, l=0, r=0, t=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, autorange=True),
                height=550
            )
            
            col_graph, col_panel = st.columns([2, 1])
            
            with col_graph:
                st.markdown(
                    "<div style='display: flex; gap: 1.2rem; justify-content: flex-start; align-items: center; font-size: 0.9rem; color: #9ca3af; padding-top: 5px; margin-bottom: 10px;'>"
                    "<span>🔴 Failed</span>"
                    "<span>🟢 Success</span>"
                    "<span>🟡 Proposed</span>"
                    "<span style='border-left: 1px solid #374151; padding-left: 1.2rem; font-style: italic;'>Node size scales with score</span>"
                    "</div>",
                    unsafe_allow_html=True
                )
                
                # Render using st.plotly_chart with select events
                event_data = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="points"
                )
                
                if event_data and hasattr(event_data, "selection") and "points" in event_data.selection:
                    points = event_data.selection["points"]
                    if len(points) > 0:
                        clicked_id = points[0].get("customdata")
                        if clicked_id:
                            if isinstance(clicked_id, (list, tuple)) and len(clicked_id) > 0:
                                clicked_id = clicked_id[0]
                            if clicked_id != st.session_state.get("selected_node"):
                                st.session_state.selected_node = clicked_id
                                st.rerun()
                
            with col_panel:
                st.markdown("### 🔍 Node Details")
                current_selected = st.session_state.get("selected_node")
                if current_selected and current_selected in id_to_record:
                    rec = id_to_record[current_selected]
                    
                    st.markdown(f"#### {rec['outcome']} | {rec['domain']}")
                    
                    col_score_m, col_score_p = st.columns([2, 3])
                    with col_score_m:
                        st.metric("Conviction Score", f"{rec['conviction']:.0f}/100")
                    with col_score_p:
                        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
                        st.progress(float(rec['conviction'] / 100.0))
                    
                    st.markdown(f"**Hypothesis:**\n*{rec['text']}*")
                    st.caption(f"Buried by {rec['contributor']} on {rec['date']}")
                    st.markdown("---")
                    st.markdown("**Institutional Notes:**")
                    st.write(rec['notes'])
                    
                    if st.button("Clear Highlight", use_container_width=True):
                        st.session_state.selected_node = None
                        st.rerun()
                else:
                    st.info("Click on any node in the map to view its details and institutional lessons.")
        else:
            st.info("The graveyard is currently empty.")
    except Exception as e:
        st.error(f"Error loading graveyard map: {e}")
