import os
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class FinalVerdict(BaseModel):
    """
    Structured schema representing the Verdict Agent's final evaluation of the proposal.
    
    Design decision: Using a structured output for the final verdict allows the Streamlit 
    application to render color-coded dashboards, clean bulleted lists of pros and cons, 
    and progress checklists for milestones without needing fragile regex parsers.
    """
    conviction_score: float = Field(description="The final conviction score (0-100), representing the strength/viability of this hypothesis in its steelmanned form.")
    verdict_category: str = Field(description="Categorical verdict: e.g., 'TEST IMMEDIATELY', 'PROCEED WITH CAUTION', 'REJECT', 'REVISE AND RESUBMIT'.")
    executive_summary: str = Field(description="A concise executive summary of the multi-agent findings and the reasoning behind the score.")
    pros: list[str] = Field(description="Key strengths and reasons to test this hypothesis.")
    cons: list[str] = Field(description="Key risks, flaws, or reasons to be skeptical.")
    critical_milestones: list[str] = Field(description="Actionable next steps or milestones to validate the hypothesis.")

def evaluate_verdict(captured_hyp, archaeology_rep, steelman_rep, critic_feedback: str = None) -> FinalVerdict:
    """
    Synthesize all agent outputs and generate a final verdict and conviction score using Gemini.
    
    Args:
        captured_hyp (CapturedHypothesis): The structured hypothesis.
        archaeology_rep (ArchaeologyReport): The historical analysis report.
        steelman_rep (SteelmanReport): The steelmanned hypothesis report.
        critic_feedback (str): Optional feedback/critique from the Critic Agent to revise the initial verdict.
        
    Returns:
        FinalVerdict: The final evaluation and conviction score.
        
    Design decisions:
    1. Multi-Agent Synthesis: The Verdict Agent sits at the end of the pipeline 
       and acts as an objective committee. It reads the raw capture, the archaeology findings, 
       and the steelmanned proposal to arrive at an unbiased evaluation.
    2. Anchoring Bias Prevention: System instructions explicitly force the model to anchor 
       its conviction score to the Archaeology Agent's historical outcomes (precedents), 
       counterbalancing the Steelman Agent's structural optimism.
    3. Failure Mode Checking: The agent is instructed to distinguish between process improvements 
       (e.g., adding more unit tests or monitoring) and fundamental architectural shifts. 
       Adding process does not fix a design flaw; only a change in the technical approach justifies 
       raising the score of a historically failed concept.
    4. Conservative Baseline for Novelty: If no relevant precedents exist, the model keeps 
       the score in a cautious baseline (50-60) because novelty implies high uncertainty, 
       preventing premature high-conviction scores.
    5. Critic Reinforcement Loop: If `critic_feedback` is provided, we append it to the prompt. 
       This enables a reflection loop where the Verdict Agent adjusts its score based on critique, 
       emulating a peer-review debate.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("GEMINI_API_KEY is not set. Please update the .env file.")
        
    client = genai.Client(api_key=api_key)
    
    prompt = (
        f"1. ORIGINAL HYPOTHESIS:\n"
        f"Title: {captured_hyp.title}\n"
        f"Core Idea: {captured_hyp.core_idea}\n"
        f"Rationale: {captured_hyp.rationale}\n"
        f"Domain: {captured_hyp.domain}\n"
        f"Assumptions: {', '.join(captured_hyp.assumptions)}\n"
        f"Proposed Test: {captured_hyp.proposed_test}\n\n"
        f"2. HISTORICAL PRECEDENTS & LESSONS:\n"
        f"Summary: {archaeology_rep.historical_precedents_summary}\n"
        f"Recommendation: {archaeology_rep.recommendation_from_history}\n\n"
        f"3. STEELMANNED HYPOTHESIS & TESTING PROTOCOL:\n"
        f"Steelmanned Statement: {steelman_rep.steelmanned_statement}\n"
        f"Improvements: {', '.join(steelman_rep.key_improvements)}\n"
        f"Robust Testing: {steelman_rep.robust_testing_protocol}\n\n"
    )
    
    # If the Critic Agent provided feedback, append it to initiate a review/adjustment pass
    if critic_feedback:
        prompt += (
            f"4. CRITIC FEEDBACK / REVISION REQUEST:\n"
            f"{critic_feedback}\n\n"
            f"Please carefully review the Critic Agent's feedback above. Adjust the conviction score, "
            f"verdict category, executive summary, pros/cons, and critical milestones to address the critique."
        )
    else:
        prompt += "Please synthesize these inputs to deliver a final verdict and a conviction score (0-100)."
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FinalVerdict,
            system_instruction=(
                "You are the Verdict Agent. Your job is to synthesize the original proposal, "
                "historical findings, and the steelmanned version. You must act as an objective, "
                "highly critical investment committee / senior architect. Weigh the pros and cons, "
                "assign a conviction score between 0 and 100 (where 0 means a guaranteed failure and "
                "100 means an absolute must-run with high success likelihood), choose a clear categorical "
                "verdict, and outline the critical milestones for validation.\n\n"
                "CRITICAL INSTRUCTIONS FOR CONVICTION SCORING:\n"
                "1. Treat the Archaeology Agent's matched precedents as the primary anchor for the conviction score, "
                "not the Steelman Agent's optimistic reframing.\n"
                "2. If a closely matching historical precedent failed, the conviction score must stay below 50 unless "
                "the steelmanned version fundamentally eliminates the exact root cause of that historical failure—not "
                "just adds more process or testing rigor around it. Adding more testing steps, more KPIs, or more "
                "monitoring does NOT eliminate a root cause like 'distributed tracing overhead added 340ms latency' "
                "or 'team lacked operational maturity.' Only a fundamental change to the technical approach itself "
                "counts as eliminating the root cause. Be skeptical of steelmanned versions that just add process "
                "without changing the core technical risk.\n"
                "3. In your executive summary / reasoning, you must explicitly state which specific root cause from the "
                "matched precedent you checked against, and whether the steelman actually addresses that exact "
                "root cause or just surrounds it with more rigor.\n"
                "4. If no strong relevant precedent exists in the graveyard (e.g., the Archaeology Agent states "
                "'No strong relevant precedent found in the graveyard'), you must treat the conviction score with "
                "appropriately less certainty, keeping it conservative and closer to a neutral/cautious baseline "
                "(e.g., 50-60), rather than defaulting to high confidence."
            )
        )
    )
    
    return FinalVerdict.model_validate_json(response.text)
