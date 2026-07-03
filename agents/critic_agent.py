import os
import sys
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class CriticReport(BaseModel):
    status: str = Field(description="Must be either 'APPROVED' or 'REVISE'.")
    reasoning: str = Field(description="A brief explanation of the decision.")
    critique: str = Field(description="A detailed critique of the initial verdict, explaining why the conviction score is/is not justified, and identifying any bias (overly optimistic or overly cautious).")
    suggested_score_adjustment: float = Field(description="The suggested score adjustment (e.g. -15.0, +10.0, or 0.0).")

def review_verdict(captured_hyp, archaeology_rep, steelman_rep, initial_verdict) -> CriticReport:
    """
    Independently review the initial verdict against the original hypothesis, 
    archaeology findings, and steelmanned case.
    
    Args:
        captured_hyp (CapturedHypothesis): The structured hypothesis.
        archaeology_rep (ArchaeologyReport): The historical analysis report.
        steelman_rep (SteelmanReport): The steelmanned hypothesis report.
        initial_verdict (FinalVerdict): The initial verdict and conviction score.
        
    Returns:
        CriticReport: The critic's evaluation and recommended action.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("GEMINI_API_KEY is not set. Please update the .env file.")
        
    client = genai.Client(api_key=api_key)
    
    # Format similar cases if they exist
    past_cases_str = ""
    if hasattr(archaeology_rep, 'similar_cases_found') and archaeology_rep.similar_cases_found:
        for idx, case in enumerate(archaeology_rep.similar_cases_found):
            past_cases_str += f"--- CASE {idx+1} ---\n"
            past_cases_str += f"Text: {case.past_hypothesis_text}\n"
            past_cases_str += f"Outcome: {case.outcome}\n"
            past_cases_str += f"Conviction Score: {case.conviction_score}\n"
            past_cases_str += f"Lessons: {case.lessons_learned}\n\n"
    else:
        past_cases_str = "No similar historical cases found.\n"

    prompt = (
        f"1. ORIGINAL HYPOTHESIS:\n"
        f"Title: {captured_hyp.title}\n"
        f"Core Idea: {captured_hyp.core_idea}\n"
        f"Rationale: {captured_hyp.rationale}\n"
        f"Domain: {captured_hyp.domain}\n"
        f"Assumptions: {', '.join(captured_hyp.assumptions)}\n"
        f"Proposed Test: {captured_hyp.proposed_test}\n\n"
        
        f"2. HISTORICAL PRECEDENTS:\n"
        f"Summary: {archaeology_rep.historical_precedents_summary}\n"
        f"Similar Cases:\n{past_cases_str}\n"
        
        f"3. STEELMANNED CASE:\n"
        f"Steelmanned Statement: {steelman_rep.steelmanned_statement}\n"
        f"Improvements: {', '.join(steelman_rep.key_improvements)}\n"
        f"Testing Protocol: {steelman_rep.robust_testing_protocol}\n\n"
        
        f"4. INITIAL VERDICT TO REVIEW:\n"
        f"Verdict Category: {initial_verdict.verdict_category}\n"
        f"Conviction Score: {initial_verdict.conviction_score:.1f}/100\n"
        f"Summary: {initial_verdict.executive_summary}\n"
        f"Pros: {', '.join(initial_verdict.pros)}\n"
        f"Cons: {', '.join(initial_verdict.cons)}\n\n"
        
        f"Please independently critique this initial verdict. Check specifically:\n"
        f"- Is the conviction score actually justified given the strength and outcome of matched historical precedents?\n"
        f"- Is the verdict being overly optimistic despite a documented historical failure, or overly cautious despite strong supporting evidence?\n"
        f"Determine if the verdict is APPROVED as is, or if it requires a REVISE. If REVISE, provide a detailed critique and a suggested score adjustment."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CriticReport,
            system_instruction=(
                "You are the Critic Agent. Your job is to independently evaluate the Verdict Agent's "
                "conclusions. Be highly objective and rigorous. Look for gaps between history (archaeology) "
                "and the verdict. If the verdict is too optimistic about a concept that failed historically, "
                "or too pessimistic about a well-supported idea, you must flag it as 'REVISE' and provide "
                "constructive, sharp critique and a score adjustment. Otherwise, output 'APPROVED'."
            )
        )
    )
    
    return CriticReport.model_validate_json(response.text)
