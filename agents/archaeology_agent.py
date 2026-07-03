import os
import sys
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Ensure we can import graveyard from the parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import graveyard

load_dotenv()

class HistoricalComparison(BaseModel):
    """
    Structured model representing the comparative analysis of a single historical case.
    """
    past_hypothesis_text: str = Field(description="The text of the similar past hypothesis.")
    domain: str = Field(description="Domain of the past hypothesis.")
    outcome: str = Field(description="Outcome of the past hypothesis (e.g., Success, Failed).")
    conviction_score: float = Field(description="Historical conviction score.")
    overlap_analysis: str = Field(description="Analysis of how the new hypothesis overlaps with this past one.")
    lessons_learned: str = Field(description="Key lessons and reasons for the past outcome.")
    distinction: str = Field(description="How the new hypothesis differs from this past one.")

class ArchaeologyReport(BaseModel):
    """
    Structured report outlining all matching historical precedents.
    """
    similar_cases_found: list[HistoricalComparison] = Field(description="List of comparisons with similar past cases.")
    historical_precedents_summary: str = Field(description="A summary of the historical context, including common pitfalls.")
    recommendation_from_history: str = Field(description="A high-level recommendation on what to avoid or adapt based on history.")

def analyze_history(captured_hyp) -> ArchaeologyReport:
    """
    Search ChromaDB for similar hypotheses and analyze them using Gemini.
    
    Args:
        captured_hyp (CapturedHypothesis): The structured hypothesis from the Capture Agent.
        
    Returns:
        ArchaeologyReport: The historical analysis report.
        
    Design decisions:
    1. Query Formulation: We query the database using the combination of the `core_idea` 
       and the `rationale`. This ensures the semantic embedding contains both what the proposal
       wants to do and why, yielding highly relevant search results.
    2. Strict Similarity Filtering: We filter results by checking if the embedding similarity 
       is >= 75%. Distance is converted to similarity using the formula `(1.0 - (dist / 2.0)) * 100`. 
       A high threshold (75%) avoids injecting weak or unrelated noise into the LLM prompt.
    3. Honesty Policy: If no cases clear the similarity threshold, we return a blank list 
       instead of letting the model hallucinate or force-stretch distant precedents.
    4. Anti-Analogy System Instruction: We instruct the model to never force tenuous analogies 
       across unrelated domains (e.g. quant latency vs. biological drug discovery).
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("GEMINI_API_KEY is not set. Please update the .env file.")
        
    client = genai.Client(api_key=api_key)
    
    # Query ChromaDB using both what the idea is (core_idea) and why it works (rationale)
    query_text = f"{captured_hyp.core_idea} {captured_hyp.rationale}"
    similar_cases = graveyard.search_similar(query_text, n=3)
    
    # Filter similar cases by embedding similarity score >= 75%
    # Design decision: converting cosine distance to similarity score
    strong_similar_cases = []
    for case in similar_cases:
        dist = case.get('distance', 2.0)
        if dist is None:
            dist = 2.0
        sim = (1.0 - (dist / 2.0)) * 100
        if sim >= 75.0:
            strong_similar_cases.append(case)
            
    # If no strong cases exist, return immediately with a clear statement of novelty
    if not strong_similar_cases:
        return ArchaeologyReport(
            similar_cases_found=[],
            historical_precedents_summary="No strong relevant precedent found in the graveyard. This appears to be a novel hypothesis.",
            recommendation_from_history="Since no strong historical precedents exist in our database, proceed with standard validation protocols. Be sure to document the testing process thoroughly."
        )
        
    # Format case metadata as a text block to feed into the Gemini prompt
    past_cases_str = ""
    for idx, case in enumerate(strong_similar_cases):
        meta = case['metadata']
        past_cases_str += f"--- CASE {idx+1} ---\n"
        past_cases_str += f"Text: {case['text']}\n"
        past_cases_str += f"Domain: {meta.get('domain', 'Unknown')}\n"
        past_cases_str += f"Outcome: {meta.get('outcome', 'Unknown')}\n"
        past_cases_str += f"Conviction Score: {meta.get('conviction_score', 'N/A')}\n"
        past_cases_str += f"Notes/Lessons: {meta.get('notes', 'N/A')}\n\n"
        
    prompt = (
        f"NEW HYPOTHESIS:\n"
        f"Title: {captured_hyp.title}\n"
        f"Core Idea: {captured_hyp.core_idea}\n"
        f"Rationale: {captured_hyp.rationale}\n"
        f"Domain: {captured_hyp.domain}\n"
        f"Assumptions: {', '.join(captured_hyp.assumptions)}\n"
        f"Proposed Test: {captured_hyp.proposed_test}\n\n"
        f"SIMILAR HISTORICAL CASES FOUND:\n"
        f"{past_cases_str}"
        f"Please analyze these historical cases and construct an Archaeology Report."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ArchaeologyReport,
            system_instruction=(
                "You are the Archaeology Agent. Your job is to analyze historical hypotheses "
                "from our graveyard, compare them to the newly proposed hypothesis, and extract "
                "valuable lessons, overlaps, and distinctions to prevent the team from repeating past mistakes.\n\n"
                "CRITICAL RELEVANCE RULES:\n"
                "1. If the matched hypothesis is from a completely unrelated domain or subject area (e.g., "
                "quant trading latency vs. biological drug discovery), do not force a tenuous analogy. "
                "Do not stretch unrelated domain failures into false analogies just because both involve words "
                "like 'model' or 'failure.'\n"
                "2. A precedent is only valid if the underlying mechanism of failure or success is genuinely "
                "transferable to the new hypothesis's actual domain and technical approach. If the best match "
                "is a weak or superficial analogy, say so honestly rather than presenting it as strong evidence.\n"
                "3. If all matched cases are weak or unrelated, you must explicitly state 'No strong relevant precedent "
                "found in the graveyard' in your historical_precedents_summary and return an empty list for similar_cases_found."
            )
        )
    )
    
    return ArchaeologyReport.model_validate_json(response.text)
