import os
import sys
import asyncio
import uuid
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# ADK imports.
# Design decision: The Agent Development Kit (ADK) is chosen for the Steelman Agent 
# because it provides native support for built-in tools (like google_search) and 
# structures agent execution loops in a modular, standardized client architecture.
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import InMemoryRunner
from google.adk.tools import google_search

load_dotenv()

class SteelmanReport(BaseModel):
    """
    Structured Pydantic schema representing the final steelmanned hypothesis report.
    """
    steelmanned_statement: str = Field(description="The refined, strongest version of the hypothesis.")
    key_improvements: list[str] = Field(description="List of specific improvements made to the original hypothesis.")
    revised_assumptions: list[str] = Field(description="Refined or new assumptions required to make this work.")
    robust_testing_protocol: str = Field(description="Detailed, robust methodology to test the steelmanned hypothesis, addressing historical failure points.")
    mitigation_strategies: list[str] = Field(description="Strategies to mitigate risks and avoid historical pitfalls.")

def run_coroutine_sync(coro):
    """
    Runs an asynchronous coroutine synchronously, accommodating environments 
    where an event loop is already running (e.g., Streamlit).
    
    Design decision: In multi-threaded, long-running environments like Streamlit, 
    calling `asyncio.run(coro)` directly will raise a "RuntimeError: This event loop 
    is already running" if the thread is bound to a running loop. Using a 
    ThreadPoolExecutor to execute the loop in a separate, isolated thread completely 
    circumvents this issue and guarantees safe synchronous execution.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(lambda: asyncio.run(coro))
            return future.result()
    else:
        return loop.run_until_complete(coro)

async def async_steelman_hypothesis(captured_hyp, archaeology_rep) -> str:
    """
    Asynchronously run the Steelman Agent using the ADK framework and the built-in google_search tool.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("GEMINI_API_KEY is not set. Please update the .env file.")

    # Attempt to run using the ADK agent framework with Google Search
    try:
        agent = LlmAgent(
            name="steelman_agent",
            model="gemini-2.5-flash",
            instruction=(
                "You are the Steelman Agent. Your job is to take a proposed hypothesis and the "
                "historical lessons of similar failures, and reconstruct the hypothesis in its strongest, "
                "most viable form. You must focus entirely on the provided hypothesis and must not "
                "change the topic or introduce unrelated concepts under any circumstances.\n\n"
                "CRITICAL STEP:\n"
                "You must perform 2-3 targeted searches using the google_search tool to find supporting "
                "evidence, recent scientific findings, papers, or tools related to the core claims of "
                "the hypothesis. Synthesize the real search results into your steelman argument and "
                "explicitly reference specific evidence or information from those searches in your output."
            ),
            tools=[google_search]
        )
        
        runner = InMemoryRunner(agent=agent)
        user_id = "user"
        
        # Design decision: Generate a unique session ID for every execution (using uuid). 
        # In persistent app environments (like a running Streamlit app), reusing a static session 
        # ID causes the runner's session service to append subsequent, unrelated hypothesis inputs 
        # into the same thread history. This results in conversation history leaks where the model 
        # attempts to steelman previous topics. Generating a fresh UUID forces a clean conversation.
        session_id = f"steelman_{uuid.uuid4()}"
        
        await runner.session_service.create_session(
            app_name=runner.app_name,
            user_id=user_id,
            session_id=session_id
        )
        
        hypothesis_text = (
            f"Title: {captured_hyp.title}\n"
            f"Core Idea: {captured_hyp.core_idea}\n"
            f"Rationale: {captured_hyp.rationale}\n"
            f"Domain: {captured_hyp.domain}"
        )
        
        prompt = (
            f"You are steelmanning the following specific hypothesis:\n{hypothesis_text}\n"
            f"Do not deviate from this topic under any circumstances. Your entire output must be about this exact hypothesis.\n\n"
            f"ORIGINAL HYPOTHESIS DETAILS:\n"
            f"Assumptions: {', '.join(captured_hyp.assumptions)}\n"
            f"Proposed Test: {captured_hyp.proposed_test}\n\n"
            f"HISTORICAL PRECEDENTS & LESSONS:\n"
            f"Summary: {archaeology_rep.historical_precedents_summary}\n"
            f"Recommendation: {archaeology_rep.recommendation_from_history}\n\n"
            f"Please search for supporting evidence for this hypothesis's claims, perform 2-3 targeted Google searches, "
            f"and then construct your steelman argument, explicitly referencing the real search results and evidence found."
        )
        
        # Design decision: We explicitly set role="user" in the Content object. 
        # In the ADK framework, the contents request processor (`_get_contents`) skips any session 
        # event that does not have an explicitly set role. If we omit role="user", the runner 
        # ignores the user's prompt entirely, leading the agent to receive an empty prompt.
        message = types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
        agent_response_text = ""
        
        try:
            async for event in runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=message
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    agent_response_text = "".join(part.text for part in event.content.parts if part.text)
        finally:
            await runner.close()
            
        if agent_response_text:
            return agent_response_text
            
    except Exception as e:
        # Design decision: Log the error to stderr but do not crash the pipeline. 
        # If there are any issues executing the ADK agent, we want a graceful fallback.
        print(f"WARNING - ADK google_search FAILED: {repr(e)}", file=sys.stderr)
        
    # Fallback: Run standard Gemini generation if ADK run fails.
    # This guarantees the pipeline remains operational even if ADK execution fails.
    client = genai.Client(api_key=api_key)
    hypothesis_text = (
        f"Title: {captured_hyp.title}\n"
        f"Core Idea: {captured_hyp.core_idea}\n"
        f"Rationale: {captured_hyp.rationale}\n"
        f"Domain: {captured_hyp.domain}"
    )
    prompt = (
        f"You are steelmanning the following specific hypothesis:\n{hypothesis_text}\n"
        f"Do not deviate from this topic under any circumstances. Your entire output must be about this exact hypothesis.\n\n"
        f"ORIGINAL HYPOTHESIS DETAILS:\n"
        f"Assumptions: {', '.join(captured_hyp.assumptions)}\n"
        f"Proposed Test: {captured_hyp.proposed_test}\n\n"
        f"HISTORICAL PRECEDENTS & LESSONS:\n"
        f"Summary: {archaeology_rep.historical_precedents_summary}\n"
        f"Recommendation: {archaeology_rep.recommendation_from_history}\n\n"
        f"Please steelman this hypothesis. Refine it to be as robust and immune to past failures as possible."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are the Steelman Agent. Your job is to take a proposed hypothesis and the "
                "historical lessons similar failures, and reconstruct the hypothesis in its strongest, "
                "most viable form. You must focus entirely on the provided hypothesis and must not "
                "change the topic or introduce unrelated concepts under any circumstances."
            )
        )
    )
    return response.text

def steelman_hypothesis(captured_hyp, archaeology_rep) -> SteelmanReport:
    """
    Refine and strengthen the hypothesis using Gemini ADK with Google Search, 
    taking into account past failures.
    
    Args:
        captured_hyp (CapturedHypothesis): The structured hypothesis.
        archaeology_rep (ArchaeologyReport): The historical analysis report.
        
    Returns:
        SteelmanReport: The steelmanned version of the hypothesis.
        
    Design decision: Two-Stage Execution (Search then Format)
    In the ADK framework, an LlmAgent that is configured with tools cannot have an 
    `output_schema` parameter (it will throw a validation error). To work around this, 
    we first run the agent as a text-only agent to perform searches and synthesize the 
    result, and then perform a second structuring call using `client.models.generate_content` 
    with `response_schema=SteelmanReport` to format the synthesis into our structured Pydantic model.
    """
    # 1. Run ADK agent logic to perform search and generate synthesized steelman text
    agent_response_text = run_coroutine_sync(async_steelman_hypothesis(captured_hyp, archaeology_rep))
    
    # 2. Format the response text into the structured SteelmanReport schema
    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    structuring_prompt = (
        f"Based on the Steelman Agent's analysis and search findings:\n\n"
        f"{agent_response_text}\n\n"
        f"Please structure this into the SteelmanReport JSON format."
    )
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=structuring_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=SteelmanReport,
            system_instruction=(
                "You are a strict data formatter. Map the provided analysis into the requested JSON schema. "
                "Ensure all fields of the schema are populated accurately."
            )
        )
    )
    
    return SteelmanReport.model_validate_json(response.text)
