import os
import re
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class CapturedHypothesis(BaseModel):
    """
    Structured Pydantic schema representing the parsed hypothesis.
    
    Design decision: By enforcing structured outputs on the Capture Agent, 
    we ensure the subsequent pipeline agents (Archaeology, Steelman, Verdict) 
    receive a strictly validated and predictable object instead of raw, unstructured text.
    """
    title: str = Field(description="A short, descriptive title for the hypothesis.")
    core_idea: str = Field(description="The main premise or core idea.")
    rationale: str = Field(description="The underlying reasoning or justification.")
    domain: str = Field(description="The domain or field of the hypothesis (e.g., 'Quant Finance', 'Software Engineering', 'Data Science', 'Computer Vision', 'NLP / LLMs', 'Infrastructure', etc.).")
    assumptions: list[str] = Field(description="List of key assumptions made.")
    proposed_test: str = Field(description="How to test/validate this hypothesis.")

# Common vocabulary list for basic English & Domain validation to filter pure gibberish.
# Design decision: Embedding a lightweight static vocabulary set is much faster and 
# doesn't require third-party dictionary packages (like NLTK or PyEnchant) that might 
# not be installed or available in the runtime environment.
COMMON_WORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", 
    "it", "for", "not", "on", "with", "he", "as", "you", "do", "at", 
    "this", "but", "his", "by", "from", "they", "we", "say", "her", 
    "she", "or", "an", "will", "my", "one", "all", "would", "there", 
    "their", "what", "so", "up", "out", "if", "about", "who", "get", 
    "which", "go", "me", "when", "make", "can", "like", "time", "no", 
    "just", "him", "know", "take", "people", "into", "year", "your", 
    "good", "some", "could", "them", "see", "other", "than", "then", 
    "now", "look", "only", "come", "its", "over", "think", "also", 
    "back", "after", "use", "two", "how", "our", "work", "first", 
    "well", "way", "even", "new", "want", "because", "any", "these", 
    "give", "day", "most", "us", "is", "are", "was", "were", "been", 
    "has", "had", "using", "model", "data", "system", "trade", "trading", 
    "test", "testing", "price", "market", "software", "engineering", 
    "code", "database", "query", "user", "app", "application", "server", 
    "client", "network", "file", "service", "process", "hello", "world",
    "predict", "predicting", "prediction", "forecast", "forecasting", "run",
    "implement", "implementing", "build", "building", "replace", "replacing",
    "create", "creating", "design", "designing", "quant", "finance", "stock",
    "futures", "options", "portfolio", "strategy", "algorithm", "backtest"
}

def capture_hypothesis(raw_text: str, domain_hint: str = "") -> CapturedHypothesis:
    """
    Parse and structure a raw hypothesis text using Gemini's structured output.
    Applies security guardrails at the start.
    
    Args:
        raw_text (str): The raw hypothesis text.
        
    Returns:
        CapturedHypothesis: The structured hypothesis.
        
    Design decisions:
    1. Fail-Fast Guardrails: Executed at the very beginning of the function 
       to prevent empty, malicious, or garbage inputs from triggering downstream 
       processes or incurring unnecessary LLM API costs.
    2. HTML Sanitization: We strip HTML tags using a regex scanner to prevent 
       rendering-based scripting attacks if the hypothesis is displayed in the Streamlit UI.
    3. Input Cap: Restricting the input to 2,000 characters prevents context-window 
       inflation and denial-of-service/heavy-payload attacks.
    4. Prompt Injection Scanner: We check for known system-override command 
       structures (case-insensitive) to prevent the user from hijacking the model's instructions.
    5. Gibberish Checker: Consonant-sequence and dictionary checks identify 
       completely random keyboard entries to ensure only actual proposals are parsed.
    """
    # 1. Sanitize: Strip HTML tags
    sanitized = re.sub(r'<[^>]*>', '', raw_text)
    # Limit to 2000 characters maximum
    if len(sanitized) > 2000:
        sanitized = sanitized[:2000]
        
    stripped = sanitized.strip()
    
    # Guardrail 1: Reject empty or less than 10 characters
    if not stripped:
        raise ValueError("Input rejected: The hypothesis cannot be empty.")
    if len(stripped) < 10:
        raise ValueError("Input rejected: The hypothesis must be at least 10 characters long.")
        
    # Guardrail 2: Reject prompt injection attempts
    injection_patterns = [
        "ignore previous instructions",
        "ignore the instructions",
        "ignore instructions",
        "ignore above",
        "forget previous instructions",
        "forget everything",
        "override instructions",
        "override system prompt",
        "you are now",
        "new role:",
        "system prompt"
    ]
    for pattern in injection_patterns:
        if pattern in stripped.lower():
            raise ValueError(f"Input rejected: Potential prompt injection attempt detected ('{pattern}').")
            
    # Guardrail 3: Reject pure gibberish with no real words
    words = re.findall(r'[a-zA-Z]+', stripped.lower())
    if not words:
        raise ValueError("Input rejected: Contains no readable words.")
        
    has_real_word = any(w in COMMON_WORDS for w in words)
    if not has_real_word:
        # Check if all words have no vowels (pure consonants)
        all_no_vowels = all(not any(c in 'aeiouy' for c in w) for w in words)
        if all_no_vowels:
            raise ValueError("Input rejected: Contains only consonant sequences (gibberish).")
        raise ValueError("Input rejected: Pure gibberish with no recognizable real words detected.")

    # Proceed to structured generation using sanitized input
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("GEMINI_API_KEY is not set. Please update the .env file.")
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"Please parse and structure the following raw hypothesis:\n\n{stripped}"
    if domain_hint:
        prompt += f"\n\nDomain Hint: The user specified that the domain/field of this hypothesis is '{domain_hint}'."
    
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CapturedHypothesis,
            system_instruction=(
                "You are the Capture Agent. Your job is to extract, clarify, and structure a raw, "
                "messy hypothesis into a clean, structured JSON format. Ensure you identify the domain / field "
                "correctly. If a Domain Hint is provided, prioritize using that as the domain."
            )
        )
    )
    
    return CapturedHypothesis.model_validate_json(response.text)
