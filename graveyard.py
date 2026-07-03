import os
import uuid
import datetime
import chromadb
from google import genai
from google.genai import errors
from dotenv import load_dotenv

# Load environment variables (such as GEMINI_API_KEY) from the local .env file.
# Design decision: dot-env is used to keep configuration externalized and prevent
# API keys from being hardcoded in version control.
load_dotenv()

# Initialize ChromaDB.
# Design decision: We use a persistent local client stored in a 'chroma_db' directory 
# inside the workspace. This keeps the vector database fully self-contained, lightweight, 
# and easy to run/deploy locally without needing to configure or maintain an external 
# database server (like Pinecone or pgvector).
CHROMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="hypotheses")

def _get_client():
    """
    Internal helper to construct and return the Gemini GenAI Client.
    
    Design decision: Having a single centralized helper avoids code duplication 
    across functions and ensures consistent verification that the API key is 
    correctly configured before any API call is made.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError(
            "GEMINI_API_KEY is not set. Please update the .env file with a valid Gemini API key."
        )
    return genai.Client(api_key=api_key)

def _get_embedding(text: str) -> list[float]:
    """
    Generate an embedding using Gemini's gemini-embedding-001 model.
    
    Design decision: We use gemini-embedding-001 because it provides high-quality 
    semantic representations that map well to domain concepts in both Software 
    Engineering and Quant Finance. This enables accurate semantic search inside the graveyard.
    """
    client = _get_client()
    try:
        response = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text,
        )
        if not response.embeddings:
            raise ValueError("No embeddings returned from the API.")
        return response.embeddings[0].values
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {e}")

def store_hypothesis(
    text: str,
    domain: str,
    outcome: str,
    conviction_score: float,
    notes: str,
    contributor: str = "Anonymous",
    date: str = None,
    initial_conviction: float = None,
    time_invested: str = None,
    resources_used: str = None,
    impact: str = None,
    repeat_risk: str = None,
    agent_capture_json: str = None,
    agent_archaeology_json: str = None,
    agent_steelman_json: str = None,
    agent_critic_json: str = None,
    agent_verdict_json: str = None
) -> str:
    """
    Store a hypothesis in the local ChromaDB vector store.
    
    Args:
        text (str): The hypothesis text.
        domain (str): The domain (e.g., 'Quant Finance', 'Software Engineering').
        outcome (str): The outcome (e.g., 'Success', 'Failed', 'Tested').
        conviction_score (float): The conviction score (0-100).
        notes (str): Additional notes/lessons learned.
        contributor (str): The name of the contributor.
        date (str): The date the hypothesis was buried. If None, current date is used.
        initial_conviction (float): The initial conviction score.
        time_invested (str): Estimated time invested.
        resources_used (str): Estimated resources used.
        impact (str): Estimated impact.
        repeat_risk (str): Estimated risk of repeat failure.
        agent_capture_json (str): Serialized Capture Agent report.
        agent_archaeology_json (str): Serialized Archaeology Agent report.
        agent_steelman_json (str): Serialized Steelman Agent report.
        agent_critic_json (str): Serialized Critic Agent report.
        agent_verdict_json (str): Serialized Verdict Agent report.
        
    Returns:
        str: The generated unique document ID.
    """
    # 1. Generate embedding vector for semantic lookup
    embedding = _get_embedding(text)
    
    # 2. Generate a random UUID to serve as the document key in the collection
    doc_id = str(uuid.uuid4())
    
    if not date:
        date = datetime.datetime.now().strftime("%Y-%m-%d")
        
    meta = {
        "domain": domain,
        "outcome": outcome,
        "conviction_score": float(conviction_score),
        "notes": notes,
        "contributor": contributor,
        "date": date
    }
    
    if initial_conviction is not None:
        meta["initial_conviction"] = float(initial_conviction)
    if time_invested is not None:
        meta["time_invested"] = time_invested
    if resources_used is not None:
        meta["resources_used"] = resources_used
    if impact is not None:
        meta["impact"] = impact
    if repeat_risk is not None:
        meta["repeat_risk"] = repeat_risk
        
    if agent_capture_json is not None:
        meta["agent_capture_json"] = agent_capture_json
    if agent_archaeology_json is not None:
        meta["agent_archaeology_json"] = agent_archaeology_json
    if agent_steelman_json is not None:
        meta["agent_steelman_json"] = agent_steelman_json
    if agent_critic_json is not None:
        meta["agent_critic_json"] = agent_critic_json
    if agent_verdict_json is not None:
        meta["agent_verdict_json"] = agent_verdict_json
        
    collection.add(
        embeddings=[embedding],
        documents=[text],
        metadatas=[meta],
        ids=[doc_id]
    )
    return doc_id

def search_similar(text: str, n: int = 3) -> list[dict]:
    """
    Search for similar hypotheses in the graveyard.
    
    Args:
        text (str): The hypothesis text to query.
        n (int): The number of similar results to retrieve.
        
    Returns:
        list[dict]: A list of similar hypotheses with their metadata and distance.
        
    Design decision: We check the collection count first and cap the number of requested
    results (`n_results = min(n, count)`). This prevents ChromaDB from raising an error 
    if a user requests more results than are currently stored in the database.
    """
    # Check if there are any items in the collection first to avoid querying an empty database
    count = collection.count()
    if count == 0:
        return []
        
    embedding = _get_embedding(text)
    
    try:
        # Limit n to the number of items in the collection
        n_results = min(n, count)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
    except Exception as e:
        # Return empty list if query fails (e.g. schema mismatch or empty DB)
        return []
        
    formatted_results = []
    if not results or not results['documents'] or len(results['documents'][0]) == 0:
        return formatted_results
        
    # Map the raw ChromaDB results into a clean, easy-to-use Python list of dicts.
    # We include distance so callers can calculate the exact similarity percentage.
    for i in range(len(results['documents'][0])):
        distance = results['distances'][0][i] if 'distances' in results and results['distances'] else None
        formatted_results.append({
            "id": results['ids'][0][i],
            "text": results['documents'][0][i],
            "metadata": results['metadatas'][0][i],
            "distance": distance
        })
    return formatted_results
