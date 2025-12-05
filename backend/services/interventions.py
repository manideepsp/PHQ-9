from models.interventions_model import Interventions
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, distinct
from typing import List, Dict, Any, Optional
from qdrant_client import models
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from ollama import Client
from config import Config

# ====== CONFIG ======
# Using configuration from Config class
DATABASE_URL = Config.SQLALCHEMY_DATABASE_URI
QDRANT_HOST = Config.QDRANT_HOST
QDRANT_API_KEY = Config.QDRANT_API_KEY
QDRANT_PORT = Config.QDRANT_PORT
QDRANT_COLLECTION = Config.QDRANT_COLLECTION
EMBEDDING_MODEL = Config.EMBEDDING_MODEL
LLM_MODEL = Config.LLM_MODEL
LLM_HOST = Config.LLM_HOST
TOP_K = Config.TOP_K


# ====== DB & Qdrant Setup ======
def get_db_session():
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()

def get_qdrant_client():
    return QdrantClient(url=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY)

# ====== Embedding & LLM Clients ======
embedder = SentenceTransformer(EMBEDDING_MODEL)
llm_client = Client(host=LLM_HOST)


# ====== Helper Functions ======
def timeline_to_text(timeline: List[Dict]) -> str:
    """Convert timeline dict to human-readable text."""
    return "\n".join([
        f"Consultation {t.get('consultation_seq')}: "
        f"PHQ9={t.get('phq9_total_score')}, relapse={t.get('relapse')}, predicted={t.get('is_predicted')}"
        for t in timeline
    ])

def query_qdrant(query_text: str, top_k: int = TOP_K) -> List[str]:
    """Query Qdrant for top-k related documents."""
    vec = embedder.encode(query_text).tolist()
    client = get_qdrant_client()

    results = client.search(
        QDRANT_COLLECTION,   # collection_name positional
        vec,                 # vector positional
        limit=top_k,
        with_payload=True,
    )

    return [hit.payload.get("text", "") for hit in results]



def build_prompt(timeline: List[Dict], retrieved_docs: List[str]) -> str:
    """Construct prompt for LLM based on timeline + retrieved documents."""
    tl_text = timeline_to_text(timeline)
    context = "\n\n".join(retrieved_docs)
    return f"""
You are a compassionate clinical mental-health assistant.

### Patient Timeline:
{tl_text}

### Knowledge Context:
{context}

### Task:
1. Summarize the recent emotional trend in 1–2 lines.
2. Suggest 2–3 empathetic, practical interventions.
3. Identify red-flag situations where professional help is needed.
4. Speak warmly, clearly, and non-judgmentally.

Output format:
- Summary of recent emotional trend:
- Suggested Practical Interventions:
  - 1.
  - 2.
  - 3.
- Red-Flag Situations where professional help is needed:
  - 1.
  - 2.
  - 3. 
- Closing supportive statement:
  Should be warm, supportive, and encouraging, nonjudgemental tone.
"""
def generate_text(prompt: str, system_prompt: Optional[str] = None) -> str:
    """Generate text using local Ollama LLM."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = llm_client.chat(model=LLM_MODEL, messages=messages)
        return response["message"]["content"]
    except Exception as e:
        # Log the exception and allow calling code to fallback
        print(f"⚠️ Ollama LLM request failed: {e}")
        raise


# ====== Main Function ======
def generate_intervention(timeline: List[Dict], user_id: int, phq9_assessment_id: Optional[int] = None) -> str:
    """
    Generate an intervention for a user based on their PHQ-9 timeline.
    Saves the result to the database.
    """
    # Use the last few consultations for context
    recent_timeline = timeline[-3:] if len(timeline) > 3 else timeline

    # Query Qdrant for knowledge augmentation
    query_text = timeline_to_text(recent_timeline)
    retrieved_docs = query_qdrant(query_text)

    # Build prompt
    prompt = build_prompt(timeline, retrieved_docs)

    # Generate intervention via local LLM
    # Attempt to generate via local LLM; if that fails, fall back to a simple template
    try:
        intervention_text = generate_text(prompt, system_prompt="You are a mental health expert.")
    except Exception as e:
        print(f"⚠️ LLM generation failed; falling back to a simple intervention text: {e}")

        # Create a conservative, safe fallback intervention summary using timeline + docs
        trend_summary = ''
        try:
            # Build a compact summary of the most recent timeline entries
            entries = recent_timeline if recent_timeline else timeline
            trend_summary = '; '.join([
                f"C{t.get('consultation_seq')}: score={t.get('phq9_total_score')}" for t in entries
            ])
        except Exception:
            trend_summary = 'No timeline available.'

        retrieved_context = '\n'.join(retrieved_docs) if retrieved_docs else 'No contextual docs available.'

        intervention_text = (
            f"Summary: Recent timeline — {trend_summary}\n\n"
            "Suggested interventions:\n"
            "1. Encourage regular check-ins and brief behavioral activation (e.g., small daily activities).\n"
            "2. Recommend staying connected with supportive friends/family and scheduling a follow-up appointment.\n"
            "3. If symptoms worsen or suicidal ideation appears, seek urgent professional help or emergency services.\n\n"
            "Context used:\n" + retrieved_context + "\n\n"
            "Closing: Be compassionate, validate feelings, and highlight small, achievable steps forward."
        )

    # Save to database
    session = get_db_session()
    new_intervention = Interventions(
        user_id=user_id,
        phq9_assessment_id=phq9_assessment_id,
        intervention=intervention_text
    )
    session.add(new_intervention)
    session.commit()
    session.close()

    return intervention_text



