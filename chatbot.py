# chatbot.py
from openai import OpenAI  # Keep for future use
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import pandas as pd
from google import genai
from google.genai import types
import os
import time
import ui

load_dotenv()
# openai_key = os.getenv("OPENAI_API_KEY")  # Uncomment for future OpenAI use

GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
LLM_MODELS = {
    "Gemini 2.5 Flash": "gemini-2.5-flash",
    "Gemini 2.5 Pro": "gemini-2.5-pro",
}

_gemini_client = None


def get_gemini_client():
    """Create the Gemini client on first use (needs GEMINI_API_KEY)."""
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini API key not found. Set the GEMINI_API_KEY environment "
                "variable (or add it to a .env file)."
            )
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def embed_with_gemini(texts, task_type):
    """Embed a list of texts with Gemini (batched to stay within API limits)."""
    client = get_gemini_client()
    vectors = []
    for start in range(0, len(texts), 100):
        batch = list(texts[start:start + 100])
        result = client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        vectors.extend(e.values for e in result.embeddings)
    return vectors

# Load facts
df = pd.read_csv("space_facts.csv")

# Initialize Chroma client and collection
chroma_client = chromadb.PersistentClient('./db/myfactsdb')
collection_name = "MySpaceFacts"

# Precompute embeddings once
def generate_embeddings(embedding_name):
    # if embedding_name=='OpenAI Embedding':
    #     embedding = embedding_functions.OpenAIEmbeddingFunction(
    #         api_key=openai_key,
    #         model_name="text-embedding-3-small"
    #     )
    if embedding_name == 'Chroma Default':
        return embedding_functions.DefaultEmbeddingFunction()
    # Gemini embeddings are computed explicitly (see embed_with_gemini), so the
    # collection is created without a Chroma embedding function.
    return None


def get_collection(embedding_name):
    """Return the facts collection for the selected embedding model.

    Each embedding model gets its own collection because vectors from
    different models have different dimensions and cannot be mixed.
    """
    use_gemini = embedding_name != 'Chroma Default'
    col = chroma_client.get_or_create_collection(
        name=collection_name + ("_gemini" if use_gemini else ""),
        embedding_function=generate_embeddings(embedding_name),
    )
    ids = [str(i) for i in df['id']]
    facts = list(df['fact'])
    if use_gemini:
        # Only call the embedding API when the collection is not populated yet.
        if col.count() != len(ids):
            col.upsert(
                ids=ids,
                documents=facts,
                embeddings=embed_with_gemini(facts, "RETRIEVAL_DOCUMENT"),
            )
    else:
        col.upsert(ids=ids, documents=facts)
    return col


# Initialize LLM client
def initialize_client(model_name):
    # if model_name=='OpenAI GPT-4':
    #     client = OpenAI(api_key=openai_key)
    #     model = "gpt-3.5-turbo"
    model = LLM_MODELS.get(model_name, "gemini-2.5-flash")
    return get_gemini_client(), model

# Query relevant chunks from Chroma
def query_documents(question, n_results=2, max_chunk_length=500):
    if embedding_name == 'Chroma Default':
        results = collection.query(query_texts=question, n_results=n_results)
    else:
        results = collection.query(
            query_embeddings=embed_with_gemini([question], "RETRIEVAL_QUERY"),
            n_results=n_results,
        )
    # Truncate chunks to reduce prompt size
    truncated = [doc[:max_chunk_length] for doc in results['documents'][0]]
    return truncated

# Generate a streamed response from Gemini
def generate_response(question, relevant_chunks, client, model):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        f'''You are a chatbot who has a wide knowledge of space. Answer the query from the context provided in an informative way. Produce consise response with 5 lines and dont take more than 10 seconds to generate response and provide the source links from where you have taken info in the end of the response\n"
        f"Context: {context}\nQuery: {question}
        '''
    )

    stream = client.models.generate_content_stream(
        model=model,
        contents=question,
        config=types.GenerateContentConfig(system_instruction=prompt),
    )
    # Yield plain text chunks so the UI can stream them as they arrive
    for chunk in stream:
        if chunk.text:
            yield chunk.text

embedding_name = ui.embedmodel
model_name = ui.modeltype
collection = get_collection(embedding_name)
client, model = initialize_client(model_name)

# Main function called by UI
def run_chatbot(question):
    # Only the query embedding is dynamic; documents are pre-embedded

    relevant_chunks = query_documents(question)

    response = generate_response(question, relevant_chunks, client, model)

    return response, relevant_chunks
    