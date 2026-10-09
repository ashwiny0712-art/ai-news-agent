import os
import asyncio
import warnings
from typing import List, Dict, Any
from difflib import SequenceMatcher
import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

from langchain_core.tools import tool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

warnings.filterwarnings("ignore")

# 1. ENVIRONMENT VARIABLES SETUP
load_dotenv()

# Read from os.environ first, fallback to Streamlit Cloud Secrets
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY") or st.secrets.get("NVIDIA_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY")

if not NVIDIA_API_KEY or not TAVILY_API_KEY:
    raise ValueError("Missing API keys in environment variables or Streamlit Secrets.")

os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

SEARCH_HISTORY: List[str] = []
FETCHED_SOURCES: List[Dict[str, str]] = []  # Stores structured sources

# NVIDIA NIM Model endpoints with explicit tool-calling support
NVIDIA_MODELS = [
    "meta/llama-3.3-70b-instruct",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "meta/llama-3.1-70b-instruct"
]

# 2. VECTORSTORE & TOOLS
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma(collection_name="news_relational_memory", embedding_function=embeddings)


def calculate_similarity(text1: str, text2: str) -> float:
    return SequenceMatcher(None, text1, text2).ratio()


@tool
def recall_past_relations(query: str) -> str:
    """Checks vector store for prior conversation context."""
    results = vectorstore.similarity_search(query, k=2)
    if not results:
        return "No past relations found in long-term memory."
    return "\n---\n".join([f"Past Context Memory: {doc.page_content}" for doc in results])


@tool
def search_and_extract_news(query: str) -> str:
    """Searches real-time news via Tavily with 3-loop cap and similarity safeguards."""
    global SEARCH_HISTORY, FETCHED_SOURCES

    if len(SEARCH_HISTORY) >= 3:
        return "LIMIT_REACHED: Maximum 3 search loops completed. Halt tool calls and generate final response now."

    try:
        response = tavily_client.search(query=query, topic="news", max_results=3, include_answer=True)
        results = response.get('results', [])
        if not results:
            return "No recent news found for this topic."

        # Cache structured metadata for Streamlit rendering
        for r in results:
            FETCHED_SOURCES.append({
                "title": r.get('title', 'No Title'),
                "url": r.get('url', '#'),
                "content": r.get('content', '')
            })

        news_summary = [
            f"Title: {r.get('title')}\nURL: {r.get('url')}\nContent: {r.get('content')}\n"
            for r in results
        ]
        current_text = "\n---\n".join(news_summary)

        if SEARCH_HISTORY:
            last_text = SEARCH_HISTORY[-1]
            similarity = calculate_similarity(last_text, current_text)
            if similarity >= 0.90:
                return (
                    f"SIMILARITY_MATCH: Content is {similarity * 100:.1f}% identical to previous search. "
                    "Halt further tool calls and generate final response now."
                )

        SEARCH_HISTORY.append(current_text)
        return current_text
    except Exception as e:
        return f"Error fetching news: {str(e)}"


@tool
def save_to_memory(summary_content: str) -> str:
    """Saves synthesized insights to ChromaDB vector store."""
    vectorstore.add_texts(texts=[summary_content])
    return "Successfully persisted to long-term memory."


tools = [recall_past_relations, search_and_extract_news, save_to_memory]

system_prompt = (
    "You are an AI News & Tactical Advisory Agent powered by NVIDIA AI.\n\n"
    "OPERATIONAL INSTRUCTIONS:\n"
    "1. Always search for news when queried about events, weather, or situation reports.\n"
    "2. If a tool returns 'LIMIT_REACHED' or 'SIMILARITY_MATCH', DO NOT call tools again. Synthesize instantly.\n"
    "3. Structure your response strictly according to the following layout:\n"
    "   - Brief situational intro paragraph.\n"
    "   - ### 📌 WHAT YOU SHOULD KNOW\n"
    "   - ### 💡 RECOMMENDED ACTIONS\n"
    "   - ### 🤖 SUGGESTED NEXT PROMPTS\n"
)

memory = MemorySaver()


async def run_agent(user_input: str, config: dict) -> Dict[str, Any]:
    global SEARCH_HISTORY, FETCHED_SOURCES
    SEARCH_HISTORY = []
    FETCHED_SOURCES = []  # Clear previous call metadata

    last_exception = None

    for model_name in NVIDIA_MODELS:
        try:
            llm = ChatNVIDIA(
                model=model_name,
                api_key=NVIDIA_API_KEY,
                temperature=0
            )
            app = create_react_agent(
                model=llm,
                tools=tools,
                prompt=system_prompt,
                checkpointer=memory
            )

            response = await app.ainvoke({"messages": [("user", user_input)]}, config=config)

            # Extract final response content safely across different message formats
            last_msg = response["messages"][-1]
            final_message = last_msg.content

            if isinstance(final_message, list):
                final_message = "\n".join([item.get('text', '') for item in final_message if isinstance(item, dict)])

            # Explicitly append reference links to content payload if missing
            if FETCHED_SOURCES and "http" not in final_message:
                final_message += "\n\n### 🔗 RELEVANT SOURCES & HEADLINES\n"
                seen_urls = set()
                for src in FETCHED_SOURCES:
                    if src['url'] not in seen_urls:
                        final_message += f"* [{src['title']}]({src['url']})\n"
                        seen_urls.add(src['url'])

            return {
                "text": final_message,
                "sources": FETCHED_SOURCES
            }

        except Exception as e:
            last_exception = e
            # Log the exact failure in Streamlit Cloud logs
            print(f"Model {model_name} failed: {type(e).__name__} - {str(e)}")
            continue

    raise RuntimeError(f"NVIDIA API Error ({type(last_exception).__name__}): {str(last_exception)}")