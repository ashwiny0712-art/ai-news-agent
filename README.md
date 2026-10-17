# 📰 Secure Clarity: Agentic AI News Agent

An autonomous, vector-backed AI agent designed to aggregate, analyze, and synthesize real-time global news. Built with a ReAct-based agentic workflow, it dynamically queries live sources, prevents redundant API loops, and maintains long-term relational memory for deep context.

## ✨ Key Features
- **Real-Time Intelligence:** Live web search and structured data extraction via the Tavily API.
- **Multi-Model Backend:** Powered by OpenRouter for dynamic, reliable access to advanced models including Llama 3.3, Gemini 2.0, and DeepSeek.
- **Long-Term Memory:** Semantic similarity search using ChromaDB and HuggingFace (`all-MiniLM-L6-v2`) embeddings to recall past conversational context.
- **Agentic Routing:** Built on LangGraph to autonomously decide when to search the web, synthesize data, or recall historical facts.
- **Interactive UI:** Clean, responsive, and asynchronous-safe web interface built with Streamlit.

## 🛠️ Tech Stack
- **Language:** Python
- **Frameworks:** Streamlit, LangChain, LangGraph
- **Database:** ChromaDB (Local Vector Store)
- **APIs:** OpenRouter (LLM API Routing), Tavily (Real-Time Search)

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone [https://github.com/ashwiny0712-art/ai-news-agent.git](https://github.com/ashwiny0712-art/ai-news-agent.git)
cd ai-news-agent