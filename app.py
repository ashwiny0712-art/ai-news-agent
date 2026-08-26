import streamlit as st
import asyncio
import time

# 1. Import updated backend function
from news_agent import run_agent

# 2. Page Configuration
st.set_page_config(
    page_title="AI Strategic News Agent",
    page_icon="🤖",
    layout="wide"
)

# 3. Apply Custom CSS Palette
st.markdown("""
    <style>
    .stApp {
        background-color: #FAF9F6 !important;
        color: #111827 !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #111827 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #7B6F6D !important;
        border-right: 1px solid #C4BDAC;
    }

    section[data-testid="stSidebar"] * {
        color: #FAF9F6 !important;
    }

    div[data-testid="stSidebar"] button {
        background-color: #005697 !important;
        color: #FAF9F6 !important;
        border: none !important;
        border-radius: 8px !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatar-user"]) {
        background-color: #005697 !important;
        color: #FAF9F6 !important;
        border-radius: 8px;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatar-user"]) * {
        color: #FAF9F6 !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatar-assistant"]) {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px;
    }

    div[data-testid="stChatMessage"] h3 {
        color: #005697 !important;
        border-bottom: 2px solid #FF4500;
        padding-bottom: 4px;
    }

    div[data-testid="stChatMessage"] strong {
        color: #FF4500 !important;
    }

    a {
        color: #005697 !important;
        font-weight: bold;
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# 4. State Management
if "chats" not in st.session_state:
    st.session_state.chats = {
        1: {
            "title": "Default Topic Analysis",
            "messages": [{"role": "assistant",
                          "content": "Hello! Ask me about current live events, news, or situation reports."}]
        }
    }
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = 1


def create_new_chat():
    new_id = len(st.session_state.chats) + 1
    st.session_state.chats[new_id] = {
        "title": f"New Chat {new_id}",
        "messages": [
            {"role": "assistant", "content": "Hello! Ask me about current live events, news, or situation reports."}]
    }
    st.session_state.active_chat_id = new_id


# 5. Sidebar Navigation
with st.sidebar:
    st.header("Chat History")
    if st.button("➕ New Chat", key="btn_new_chat", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()

    st.divider()

    for chat_id, chat_data in st.session_state.chats.items():
        btn_type = "primary" if chat_id == st.session_state.active_chat_id else "secondary"
        if st.button(f"💬 {chat_data['title']}", key=f"chat_{chat_id}", type=btn_type, use_container_width=True):
            st.session_state.active_chat_id = chat_id
            st.rerun()

current_chat = st.session_state.chats[st.session_state.active_chat_id]

# 6. Main Interface
st.title("🤖 AI Strategic News Agent")
st.caption("Powered by Mistral AI, Tavily Search & ChromaDB Memory")

chat_container = st.container()

with chat_container:
    for message in current_chat["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 7. Processing Pipeline
if prompt := st.chat_input("Ask about live news or events..."):
    current_chat["messages"].append({"role": "user", "content": prompt})

    if len(current_chat["messages"]) == 2:
        current_chat["title"] = prompt[:20] + "..." if len(prompt) > 20 else prompt

    with chat_container:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.status("🔍 Agent searching & analyzing live news...", expanded=True) as status:
                st.write("Fetching real-time feeds & executing vector search...")

                config = {"configurable": {"thread_id": f"session_{st.session_state.active_chat_id}"}}

                # Fetch payload containing both final text and retrieved sources
                result = asyncio.run(run_agent(prompt, config))
                agent_output = result["text"]
                sources = result["sources"]

                if sources:
                    st.write(f" Found {len(sources)} news articles via Tavily API.")

                status.update(label="Analysis Complete!", state="complete", expanded=False)


            # Stream response text smoothly
            def stream_response():
                for word in agent_output.split(" "):
                    yield word + " "
                    time.sleep(0.015)


            full_response = st.write_stream(stream_response)

            # Render clickable source expanders if articles were retrieved
            if sources:
                with st.expander("🔗 View Extracted Source Documents & Headlines"):
                    seen = set()
                    for src in sources:
                        if src["url"] not in seen:
                            st.markdown(f"**[{src['title']}]({src['url']})**")
                            st.caption(f"{src['content'][:180]}...")
                            st.divider()
                            seen.add(src["url"])

    current_chat["messages"].append({"role": "assistant", "content": full_response})