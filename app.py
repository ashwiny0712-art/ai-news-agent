import streamlit as st
import asyncio
import time

# 1. Import backend agent execution function
from news_agent import run_agent

# 2. Page Configuration
st.set_page_config(
    page_title="Secure Clarity - AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 3. State Management
if "chats" not in st.session_state:
    st.session_state.chats = {
        1: {
            "title": "Recent Chat",
            "messages": []
        }
    }
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = 1


def create_new_chat():
    new_id = len(st.session_state.chats) + 1
    st.session_state.chats[new_id] = {
        "title": f"Chat {new_id}",
        "messages": []
    }
    st.session_state.active_chat_id = new_id


# 4. Sidebar Navigation (Pure Python Layout)
with st.sidebar:
    # Sidebar Header Branding using native layout columns
    col_icon, col_text = st.columns([1, 3], vertical_alignment="center")
    with col_icon:
        st.button("SC", disabled=True, type="primary")
    with col_text:
        st.subheader("Secure Clarity")
        st.caption("Professional AI Agent")

    st.write("")

    # New Chat Primary Action Button
    if st.button("＋ New Chat", key="btn_new_chat", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()

    st.divider()

    # Chat History List
    st.caption("Recent Chats")
    for chat_id, chat_data in st.session_state.chats.items():
        is_active = chat_id == st.session_state.active_chat_id
        btn_type = "primary" if is_active else "secondary"

        if st.button(f"🕒 {chat_data['title']}", key=f"chat_{chat_id}", type=btn_type, use_container_width=True):
            st.session_state.active_chat_id = chat_id
            st.rerun()

current_chat = st.session_state.chats[st.session_state.active_chat_id]

# 5. Main Interface Container
chat_container = st.container()

with chat_container:
    # Initial Hero / Empty State UI
    if len(current_chat["messages"]) == 0:
        st.space = st.write("")

        # Center Icon & Title
        _, center_col, _ = st.columns([1, 2, 1])
        with center_col:
            st.title("How can I help you?")
            st.write("")

        # Suggestion Action Cards using Native Button Columns
        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                    "📰  Today's News\n\nWant to know about today's news?",
                    key="card_news",
                    use_container_width=True
            ):
                st.session_state.suggested_prompt = "What are today's top live news stories?"
                st.rerun()

        with col2:
            if st.button(
                    "📑  Summarize Data\n\nAnalyze a recent report or document.",
                    key="card_summary",
                    use_container_width=True
            ):
                st.session_state.suggested_prompt = "Summarize recent strategic intelligence data."
                st.rerun()

    else:
        # Display Message History
        for message in current_chat["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

# Check for suggestion clicks
input_default = st.session_state.pop("suggested_prompt", None)

# 6. Processing Pipeline & Input Bar
prompt = st.chat_input("Message Secure Clarity...")
if input_default and not prompt:
    prompt = input_default

if prompt:
    current_chat["messages"].append({"role": "user", "content": prompt})

    # Auto-generate topic title on first input
    if len(current_chat["messages"]) == 1:
        current_chat["title"] = prompt[:20] + "..." if len(prompt) > 20 else prompt

    with chat_container:
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.status("🔍 Agent searching & analyzing live news...", expanded=True) as status:
                st.write("Fetching real-time feeds & executing vector search...")

                config = {"configurable": {"thread_id": f"session_{st.session_state.active_chat_id}"}}

                # Async execution wrapper
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(run_agent(prompt, config))
                loop.close()

                agent_output = result["text"]
                sources = result["sources"]

                if sources:
                    st.write(f"Extracted {len(sources)} source articles via Tavily API.")

                status.update(label="Analysis Complete!", state="complete", expanded=False)


            # Response streaming output
            def stream_response():
                for word in agent_output.split(" "):
                    yield word + " "
                    time.sleep(0.015)


            full_response = st.write_stream(stream_response)

            # Expandable references section
            if sources:
                with st.expander("🔗 View Extracted Source Documents"):
                    seen = set()
                    for src in sources:
                        if src["url"] not in seen:
                            st.write(f"**[{src['title']}]({src['url']})**")
                            st.caption(f"{src['content'][:180]}...")
                            st.divider()
                            seen.add(src["url"])

    current_chat["messages"].append({"role": "assistant", "content": full_response})
    st.rerun()