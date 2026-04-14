import asyncio
import os
from typing import List, Dict, Any

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from openai import OpenAI
from agents import Agent, Runner, function_tool

# Optional: reduce tracing noise while developing.
try:
    from agents import set_tracing_disabled

    set_tracing_disabled(True)
except Exception:
    pass


# -----------------------------
# OpenAI clients
# -----------------------------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------
# Tool: web search
# -----------------------------
@function_tool
def web_search_coaching(query: str) -> str:
    """
    Search the web for motivational content, self-improvement tips,
    and habit-building advice. Use this whenever fresh or practical advice
    from the web would help the user.
    """
    response = openai_client.responses.create(
        model="gpt-5",
        tools=[{"type": "web_search"}],
        input=query,
    )

    # Simple return for the agent. If you want citations in the UI later,
    # you can parse response.output for annotations.
    return response.output_text


# -----------------------------
# Agent definition
# -----------------------------
life_coach_agent = Agent(
    name="Life Coach",
    model="gpt-5.4",
    instructions=(
        "You are a warm, encouraging life coach. "
        "Your job is to help the user with motivation, self-development, "
        "habit formation, productivity, mindset, and emotional encouragement.\n\n"
        "Behavior rules:\n"
        "1. Always sound supportive, practical, and optimistic.\n"
        "2. Give advice that is easy to act on immediately.\n"
        "3. For habit, motivation, self-improvement, or productivity questions, "
        "use the web_search_coaching tool when it would help to provide stronger, fresher, or more specific advice.\n"
        "4. After using the search tool, summarize the advice naturally instead of dumping raw search results.\n"
        "5. Structure answers clearly: encouragement first, then practical steps.\n"
        "6. Keep responses conversational but useful.\n"
        "7. Remember the ongoing conversation and refer to earlier user goals when helpful."
    ),
    tools=[web_search_coaching],
)


# -----------------------------
# Async runner helper
# -----------------------------
async def run_agent(user_input: str, history: List[Dict[str, str]]) -> str:
    """
    Build a conversation string from session memory and run the agent.
    This is a simple memory approach using Streamlit session state.
    """
    conversation_lines = []
    for message in history:
        role = "User" if message["role"] == "user" else "Coach"
        conversation_lines.append(f"{role}: {message['content']}")

    conversation_lines.append(f"User: {user_input}")
    full_prompt = "\n".join(conversation_lines)

    result = await Runner.run(life_coach_agent, full_prompt)
    return result.final_output


def run_async(coro):
    """
    Safe async execution helper for Streamlit.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Life Coach Agent", page_icon="💪", layout="centered")
st.title("💪 Life Coach Agent")
st.caption("동기부여, 자기계발, 습관 형성을 도와주는 코치")

# Session memory initialization
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "안녕하세요! 저는 당신의 라이프 코치예요. 목표, 습관, 동기부여에 대해 편하게 이야기해 주세요.",
        }
    ]


# Render chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Chat input
user_prompt = st.chat_input("예: 아침에 일찍 일어나는 습관을 만들고 싶어요")

if user_prompt:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Generate coach response
    with st.chat_message("assistant"):
        with st.spinner("코치가 생각하고 있어요..."):
            try:
                assistant_reply = run_async(
                    run_agent(user_prompt, st.session_state.messages[:-1])
                )
            except Exception as e:
                assistant_reply = (
                    "문제가 발생했어요. 아래 항목을 확인해 주세요.\n\n"
                    "- OPENAI_API_KEY가 설정되어 있는지\n"
                    "- openai, openai-agents, streamlit 패키지가 설치되어 있는지\n"
                    f"- 에러: `{e}`"
                )

        st.markdown(assistant_reply)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})


# -----------------------------
# Sidebar guide
# -----------------------------
with st.sidebar:
    st.header("실행 방법")
    st.code(
        """pip install streamlit openai openai-agents
export OPENAI_API_KEY=your_api_key
streamlit run life_coach_agent_streamlit.py""",
        language="bash",
    )

    st.subheader("예시 질문")
    st.markdown(
        """
- 아침에 일찍 일어나고 싶은데 자꾸 실패해
- 좋은 습관을 만들려면 어떻게 해야 해?
- 요즘 의욕이 너무 없는데 어떻게 해야 할까?
- 목표를 꾸준히 실천하는 방법을 알려줘
        """
    )
