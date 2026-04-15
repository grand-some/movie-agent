import os
import time
import uuid
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from openai import OpenAI
from agents import Agent, Runner, SQLiteSession, FileSearchTool, WebSearchTool

# -----------------------------
# OpenAI client
# -----------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# -----------------------------
# Helpers: journal and vector store
# -----------------------------

def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "안녕하세요! 저는 당신의 Life Coach예요. 목표 문서를 업로드하고, 일기를 남기면 그 기록을 참고해 더 개인화된 조언을 드릴게요.",
            }
        ]

    if "journal_entries" not in st.session_state:
        st.session_state.journal_entries = []

    if "vector_store_id" not in st.session_state:
        st.session_state.vector_store_id = None

    if "uploaded_goal_file_name" not in st.session_state:
        st.session_state.uploaded_goal_file_name = None

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"life-coach-{uuid.uuid4()}"

    if "last_uploaded_goal_bytes" not in st.session_state:
        st.session_state.last_uploaded_goal_bytes = None

    if "last_uploaded_goal_type" not in st.session_state:
        st.session_state.last_uploaded_goal_type = None


def build_journal_text(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return "아직 저장된 일기 항목이 없습니다."

    lines = [
        "# Personal Progress Journal",
        "이 문서는 Life Coach Agent가 사용자의 진행 상황을 시간순으로 파악하기 위한 일기 기록입니다.",
        "",
    ]

    for idx, entry in enumerate(entries, start=1):
        lines.extend(
            [
                f"## Entry {idx}",
                f"Date: {entry['date']}",
                f"Area: {entry['area']}",
                f"Progress Score: {entry['score']}",
                f"Content: {entry['content']}",
                "",
            ]
        )

    return "\n".join(lines)


def upload_bytes_as_openai_file(file_bytes: bytes, filename: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{filename}") as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        openai_file = client.files.create(file=f, purpose="assistants")

    return openai_file


def create_journal_file(entries: List[Dict[str, Any]]):
    journal_text = build_journal_text(entries)
    journal_bytes = journal_text.encode("utf-8")
    return upload_bytes_as_openai_file(journal_bytes, "journal_entries.txt")


def wait_until_vector_store_ready(vector_store_id: str, timeout_seconds: int = 120) -> None:
    start_time = time.time()

    while time.time() - start_time < timeout_seconds:
        file_list = client.vector_stores.files.list(vector_store_id=vector_store_id)
        data = getattr(file_list, "data", [])

        if not data:
            time.sleep(2)
            continue

        statuses = [getattr(item, "status", None) for item in data]

        if all(status == "completed" for status in statuses):
            return

        if any(status == "failed" for status in statuses):
            raise RuntimeError("벡터 스토어에 파일을 처리하는 중 일부 파일 업로드가 실패했습니다.")

        time.sleep(2)

    raise TimeoutError("벡터 스토어 파일 인덱싱이 시간 내에 완료되지 않았습니다.")


def rebuild_vector_store(goal_file_bytes: bytes, goal_filename: str, journal_entries: List[Dict[str, Any]]) -> str:
    goal_file = upload_bytes_as_openai_file(goal_file_bytes, goal_filename)
    journal_file = create_journal_file(journal_entries)

    vector_store = client.vector_stores.create(name="life_coach_goals_and_journal")

    client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=goal_file.id,
    )
    client.vector_stores.files.create(
        vector_store_id=vector_store.id,
        file_id=journal_file.id,
    )

    wait_until_vector_store_ready(vector_store.id)
    return vector_store.id


# -----------------------------
# Agent factory
# -----------------------------

def build_life_coach_agent(vector_store_id: Optional[str]) -> Agent:
    tools = [WebSearchTool()]

    if vector_store_id:
        tools.append(
            FileSearchTool(
                max_num_results=5,
                vector_store_ids=[vector_store_id],
            )
        )

    instructions = """
You are a warm, encouraging, practical life coach.

Your job:
- Help the user with motivation, self-development, habits, consistency, mindset, and goal achievement.
- When a file search tool is available, search the uploaded goals and journal before giving advice about progress, consistency, or personal plans.
- Use the user's past goals and diary entries to personalize advice.
- When helpful, use web search to combine personal context with fresh evidence, techniques, or motivational advice.
- If both file search and web search are used, first summarize what the user's goals/history say, then connect it to practical outside advice.
- Track progress over time by comparing recent journal entries with older entries.
- Always sound supportive and clear.
- Prefer this response structure:
  1. Encouragement
  2. What I found from your goals/journal
  3. What outside best practices suggest
  4. A small next-step plan for today or this week
- If no file search tool is available, be honest that no goals document has been uploaded yet and provide general coaching advice.
"""

    return Agent(
        name="Life Coach with Memory",
        model="gpt-5.4",
        instructions=instructions,
        tools=tools,
    )


# -----------------------------
# Streamlit app
# -----------------------------
init_state()
st.set_page_config(page_title="Life Coach Agent + File Search", page_icon="🎯", layout="centered")
st.title("🎯 Life Coach Agent + File Search")
st.caption("목표 문서와 일기 항목을 기억하고, 웹 검색까지 결합하는 개인 코치")

# Session object for automatic conversation memory
agent_session = SQLiteSession(st.session_state.session_id)

with st.sidebar:
    st.header("1) 목표 문서 업로드")
    uploaded_goal = st.file_uploader(
        "PDF 또는 TXT 파일 업로드",
        type=["pdf", "txt"],
        accept_multiple_files=False,
    )

    if uploaded_goal is not None:
        st.session_state.last_uploaded_goal_bytes = uploaded_goal.getvalue()
        st.session_state.last_uploaded_goal_type = uploaded_goal.type
        st.session_state.uploaded_goal_file_name = uploaded_goal.name

        if st.button("목표 문서 + 일기 기록 인덱싱하기"):
            with st.spinner("목표 문서와 일기 기록을 검색 가능하게 준비하고 있어요..."):
                try:
                    st.session_state.vector_store_id = rebuild_vector_store(
                        goal_file_bytes=st.session_state.last_uploaded_goal_bytes,
                        goal_filename=st.session_state.uploaded_goal_file_name,
                        journal_entries=st.session_state.journal_entries,
                    )
                    st.success("업로드 완료! 이제 코치가 목표와 기록을 검색해서 참고할 수 있어요.")
                except Exception as e:
                    st.error(f"인덱싱 중 오류가 발생했습니다: {e}")

    if st.session_state.uploaded_goal_file_name:
        st.info(f"현재 목표 문서: {st.session_state.uploaded_goal_file_name}")

    st.divider()
    st.header("2) 일기 / 진행 상황 기록")
    diary_area = st.selectbox(
        "분야 선택",
        ["운동", "공부", "업무", "수면", "식단", "마음가짐", "기타"],
    )
    diary_score = st.slider("오늘의 진행 점수", min_value=0, max_value=100, value=60, step=5)
    diary_content = st.text_area(
        "오늘의 기록",
        placeholder="예: 이번 주 운동 2번 했고, 어제는 피곤해서 쉬었습니다. 다음엔 월/수/금으로 고정해 보려고 합니다.",
        height=120,
    )

    if st.button("일기 저장"):
        if diary_content.strip():
            st.session_state.journal_entries.append(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "area": diary_area,
                    "score": diary_score,
                    "content": diary_content.strip(),
                }
            )
            st.success("일기 항목이 저장되었습니다.")
        else:
            st.warning("일기 내용을 입력해 주세요.")

    if st.button("일기 반영하여 재인덱싱"):
        if st.session_state.last_uploaded_goal_bytes and st.session_state.uploaded_goal_file_name:
            with st.spinner("최신 일기까지 반영해서 다시 인덱싱하고 있어요..."):
                try:
                    st.session_state.vector_store_id = rebuild_vector_store(
                        goal_file_bytes=st.session_state.last_uploaded_goal_bytes,
                        goal_filename=st.session_state.uploaded_goal_file_name,
                        journal_entries=st.session_state.journal_entries,
                    )
                    st.success("최신 일기까지 반영되었습니다.")
                except Exception as e:
                    st.error(f"재인덱싱 중 오류가 발생했습니다: {e}")
        else:
            st.warning("먼저 목표 문서를 업로드해 주세요.")

    st.divider()
    st.header("3) 최근 일기")
    if st.session_state.journal_entries:
        for entry in reversed(st.session_state.journal_entries[-5:]):
            st.markdown(
                f"**{entry['date']} | {entry['area']} | 점수 {entry['score']}**\n\n{entry['content']}"
            )
    else:
        st.write("아직 저장된 일기가 없습니다.")

    st.divider()
    st.header("예시 질문")
    st.markdown(
        """
- 내 운동 목표 달성은 잘 되어가고 있어?
- 내 최근 일기를 보면 어떤 패턴이 보여?
- 내 목표를 기준으로 이번 주 실행 계획을 짜줘
- 목표와 최근 기록을 보고 동기부여 조언을 해줘
        """
    )


# Chat history render
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_prompt = st.chat_input("예: 내 운동 목표 달성은 잘 되어가고 있어?")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("코치가 목표 문서, 일기, 웹 정보를 함께 살펴보고 있어요..."):
            try:
                agent = build_life_coach_agent(st.session_state.vector_store_id)
                result = Runner.run_sync(agent, user_prompt, session=agent_session)
                assistant_reply = result.final_output
            except Exception as e:
                assistant_reply = (
                    "문제가 발생했어요. 아래를 확인해 주세요.\n\n"
                    "- OPENAI_API_KEY 설정 여부\n"
                    "- openai, openai-agents, streamlit 설치 여부\n"
                    "- 목표 문서를 인덱싱했는지 여부\n"
                    f"- 에러: `{e}`"
                )
        st.markdown(assistant_reply)

    st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
