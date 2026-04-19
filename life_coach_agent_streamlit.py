import os
import io
import time
import uuid
import base64
import tempfile
from datetime import datetime
from typing import List, Dict, Any, Optional


import streamlit as st
from PIL import Image
from openai import OpenAI
from agents import (
    Agent,
    Runner,
    SQLiteSession,
    FileSearchTool,
    WebSearchTool,
    ImageGenerationTool,
)

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY를 찾지 못했습니다. .env 파일 또는 환경변수를 확인하세요.")

client = OpenAI(api_key=api_key)

# ---------------------------------
# OpenAI client
# ---------------------------------
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------
# Streamlit session state
# ---------------------------------

def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": (
                    "안녕하세요! 저는 당신의 Life Coach Agent예요. "
                    "목표 문서를 참고하고, 웹에서 팁을 찾고, 필요하면 비전 보드나 동기부여 이미지도 만들어 드릴게요."
                ),
            }
        ]

    if "journal_entries" not in st.session_state:
        st.session_state.journal_entries = []

    if "vector_store_id" not in st.session_state:
        st.session_state.vector_store_id = None

    if "goal_filename" not in st.session_state:
        st.session_state.goal_filename = None

    if "goal_file_bytes" not in st.session_state:
        st.session_state.goal_file_bytes = None

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"life-coach-{uuid.uuid4()}"

    if "generated_images" not in st.session_state:
        st.session_state.generated_images = []


# ---------------------------------
# Helpers: journal -> text file
# ---------------------------------

def build_journal_text(entries: List[Dict[str, Any]]) -> str:
    if not entries:
        return "아직 저장된 일기 항목이 없습니다."

    lines = [
        "# Personal Progress Journal",
        "Life Coach Agent가 사용자의 진행 상황을 추적하기 위한 기록입니다.",
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


def upload_temp_file(file_bytes: bytes, filename: str):
    suffix = os.path.splitext(filename)[1] or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="assistants")

    return uploaded


def create_journal_file(entries: List[Dict[str, Any]]):
    journal_text = build_journal_text(entries)
    return upload_temp_file(journal_text.encode("utf-8"), "journal_entries.txt")


def wait_for_vector_store_ready(vector_store_id: str, timeout_seconds: int = 120) -> None:
    start = time.time()
    while time.time() - start < timeout_seconds:
        file_list = client.vector_stores.files.list(vector_store_id=vector_store_id)
        data = getattr(file_list, "data", [])

        if not data:
            time.sleep(2)
            continue

        statuses = [getattr(item, "status", None) for item in data]

        if all(status == "completed" for status in statuses):
            return
        if any(status == "failed" for status in statuses):
            raise RuntimeError("벡터 스토어 파일 처리 중 실패한 항목이 있습니다.")

        time.sleep(2)

    raise TimeoutError("벡터 스토어 인덱싱이 시간 내에 끝나지 않았습니다.")


def rebuild_vector_store(goal_bytes: bytes, goal_filename: str, journal_entries: List[Dict[str, Any]]) -> str:
    goal_file = upload_temp_file(goal_bytes, goal_filename)
    journal_file = create_journal_file(journal_entries)

    vector_store = client.vector_stores.create(name="life-coach-goals-and-journal")
    client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=goal_file.id)
    client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=journal_file.id)

    wait_for_vector_store_ready(vector_store.id)
    return vector_store.id


# ---------------------------------
# Agent factory
# ---------------------------------

def build_agent(vector_store_id: Optional[str]) -> Agent:
    tools = [
    WebSearchTool(),
    ImageGenerationTool(
        tool_config={
            "type": "image_generation",
            "size": "1024x1024",
            "quality": "medium",
            "background": "opaque",
        }
    ),
            ]      

    if vector_store_id:
        tools.append(
            FileSearchTool(
                vector_store_ids=[vector_store_id],
                max_num_results=5,
            )
        )

    instructions = """
You are a warm, encouraging, practical Life Coach Agent.

Your job:
- Help the user with goals, habits, mindset, consistency, motivation, and progress reviews.
- If the user has uploaded goals/journal data, search it first before answering questions about personal goals, progress, routines, or history.
- Use web search when outside tips, techniques, research, or practical advice would improve the answer.
- Use image generation when the user asks for a vision board, motivational poster, celebration image, milestone graphic, progress visual, or any inspirational image.
- You may combine tools naturally in one turn.

Behavior rules:
1. Start with encouragement.
2. If file search is relevant, summarize what the user's goals or journal say.
3. If web search is relevant, combine those best practices with the user's personal context.
4. If image generation is relevant, create a vivid image prompt that reflects the user's goals and achievements.
5. When generating an image, also provide a short supportive caption.
6. For progress questions, compare recent patterns against earlier entries when possible.
7. Be concise, useful, and personal.
8. If no goals document has been uploaded yet, say so honestly and continue with general coaching.
"""

    return Agent(
        name="Life Coach Agent",
        model="gpt-5.4",
        instructions=instructions,
        tools=tools,
    )


# ---------------------------------
# Extract generated images from runner result
# ---------------------------------

def extract_generated_images(run_result) -> List[Dict[str, Any]]:
    extracted = []

    raw_responses = getattr(run_result, "raw_responses", None) or []
    for response in raw_responses:
        outputs = getattr(response, "output", None) or []
        for item in outputs:
            item_type = getattr(item, "type", None)
            if item_type == "image_generation_call":
                image_base64 = getattr(item, "result", None)
                revised_prompt = getattr(item, "revised_prompt", None)
                if image_base64:
                    extracted.append(
                        {
                            "image_base64": image_base64,
                            "revised_prompt": revised_prompt,
                        }
                    )
    return extracted


# ---------------------------------
# UI
# ---------------------------------
init_state()
st.set_page_config(page_title="Life Coach Agent - Full Mission", page_icon="🎯", layout="wide")
st.title("🎯 Life Coach Agent - Full Mission")
st.caption("Web Search + File Search + Image Generation")

agent_session = SQLiteSession(st.session_state.session_id)

with st.sidebar:
    st.header("1) 목표 문서 업로드")
    uploaded_goal = st.file_uploader(
        "PDF 또는 TXT 파일 업로드",
        type=["pdf", "txt"],
        accept_multiple_files=False,
    )

    if uploaded_goal is not None:
        st.session_state.goal_file_bytes = uploaded_goal.getvalue()
        st.session_state.goal_filename = uploaded_goal.name
        st.info(f"선택된 파일: {uploaded_goal.name}")

    if st.button("목표 문서 + 일기 인덱싱"):
        if st.session_state.goal_file_bytes and st.session_state.goal_filename:
            with st.spinner("파일 검색용 인덱스를 만드는 중입니다..."):
                try:
                    st.session_state.vector_store_id = rebuild_vector_store(
                        st.session_state.goal_file_bytes,
                        st.session_state.goal_filename,
                        st.session_state.journal_entries,
                    )
                    st.success("이제 목표 문서와 일기를 검색해서 답변할 수 있습니다.")
                except Exception as e:
                    st.error(f"인덱싱 실패: {e}")
        else:
            st.warning("먼저 목표 문서를 업로드해 주세요.")

    st.divider()
    st.header("2) 일기 / 진행 기록")
    area = st.selectbox("분야", ["운동", "독서", "공부", "업무", "수면", "식단", "기타"])
    score = st.slider("오늘의 진행 점수", 0, 100, 70, 5)
    content = st.text_area(
        "기록 내용",
        placeholder="예: 이번 주 책 2권 읽었고, 아침 독서를 4일 유지했습니다.",
        height=120,
    )

    if st.button("일기 저장"):
        if content.strip():
            st.session_state.journal_entries.append(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "area": area,
                    "score": score,
                    "content": content.strip(),
                }
            )
            st.success("일기가 저장되었습니다.")
        else:
            st.warning("일기 내용을 입력해 주세요.")

    if st.button("최신 일기까지 재인덱싱"):
        if st.session_state.goal_file_bytes and st.session_state.goal_filename:
            with st.spinner("최신 기록을 반영하는 중입니다..."):
                try:
                    st.session_state.vector_store_id = rebuild_vector_store(
                        st.session_state.goal_file_bytes,
                        st.session_state.goal_filename,
                        st.session_state.journal_entries,
                    )
                    st.success("최신 일기까지 반영되었습니다.")
                except Exception as e:
                    st.error(f"재인덱싱 실패: {e}")
        else:
            st.warning("먼저 목표 문서를 업로드하세요.")

    st.divider()
    st.header("예시 질문")
    st.markdown(
        """
- 내 운동 목표 달성은 잘 되어가고 있어?
- 내 목표를 반영한 2025 비전 보드를 만들어 줘
- 올해 책 10권 읽기 목표를 달성했어. 축하 포스터를 만들어 줘
- 내 최근 기록을 보고 동기부여 메시지와 이미지를 만들어 줘
- 내 목표와 최신 팁을 바탕으로 이번 주 계획을 짜줘
        """
    )

left_col, right_col = st.columns([1.3, 1.0])

with left_col:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("예: 2025년 목표로 비전 보드를 만들어 줘")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        with st.chat_message("assistant"):
            with st.spinner("코치가 목표, 기록, 웹 정보, 이미지 생성을 함께 준비하고 있어요..."):
                try:
                    agent = build_agent(st.session_state.vector_store_id)
                    result = Runner.run_sync(agent, user_prompt, session=agent_session)
                    reply = result.final_output
                    images = extract_generated_images(result)

                    st.markdown(reply)

                    if images:
                        st.markdown("### 생성된 이미지")
                        for idx, img in enumerate(images, start=1):
                            image_bytes = base64.b64decode(img["image_base64"])
                            pil_image = Image.open(io.BytesIO(image_bytes))
                            st.image(pil_image, caption=f"Generated Image {idx}", use_container_width=True)
                            st.download_button(
                                label=f"이미지 {idx} 다운로드",
                                data=image_bytes,
                                file_name=f"life_coach_image_{idx}.png",
                                mime="image/png",
                            )
                            st.session_state.generated_images.append(img)
                    else:
                        st.info("이번 답변에서는 이미지가 생성되지 않았습니다.")

                except Exception as e:
                    reply = (
                        "문제가 발생했습니다. 아래를 확인해 주세요.\n\n"
                        "- OPENAI_API_KEY 설정 여부\n"
                        "- openai, openai-agents, pillow, streamlit 설치 여부\n"
                        "- 목표 문서를 인덱싱했는지 여부\n"
                        f"- 에러: `{e}`"
                    )
                    st.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})

with right_col:
    st.subheader("최근 진행 기록")
    if st.session_state.journal_entries:
        for entry in reversed(st.session_state.journal_entries[-5:]):
            st.markdown(
                f"**{entry['date']} | {entry['area']} | 점수 {entry['score']}**\n\n{entry['content']}"
            )
    else:
        st.write("아직 저장된 기록이 없습니다.")

    st.divider()
    st.subheader("과제 시연 체크리스트")
    st.markdown(
        """
1. 목표 문서 업로드 및 인덱싱
2. 파일 검색 질문 시연
3. 웹 검색 질문 시연
4. 비전 보드 또는 축하 포스터 생성 시연
5. 생성된 이미지 다운로드 화면 보여주기

        """
    )

