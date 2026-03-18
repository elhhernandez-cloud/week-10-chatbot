import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
import streamlit as st

st.set_page_config(page_title="My AI Chat", layout="wide")

APP_TITLE = "My AI Chat"
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
CHATS_DIR = Path("chats")


def ensure_chats_dir() -> None:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)


def chat_file_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def load_chats_from_disk() -> Dict[str, dict]:
    ensure_chats_dir()
    chats: Dict[str, dict] = {}
    for path in CHATS_DIR.glob("*.json"):
        try:
            data = json.loads(path.read_text())
            if "id" in data and "messages" in data:
                chats[data["id"]] = data
        except (json.JSONDecodeError, OSError):
            # Skip unreadable or corrupted chat files
            continue
    return chats


def save_chat_to_disk(chat: dict) -> None:
    ensure_chats_dir()
    path = chat_file_path(chat["id"])
    path.write_text(json.dumps(chat, indent=2))


def delete_chat_from_disk(chat_id: str) -> None:
    path = chat_file_path(chat_id)
    if path.exists():
        path.unlink()


def new_chat() -> dict:
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    chat = {
        "id": chat_id,
        "title": "New chat",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "messages": [],
    }
    save_chat_to_disk(chat)
    return chat


def pick_most_recent_chat(chats: Dict[str, dict]) -> str:
    most_recent = max(
        chats.values(),
        key=lambda c: c.get("created_at", ""),
    )
    return most_recent["id"]


def build_prompt(messages: List[dict]) -> str:
    lines = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    lines.append("Assistant:")
    return "\n".join(lines)


def call_hf_api(messages: List[dict], token: str, model: str) -> str:
    url = f"https://router.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": build_prompt(messages),
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    if response.status_code != 200:
        try:
            error_payload = response.json()
            error_message = error_payload.get("error", "Unknown error")
        except Exception:
            error_message = response.text
        raise RuntimeError(f"API error {response.status_code}: {error_message}")

    data = response.json()
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"API error: {data['error']}")

    if isinstance(data, list) and data:
        generated = data[0].get("generated_text", "")
        return generated.strip()

    raise RuntimeError("Unexpected API response format.")


st.title(APP_TITLE)

# Load token safely
hf_token = st.secrets.get("HF_TOKEN", "").strip()
if not hf_token:
    st.error("Missing Hugging Face token. Please add HF_TOKEN to .streamlit/secrets.toml.")
    st.stop()

hf_model = st.secrets.get("HF_MODEL", DEFAULT_MODEL)

# Load chats into session state once
if "chats" not in st.session_state:
    st.session_state.chats = load_chats_from_disk()

if "active_chat_id" not in st.session_state:
    if st.session_state.chats:
        st.session_state.active_chat_id = pick_most_recent_chat(st.session_state.chats)
    else:
        new = new_chat()
        st.session_state.chats[new["id"]] = new
        st.session_state.active_chat_id = new["id"]

# Sidebar: new chat + chat list
st.sidebar.title("Chats")
if st.sidebar.button("New Chat"):
    new = new_chat()
    st.session_state.chats[new["id"]] = new
    st.session_state.active_chat_id = new["id"]
    st.rerun()

chat_list_container = st.sidebar.container(height=400)

# Sort chats by created_at (newest first)
chat_items = sorted(
    st.session_state.chats.values(),
    key=lambda c: c.get("created_at", ""),
    reverse=True,
)

for chat in chat_items:
    is_active = chat["id"] == st.session_state.active_chat_id
    title = chat.get("title", "Chat")
    timestamp = chat.get("created_at", "")
    label = f"{'>>' if is_active else '  '} {title}"

    with chat_list_container:
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            if st.button(label, key=f"open_{chat['id']}"):
                st.session_state.active_chat_id = chat["id"]
                st.rerun()
            st.caption(timestamp)
        with cols[1]:
            if st.button("x", key=f"delete_{chat['id']}"):
                delete_chat_from_disk(chat["id"])
                del st.session_state.chats[chat["id"]]

                # Pick a new active chat or create one
                if st.session_state.chats:
                    st.session_state.active_chat_id = pick_most_recent_chat(
                        st.session_state.chats
                    )
                else:
                    new = new_chat()
                    st.session_state.chats[new["id"]] = new
                    st.session_state.active_chat_id = new["id"]
                st.rerun()

# Main chat area
active_chat = st.session_state.chats.get(st.session_state.active_chat_id)

if not active_chat:
    st.info("No active chat. Create a new chat from the sidebar.")
    st.stop()

# Part A test: if the chat is empty, send a single hardcoded test message once
if not active_chat["messages"]:
    test_message = {"role": "user", "content": "Hello!"}
    active_chat["messages"].append(test_message)
    try:
        reply = call_hf_api(active_chat["messages"], hf_token, hf_model)
    except RuntimeError as err:
        st.error(str(err))
        reply = "Sorry, I ran into a problem reaching the API. Please try again."
    active_chat["messages"].append({"role": "assistant", "content": reply})
    save_chat_to_disk(active_chat)

messages_container = st.container(height=500)
with messages_container:
    for message in active_chat["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

user_prompt = st.chat_input("Type your message")

if user_prompt:
    # Add user message
    active_chat["messages"].append({"role": "user", "content": user_prompt})

    # Create a title when the first user message arrives
    if active_chat["title"] == "New chat":
        active_chat["title"] = user_prompt[:30] + ("..." if len(user_prompt) > 30 else "")

    # Call API and add assistant response
    try:
        reply = call_hf_api(active_chat["messages"], hf_token, hf_model)
    except RuntimeError as err:
        st.error(str(err))
        reply = "Sorry, I ran into a problem reaching the API. Please try again."

    active_chat["messages"].append({"role": "assistant", "content": reply})

    save_chat_to_disk(active_chat)
    st.rerun()
