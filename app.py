import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import requests
import streamlit as st

st.set_page_config(page_title="My AI Chat", layout="wide")

APP_TITLE = "My AI Chat"
DEFAULT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
CHATS_DIR = Path("chats")
MEMORY_FILE = Path("memory.json")


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


def load_memory() -> dict:
    if not MEMORY_FILE.exists():
        return {}
    try:
        return json.loads(MEMORY_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_memory(memory: dict) -> None:
    MEMORY_FILE.write_text(json.dumps(memory, indent=2))


def merge_memory(existing: dict, updates: dict) -> dict:
    merged = existing.copy()
    for key, value in updates.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if value == {} or value == []:
            continue
        merged[key] = value
    return merged


def memory_to_system_prompt(memory: dict) -> str:
    if not memory:
        return ""
    lines = ["You are a helpful assistant. Use these user preferences to personalize responses:"]
    for key, value in memory.items():
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def parse_json_from_text(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # remove optional language label like json
        cleaned = cleaned.replace("json", "", 1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            return {}
    return {}


def pick_most_recent_chat(chats: Dict[str, dict]) -> str:
    most_recent = max(
        chats.values(),
        key=lambda c: c.get("created_at", ""),
    )
    return most_recent["id"]


def stream_hf_api(messages: List[dict], token: str, model: str):
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "stream": True,
    }

    try:
        response = requests.post(
            url, headers=headers, json=payload, timeout=30, stream=True
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Network error: {exc}") from exc

    if response.status_code != 200:
        try:
            error_payload = response.json()
            error_message = error_payload.get("error", "Unknown error")
        except Exception:
            error_message = response.text
        raise RuntimeError(f"API error {response.status_code}: {error_message}")

    for raw_line in response.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        if not raw_line.startswith("data:"):
            continue
        data_str = raw_line.replace("data:", "", 1).strip()
        if data_str == "[DONE]":
            break
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(f"API error: {payload['error']}")
        choices = payload.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content", "")
        if content:
            yield content


def call_hf_api(
    messages: List[dict], token: str, model: str, max_tokens: int, temperature: float
) -> dict:
    url = "https://router.huggingface.co/v1/chat/completions"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
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

    return data


st.title(APP_TITLE)

# Load token safely
hf_token = st.secrets.get("HF_TOKEN", "").strip()
if not hf_token:
    st.error("Missing Hugging Face token. Please add HF_TOKEN to .streamlit/secrets.toml.")
    st.stop()

hf_model = st.secrets.get("HF_MODEL", DEFAULT_MODEL)

# Load memory once
if "user_memory" not in st.session_state:
    st.session_state.user_memory = load_memory()

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

with st.sidebar.expander("User Memory", expanded=False):
    st.json(st.session_state.user_memory)
    if st.session_state.get("last_memory_raw"):
        st.caption("Last extraction (raw):")
        st.code(st.session_state.last_memory_raw)
    if st.button("Clear Memory"):
        st.session_state.user_memory = {}
        st.session_state.last_memory_raw = ""
        save_memory(st.session_state.user_memory)
        st.rerun()

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
            if st.button("Delete", key=f"delete_{chat['id']}"):
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
def extract_memory_from_user_message(user_text: str) -> dict:
    prompt = (
        "Extract any personal facts or preferences from the user message as JSON. "
        "Use only these keys when relevant: name, language, interests, communication_style, "
        "favorite_topics, location. If none, return {}. Only return valid JSON.\n\n"
        f"User message: {user_text}"
    )
    extraction_messages = [
        {"role": "system", "content": "You are a JSON extraction assistant."},
        {"role": "user", "content": prompt},
    ]
    data = call_hf_api(extraction_messages, hf_token, hf_model, 128, 0.0)
    try:
        content = data["choices"][0]["message"]["content"]
        st.session_state.last_memory_raw = content
        return parse_json_from_text(content)
    except Exception:
        return {}


def build_messages_with_memory(messages: List[dict], memory: dict) -> List[dict]:
    system_prompt = memory_to_system_prompt(memory)
    if system_prompt:
        return [{"role": "system", "content": system_prompt}] + messages
    return messages


if not active_chat["messages"]:
    test_message = {"role": "user", "content": "Hello!"}
    active_chat["messages"].append(test_message)
    assistant_text = ""
    try:
        messages_with_memory = build_messages_with_memory(
            active_chat["messages"], st.session_state.user_memory
        )
        for chunk in stream_hf_api(messages_with_memory, hf_token, hf_model):
            assistant_text += chunk
    except RuntimeError as err:
        st.error(str(err))
        assistant_text = "Sorry, I ran into a problem reaching the API. Please try again."
    active_chat["messages"].append({"role": "assistant", "content": assistant_text})
    try:
        updates = extract_memory_from_user_message(test_message["content"])
        st.session_state.user_memory = merge_memory(
            st.session_state.user_memory, updates
        )
        save_memory(st.session_state.user_memory)
    except RuntimeError:
        pass
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

    # Call API and add assistant response (streamed)
    try:
        messages_with_memory = build_messages_with_memory(
            active_chat["messages"], st.session_state.user_memory
        )
        assistant_text = ""
        with st.chat_message("assistant"):
            placeholder = st.empty()
            for chunk in stream_hf_api(messages_with_memory, hf_token, hf_model):
                assistant_text += chunk
                placeholder.write(assistant_text)
                time.sleep(0.02)
    except RuntimeError as err:
        st.error(str(err))
        assistant_text = "Sorry, I ran into a problem reaching the API. Please try again."

    active_chat["messages"].append({"role": "assistant", "content": assistant_text})

    # Extract memory from the latest user message
    try:
        updates = extract_memory_from_user_message(user_prompt)
        st.session_state.user_memory = merge_memory(
            st.session_state.user_memory, updates
        )
        save_memory(st.session_state.user_memory)
    except RuntimeError:
        pass

    save_chat_to_disk(active_chat)
    st.rerun()
