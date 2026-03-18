import streamlit as st
from datetime import datetime

# Simple chatbot rules (beginner-friendly and easy to extend)
def generate_response(user_text: str) -> str:
    text = user_text.lower().strip()

    if not text:
        return "Say something and I can respond."

    if any(greet in text for greet in ["hi", "hello", "hey"]):
        return "Hi! I'm your chatbot. Ask me a question or type 'help' for tips."

    if "help" in text:
        return (
            "Try asking things like:\n"
            "- What can you do?\n"
            "- What time is it?\n"
            "- Give me a study tip"
        )

    if "time" in text:
        now = datetime.now().strftime("%I:%M %p")
        return f"It is currently {now}."

    if "study" in text or "tip" in text:
        return (
            "Study tip: break the problem into small steps and test each step. "
            "Small wins add up!"
        )

    if "what can you do" in text or "capabilities" in text:
        return "I can answer simple questions, give tips, and chat with you."

    # Fallback response
    return "I'm not sure yet, but I'm learning. Try asking in a different way."


st.set_page_config(page_title="Chatbot")

st.title("Chatbot")
st.write("A simple Streamlit chatbot. Ask me anything!")

# Store chat history in session state so it persists across reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm your chatbot. How can I help?"}
    ]

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input box
user_prompt = st.chat_input("Type your message")

if user_prompt:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    # Generate and add assistant response
    bot_reply = generate_response(user_prompt)
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    # Rerun to display the new messages
    st.rerun()
