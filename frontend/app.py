import os
from uuid import uuid4

import httpx
import streamlit as st


REQUEST_TIMEOUT_SECONDS = 90.0
RESET_TIMEOUT_SECONDS = 8.0

st.set_page_config(
    page_title="Air Bank Information Assistant",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded",
)


def _backend_url() -> str:
    """Resolve the backend URL locally or from Streamlit Community Cloud secrets."""

    value = os.getenv("BACKEND_URL")
    if not value:
        try:
            value = st.secrets.get("BACKEND_URL")
        except (FileNotFoundError, KeyError):
            value = None
    return (value or "http://localhost:8000").rstrip("/")


BACKEND_URL = _backend_url()


def _start_new_local_session() -> None:
    st.session_state.session_id = str(uuid4())
    st.session_state.messages = []


def _ensure_session() -> None:
    if "session_id" not in st.session_state:
        _start_new_local_session()


def reset_conversation() -> None:
    """Best-effort backend cleanup followed by a guaranteed fresh local session."""

    old_session_id = st.session_state.get("session_id")
    if old_session_id:
        try:
            httpx.post(
                f"{BACKEND_URL}/reset",
                json={"session_id": old_session_id},
                timeout=RESET_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError:
            pass

    _start_new_local_session()


def render_sources(sources: list[dict]) -> None:
    """Render unique consulted sources without exposing retrieval scores."""

    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"- [{source['title']}]({source['url']})")


def render_message(message: dict) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))


_ensure_session()

st.title("Air Bank Information Assistant")
st.caption("Unofficial portfolio demo using publicly available Air Bank information.")

with st.sidebar:
    st.header("What this demo does")
    st.write(
        "Ask questions about Air Bank products and services, including accounts, "
        "cards, savings, loans, mortgages, payments, fees, and digital banking."
    )
    st.write(
        "Before answering, the assistant searches a local knowledge base built from "
        "public Air Bank pages and documents. Consulted sources are shown below each answer."
    )
    st.warning(
        "This is not an official Air Bank service. It cannot access accounts or perform "
        "banking actions. Do not enter personal, login, card, or payment information."
    )
    st.button(
        "Start a new conversation",
        type="secondary",
        use_container_width=True,
        on_click=reset_conversation,
    )

if not st.session_state.messages:
    st.markdown("**Example questions**")
    st.markdown(
        "- Jak funguje bonusové úročení spořicího účtu?\n"
        "- Můžu ze spořicího účtu platit kartou?\n"
        "- Jaké jsou aktuální podmínky hypotéky?"
    )

for stored_message in st.session_state.messages:
    render_message(stored_message)

prompt = st.chat_input("Ask a question about Air Bank...")

if prompt:
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    try:
        with st.chat_message("assistant"):
            with st.spinner("Searching Air Bank sources..."):
                response = httpx.post(
                    f"{BACKEND_URL}/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                    },
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )

            if response.is_error:
                try:
                    detail = response.json().get("detail", "The request failed.")
                except ValueError:
                    detail = "The request failed."
                st.error(detail)
            else:
                data = response.json()
                answer = data["answer"]
                sources = data.get("sources", [])
                st.markdown(answer)
                render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

    except httpx.RequestError:
        with st.chat_message("assistant"):
            st.error("The assistant service is currently unavailable. Please try again shortly.")

st.divider()
st.caption("Portfolio demonstration only — not an official Air Bank service or financial advice.")
