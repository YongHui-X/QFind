"""Streamlit demo for ClauseLens contract clause retrieval."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import streamlit as st
import streamlit.components.v1 as components

from app.chat_history import (
    delete_chat,
    list_chats,
    load_chat,
    save_chat,
)
from app.telemetry import (
    append_feedback,
    append_query_metric,
    load_query_metrics,
    summarize_query_metrics,
)

API_BASE_URL = "http://127.0.0.1:8000"
EVAL_RESULTS_PATH = Path("data/processed/eval_results.json")
DEFAULT_GREETING = "Ask a contract question and I'll answer using retrieved evidence."
DEFAULT_CLAUSE_TYPE = "All clause types"
DEFAULT_RESULT_LIMIT = 5
DEFAULT_RERANK_MODE = "auto"
STREAM_RENDER_INTERVAL_SECONDS = 0.03
HTTP_CLIENT = httpx.Client(
    timeout=httpx.Timeout(300.0, connect=10.0),
    limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
)

SIDEBAR_STYLE = """
<style>
    section[data-testid="stSidebar"] {
        width: 280px !important;
        min-width: 280px !important;
    }
    section[data-testid="stSidebar"] > div {
        width: 280px !important;
        min-width: 280px !important;
    }
    section[data-testid="stSidebar"] * {
        font-size: 0.82rem !important;
        line-height: 1.15 !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p {
        margin-bottom: 0.25rem !important;
    }
    section[data-testid="stSidebar"] button {
        padding-top: 0.25rem !important;
        padding-bottom: 0.25rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMetric"] {
        padding: 0.15rem 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        font-size: 0.72rem !important;
        line-height: 1.05 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
        font-size: 1.0rem !important;
        line-height: 1.1 !important;
    }
</style>
"""

THINKING_INDICATOR = """
<style>
    .clauselens-thinking {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        color: #ffffff;
        font-weight: 600;
    }
    .clauselens-thinking-dots {
        display: inline-flex;
        align-items: center;
        gap: 0.2rem;
    }
    .clauselens-thinking-dot {
        width: 0.38rem;
        height: 0.38rem;
        border-radius: 50%;
        background: #ffffff;
        animation: clauselens-thinking-bounce 1.1s infinite ease-in-out;
    }
    .clauselens-thinking-dot:nth-child(2) { animation-delay: 0.16s; }
    .clauselens-thinking-dot:nth-child(3) { animation-delay: 0.32s; }
    @keyframes clauselens-thinking-bounce {
        0%, 60%, 100% { opacity: 0.3; transform: translateY(0); }
        30% { opacity: 1; transform: translateY(-0.28rem); }
    }
</style>
<div class="clauselens-thinking">
    <span>__LABEL__</span>
    <span class="clauselens-thinking-dots" aria-label="Thinking">
        <span class="clauselens-thinking-dot"></span>
        <span class="clauselens-thinking-dot"></span>
        <span class="clauselens-thinking-dot"></span>
    </span>
</div>
"""

STATUS_LABELS = {
    "routing": "Understanding the question",
    "contextualizing": "Resolving the follow-up",
    "retrieved": "Evidence found",
    "reranked": "Evidence reranked",
    "generating": "Writing the answer",
}


def thinking_indicator(label: str) -> str:
    return THINKING_INDICATOR.replace("__LABEL__", label)


def api_get_json(api_base_url: str, path: str) -> dict[str, object]:
    response = HTTP_CLIENT.get(f"{api_base_url.rstrip('/')}{path}", timeout=30)
    response.raise_for_status()
    return dict(response.json())


def api_post_stream_json(
    api_base_url: str,
    path: str,
    payload: dict[str, object],
):
    with HTTP_CLIENT.stream(
        "POST",
        f"{api_base_url.rstrip('/')}{path}",
        json=payload,
    ) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            line = raw_line.strip()
            if line:
                yield json.loads(line)


def load_saved_eval(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def get_chat_messages() -> list[dict[str, Any]]:
    messages = st.session_state.setdefault("messages", [])
    if not messages:
        messages.append({"role": "assistant", "content": DEFAULT_GREETING})
    return messages


def new_chat() -> None:
    """Switch to a fresh unsaved conversation."""

    st.session_state.active_chat_id = None
    st.session_state.messages = [{"role": "assistant", "content": DEFAULT_GREETING}]
    st.session_state.feedback_submitted = {}
    st.session_state.clause_type_control = DEFAULT_CLAUSE_TYPE
    st.session_state.limit_control = DEFAULT_RESULT_LIMIT
    st.session_state.rerank_mode_control = DEFAULT_RERANK_MODE


def open_saved_chat(chat_id: str) -> None:
    """Load one persisted conversation into the active Streamlit session."""

    saved = load_chat(chat_id)
    if saved is None:
        new_chat()
        return
    st.session_state.active_chat_id = saved.chat_id
    st.session_state.messages = saved.messages
    st.session_state.feedback_submitted = {}
    st.session_state.clause_type_control = (
        saved.clause_type or DEFAULT_CLAUSE_TYPE
    )
    st.session_state.limit_control = saved.limit
    st.session_state.rerank_mode_control = saved.rerank_mode


def remove_saved_chat(chat_id: str) -> None:
    """Delete a saved conversation and clear it if currently active."""

    delete_chat(chat_id)
    if st.session_state.get("active_chat_id") == chat_id:
        new_chat()


def initialize_session_controls() -> None:
    st.session_state.setdefault("active_chat_id", None)
    st.session_state.setdefault("clause_type_control", DEFAULT_CLAUSE_TYPE)
    st.session_state.setdefault("limit_control", DEFAULT_RESULT_LIMIT)
    st.session_state.setdefault("rerank_mode_control", DEFAULT_RERANK_MODE)


def render_chat_history() -> None:
    """Render new-chat, open-chat, and delete controls."""

    st.button("＋ New chat", use_container_width=True, on_click=new_chat)
    chats = list_chats()
    if not chats:
        st.caption("Completed chats will appear here automatically.")
        return

    st.caption("Recent chats")
    active_chat_id = st.session_state.get("active_chat_id")
    for chat in chats:
        open_column, delete_column = st.columns([0.84, 0.16], gap="small")
        label = f"● {chat.title}" if chat.chat_id == active_chat_id else chat.title
        if open_column.button(
            label,
            key=f"open-chat-{chat.chat_id}",
            use_container_width=True,
            help=chat.title,
        ):
            open_saved_chat(chat.chat_id)
            st.rerun()
        if delete_column.button(
            "×",
            key=f"delete-chat-{chat.chat_id}",
            help=f"Delete {chat.title}",
        ):
            remove_saved_chat(chat.chat_id)
            st.rerun()


def submit_feedback(turn_id: str, rating: str) -> None:
    submitted = st.session_state.setdefault("feedback_submitted", {})
    if submitted.get(turn_id) == rating:
        return
    append_feedback(turn_id, rating)
    submitted[turn_id] = rating


def render_feedback(turn_id: str) -> None:
    submitted = st.session_state.setdefault("feedback_submitted", {})
    current = submitted.get(turn_id)
    if current:
        label = "helpful" if current == "up" else "not helpful"
        st.caption(f"Feedback recorded: {label}")
        return

    left, right, _ = st.columns([0.16, 0.2, 0.64])
    if left.button("Helpful", key=f"up-{turn_id}"):
        submit_feedback(turn_id, "up")
        st.rerun()
    if right.button("Not helpful", key=f"down-{turn_id}"):
        submit_feedback(turn_id, "down")
        st.rerun()


def render_evidence(results: list[dict[str, object]]) -> None:
    if not results:
        return

    with st.expander(f"Show retrieved evidence ({len(results)})", expanded=False):
        st.caption(
            "Scores are ranking signals, not percentages. "
            "Cross-encoder scores are unbounded."
        )
        for index, item in enumerate(results, start=1):
            reranker_score = item.get("reranker_score")
            vector_score = item.get("vector_score")
            if reranker_score is not None:
                score_label = f"reranker {float(reranker_score):.3f}"
            elif vector_score is not None:
                score_label = f"similarity {float(vector_score):.3f}"
            else:
                score_label = f"ranking {float(item['score']):.3f}"

            st.markdown(
                f"**{index}. {item['clause_type'] or 'Unknown clause'}** "
                f"`{score_label}`"
            )
            if item.get("answer"):
                st.caption(f"CUAD answer label: {item['answer']}")
            st.write(item["text"])
            st.caption(
                f"Source: {item['source_pdf'] or 'Unknown'} | "
                f"Document: {item['document_id'] or 'Unknown'}"
            )
            if index != len(results):
                st.divider()


def render_latency_summary(timings: dict[str, object]) -> None:
    total = float(timings.get("total_latency_ms", 0.0) or 0.0)
    if total <= 0.0:
        return

    first_token = float(timings.get("first_token_latency_ms", 0.0) or 0.0)
    contextualization = float(
        timings.get(
            "contextualization_latency_ms",
            timings.get("rewrite_latency_ms", 0.0),
        )
        or 0.0
    )
    retrieval = float(timings.get("retrieval_latency_ms", 0.0) or 0.0)
    embedding = float(timings.get("embedding_latency_ms", 0.0) or 0.0)
    vector_search = float(timings.get("vector_search_latency_ms", 0.0) or 0.0)
    lexical_search = float(timings.get("lexical_search_latency_ms", 0.0) or 0.0)
    reranker_loading = float(
        timings.get("reranker_loading_latency_ms", 0.0) or 0.0
    )
    reranking = float(timings.get("reranking_latency_ms", 0.0) or 0.0)
    answer = float(timings.get("answer_latency_ms", 0.0) or 0.0)
    st.caption(
        f"Turn {total / 1000:.2f}s | first token {first_token / 1000:.2f}s | "
        f"context {contextualization / 1000:.2f}s | "
        f"retrieval {retrieval / 1000:.2f}s "
        f"(embed {embedding / 1000:.2f}s, vector {vector_search / 1000:.2f}s, "
        f"lexical {lexical_search / 1000:.2f}s) | "
        f"reranker load {reranker_loading / 1000:.2f}s | "
        f"rerank {reranking / 1000:.2f}s | answer {answer / 1000:.2f}s"
    )


def render_assistant_details(message: dict[str, Any]) -> None:
    response = message.get("response")
    if not isinstance(response, dict):
        return

    render_latency_summary(dict(response.get("timings", {})))
    generation = dict(response.get("generation", {}))
    if generation.get("prompt_chars"):
        st.caption(
            "Generation prompt: "
            f"{int(generation.get('prompt_chars', 0)):,} chars "
            f"(~{int(generation.get('estimated_input_tokens', 0)):,} tokens), "
            f"{int(generation.get('evidence_chars', 0)):,} evidence chars"
        )
    if generation.get("model"):
        tier = (
            generation.get("response_service_tier")
            or generation.get("requested_service_tier")
            or "standard"
        )
        st.caption(
            f"Model: {generation['model']} | service tier: {tier} | "
            f"output ~{int(generation.get('estimated_output_tokens', 0))} tokens"
        )
    render_evidence(list(response.get("results", [])))
    st.caption(f"Standalone query: {response.get('standalone_query', '') or 'n/a'}")
    resolved_clause_type = response.get("resolved_clause_type")
    if resolved_clause_type:
        st.caption(f"Resolved clause type: {resolved_clause_type}")
    st.caption(
        "Reranking: "
        f"{'applied' if response.get('reranking_applied') else 'not applied'} "
        f"({response.get('rerank_reason', 'n/a')})"
    )
    turn_id = str(response.get("turn_id", ""))
    if turn_id:
        render_feedback(turn_id)


def render_chat_messages(messages: list[dict[str, Any]]) -> None:
    for message in messages:
        with st.chat_message(str(message["role"])):
            st.markdown(str(message["content"]))
            if message["role"] == "assistant":
                render_assistant_details(message)


def scroll_latest_message_to_top() -> None:
    components.html(
        """
        <script>
            const messages = window.parent.document.querySelectorAll(
                '[data-testid="stChatMessage"]'
            );
            const latest = messages[messages.length - 1];
            if (latest) {
                setTimeout(
                    () => latest.scrollIntoView({behavior: "smooth", block: "start"}),
                    100
                );
            }
        </script>
        """,
        height=0,
    )


def render_eval_panel() -> None:
    st.subheader("Offline retrieval accuracy")
    rows = load_saved_eval(EVAL_RESULTS_PATH)
    if not rows:
        st.caption("No saved evaluation report. Run `evaluation\\eval.py`.")
    else:
        total = len(rows)
        passed = sum(1 for row in rows if row.get("passed"))
        avg_mrr = sum(float(row.get("clause_type_mrr", 0)) for row in rows) / total
        avg_ndcg = sum(float(row.get("ndcg", 0)) for row in rows) / total
        st.metric("Passed", f"{passed}/{total}")
        st.metric("Avg MRR", f"{avg_mrr:.3f}")
        st.metric("Avg nDCG", f"{avg_ndcg:.3f}")

    st.subheader("Live query metrics")
    summary = summarize_query_metrics(load_query_metrics())
    if not summary["queries"]:
        st.caption("Live metrics update after each completed query.")
        return
    st.metric("Queries logged", int(summary["queries"]))
    st.metric("Automated checks", f"{summary['live_check_pass_rate']:.0%}")
    st.metric(
        "p95 first text",
        f"{float(summary['p95_first_visible_ms']) / 1000:.1f}s",
    )
    st.metric("p95 turn", f"{float(summary['p95_latency_ms']) / 1000:.1f}s")
    if summary["feedback_count"]:
        st.metric(
            "Helpful feedback",
            f"{summary['positive_feedback_rate']:.0%} "
            f"({int(summary['feedback_count'])} rated)",
        )


def main() -> None:
    st.set_page_config(
        page_title="ClauseLens",
        page_icon="CL",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(SIDEBAR_STYLE, unsafe_allow_html=True)
    initialize_session_controls()
    st.title("ClauseLens")
    st.caption("Grounded contract clause chatbot over CUAD with citations.")

    with st.sidebar:
        st.header("Chats")
        render_chat_history()
        st.divider()
        st.subheader("Chat controls")
        api_base_url = API_BASE_URL
        try:
            clause_type_response = api_get_json(api_base_url, "/clause-types")
            clause_types = [
                str(item) for item in clause_type_response.get("clause_types", [])
            ]
        except (httpx.HTTPError, TimeoutError, ValueError) as exc:
            st.error(f"API is not reachable: {exc}")
            st.stop()

        selected_clause_type = st.selectbox(
            "Clause type",
            [DEFAULT_CLAUSE_TYPE, *clause_types],
            key="clause_type_control",
        )
        limit = st.slider(
            "Top results",
            min_value=1,
            max_value=20,
            key="limit_control",
        )
        rerank_mode = st.selectbox(
            "Reranking",
            options=["auto", "off", "always"],
            key="rerank_mode_control",
            format_func={
                "auto": "Auto (recommended)",
                "off": "Off (fastest)",
                "always": "Always (slowest)",
            }.get,
        )
        with st.expander("Evaluation metrics", expanded=False):
            render_eval_panel()

    clause_type = (
        None if selected_clause_type == DEFAULT_CLAUSE_TYPE else selected_clause_type
    )
    messages = get_chat_messages()
    render_chat_messages(messages)

    user_prompt = st.chat_input("Ask a contract review question")
    if not user_prompt:
        return
    if not user_prompt.strip():
        st.error("Enter a non-empty question.")
        return

    request_started = time.perf_counter()
    messages.append({"role": "user", "content": user_prompt.strip()})
    render_chat_messages(messages[-1:])

    final_response: dict[str, object] | None = None
    streamed_answer = ""
    first_visible_ms = 0.0
    last_rendered_at = 0.0

    try:
        with st.chat_message("assistant"):
            assistant_placeholder = st.empty()
            initial_stage = (
                "contextualizing"
                if sum(message["role"] == "user" for message in messages) > 1
                else "routing"
            )
            assistant_placeholder.markdown(
                thinking_indicator(STATUS_LABELS.get(initial_stage, "Thinking")),
                unsafe_allow_html=True,
            )
            for event in api_post_stream_json(
                api_base_url,
                "/chat/stream",
                {
                    "messages": [
                        {"role": message["role"], "content": message["content"]}
                        for message in messages
                    ],
                    "clause_type": clause_type,
                    "limit": limit,
                    "rerank_mode": rerank_mode,
                },
            ):
                event_type = str(event.get("event", ""))
                if event_type == "status":
                    stage = str(event.get("stage", ""))
                    assistant_placeholder.markdown(
                        thinking_indicator(STATUS_LABELS.get(stage, "Thinking")),
                        unsafe_allow_html=True,
                    )
                elif event_type == "token":
                    if first_visible_ms == 0.0:
                        first_visible_ms = (
                            time.perf_counter() - request_started
                        ) * 1000
                    streamed_answer += str(event.get("delta", ""))
                    now = time.perf_counter()
                    if now - last_rendered_at >= STREAM_RENDER_INTERVAL_SECONDS:
                        assistant_placeholder.markdown(streamed_answer + " |")
                        last_rendered_at = now
                elif event_type == "final":
                    final_response = dict(event.get("data", {}))
                elif event_type == "error":
                    raise ValueError(str(event.get("detail", "Chat generation failed")))

            if final_response is None:
                raise ValueError("no final response was returned by the API")

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": str(final_response.get("answer", "")),
                "response": final_response,
            }
            assistant_placeholder.markdown(assistant_message["content"])
            append_query_metric(
                final_response,
                client_total_latency_ms=(
                    time.perf_counter() - request_started
                ) * 1000,
                client_first_visible_ms=first_visible_ms,
            )
            render_assistant_details(assistant_message)
            scroll_latest_message_to_top()
    except (httpx.HTTPError, TimeoutError, ValueError) as exc:
        st.error(
            "Chat failed. Confirm the API is ready, the OpenAI key is configured, "
            f"and the data is indexed. Details: {exc}"
        )
        messages.pop()
        st.session_state.messages = messages
        return

    messages.append(assistant_message)
    st.session_state.messages = messages
    st.session_state.active_chat_id = save_chat(
        chat_id=st.session_state.get("active_chat_id"),
        messages=messages,
        clause_type=clause_type,
        limit=limit,
        rerank_mode=rerank_mode,
    )
    st.rerun()


if __name__ == "__main__":
    main()
