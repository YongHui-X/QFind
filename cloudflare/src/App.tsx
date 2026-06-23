import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { streamChat } from "./api";
import { chatTitle, loadChats, persistChats } from "./storage";
import { initializeTurnstile, turnstileToken } from "./turnstile";
import type { Message, SavedChat } from "./types";
import "./styles.css";

const CLAUSE_TYPES = [
  "",
  "Anti-Assignment",
  "Cap On Liability",
  "License Grant",
  "Audit Rights",
  "Termination For Convenience",
];

const GREETING =
  "Ask a contract question and I’ll answer using retrieved evidence from the supported CUAD clauses.";

const STATUS_LABELS: Record<string, string> = {
  routing: "Identifying clause type",
  contextualizing: "Resolving follow-up",
  retrieved: "Evidence retrieved",
  reranked: "Evidence reranked",
  generating: "Drafting grounded answer",
};

function newChat(): SavedChat {
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    messages: [{ id: crypto.randomUUID(), role: "assistant", content: GREETING }],
    clauseType: "",
    limit: 5,
    updatedAt: new Date().toISOString(),
  };
}

export default function App() {
  const [chats, setChats] = useState<SavedChat[]>(() => loadChats());
  const [active, setActive] = useState<SavedChat>(() => loadChats()[0] ?? newChat());
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const turnstileRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (turnstileRef.current) {
      initializeTurnstile(
        turnstileRef.current,
        import.meta.env.VITE_TURNSTILE_SITE_KEY || "1x00000000000000000000AA",
      );
    }
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [active.messages, status]);

  const recentChats = useMemo(
    () => chats.slice().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)),
    [chats],
  );

  function updateActive(next: SavedChat, save = true) {
    setActive(next);
    if (!save) return;
    setChats((current) => {
      const updated = [next, ...current.filter((chat) => chat.id !== next.id)].slice(0, 20);
      persistChats(updated);
      return updated;
    });
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    const clean = question.trim();
    if (!clean || busy) return;
    setBusy(true);
    setError("");
    setQuestion("");
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: clean,
    };
    const pending: SavedChat = {
      ...active,
      title: active.title === "New chat" ? chatTitle(clean) : active.title,
      messages: [...active.messages, userMessage],
      updatedAt: new Date().toISOString(),
    };
    setActive(pending);
    let streamed = "";
    try {
      const token = await turnstileToken();
      await streamChat(
        pending.messages.filter(
          (message, index) => !(index === 0 && message.content === GREETING),
        ),
        pending.clauseType,
        pending.limit,
        token,
        (streamEvent) => {
          if (streamEvent.event === "status") {
            setStatus(STATUS_LABELS[streamEvent.stage] ?? "Working");
          } else if (streamEvent.event === "token") {
            streamed += streamEvent.delta;
            setActive({ ...pending, messages: [...pending.messages, {
              id: "streaming",
              role: "assistant",
              content: streamed,
            }] });
          } else if (streamEvent.event === "error") {
            throw new Error(streamEvent.detail);
          } else {
            const assistant: Message = {
              id: streamEvent.data.turn_id,
              role: "assistant",
              content: streamEvent.data.answer,
              response: streamEvent.data,
            };
            updateActive({
              ...pending,
              messages: [...pending.messages, assistant],
              updatedAt: new Date().toISOString(),
            });
          }
        },
      );
    } catch (caught) {
      setActive(active);
      setQuestion(clean);
      setError(caught instanceof Error ? caught.message : "Chat request failed");
    } finally {
      setBusy(false);
      setStatus("");
    }
  }

  function deleteChat(id: string) {
    const updated = chats.filter((chat) => chat.id !== id);
    persistChats(updated);
    setChats(updated);
    if (active.id === id) setActive(updated[0] ?? newChat());
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">CL</span>
          <div><strong>ClauseLens</strong><small>Contract evidence assistant</small></div>
        </div>
        <button className="new-chat" onClick={() => setActive(newChat())}>＋ New chat</button>
        <div className="recent-label">Recent chats</div>
        <nav className="chat-list">
          {recentChats.map((chat) => (
            <div className={`chat-row ${chat.id === active.id ? "active" : ""}`} key={chat.id}>
              <button onClick={() => setActive(chat)}>{chat.title}</button>
              <button className="delete" aria-label={`Delete ${chat.title}`} onClick={() => deleteChat(chat.id)}>×</button>
            </div>
          ))}
        </nav>
        <div className="controls">
          <label>
            Clause type
            <select
              value={active.clauseType}
              onChange={(event) => updateActive({ ...active, clauseType: event.target.value }, false)}
            >
              {CLAUSE_TYPES.map((type) => (
                <option key={type || "all"} value={type}>{type || "Automatic"}</option>
              ))}
            </select>
          </label>
          <label>
            Evidence results
            <input
              type="range"
              min="1"
              max="5"
              value={active.limit}
              onChange={(event) => updateActive({ ...active, limit: Number(event.target.value) }, false)}
            />
            <span>{active.limit}</span>
          </label>
        </div>
        <p className="disclaimer">Research prototype. Not legal advice.</p>
      </aside>

      <main className="main">
        <header>
          <h1>Grounded contract answers</h1>
          <p>Answers are limited to five supported CUAD clause categories and include retrieved evidence.</p>
        </header>
        <section className="messages">
          {active.messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              <div className="avatar">{message.role === "user" ? "You" : "CL"}</div>
              <div className="bubble">
                <p>{message.content}</p>
                {message.response?.results.length ? (
                  <details>
                    <summary>Show retrieved evidence ({message.response.results.length})</summary>
                    <div className="evidence-list">
                      {message.response.results.map((result, index) => (
                        <section className="evidence" key={result.id}>
                          <div><strong>[{index + 1}] {result.clause_type}</strong><span>{result.source_pdf}</span></div>
                          <p>{result.text}</p>
                          <small>
                            {result.reranker_score !== null
                              ? `reranker ${result.reranker_score.toFixed(3)}`
                              : result.fused_score !== null
                                ? `hybrid ${result.fused_score.toFixed(4)}`
                                : `vector ${result.score.toFixed(3)}`}
                          </small>
                        </section>
                      ))}
                    </div>
                  </details>
                ) : null}
              </div>
            </article>
          ))}
          {status ? <div className="status"><span />{status}</div> : null}
          <div ref={bottomRef} />
        </section>
        <form className="composer" onSubmit={submit}>
          {error ? <div className="error">{error}</div> : null}
          <div className="input-row">
            <textarea
              maxLength={1000}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Ask about assignment, liability, licensing, audit rights, or termination…"
              disabled={busy}
            />
            <button disabled={busy || !question.trim()}>{busy ? "Working…" : "Send"}</button>
          </div>
          <small>3 requests/minute and 10 requests/day per visitor.</small>
          <div ref={turnstileRef} />
        </form>
      </main>
    </div>
  );
}
