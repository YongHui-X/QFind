import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { bootstrapSession, streamChat } from "./api";
import { chatTitle, loadChats, persistChats } from "./storage";
import type { Evidence, Message, SavedChat } from "./types";
import "./styles.css";

const DEFAULT_EVIDENCE_LIMIT = 3;

const GREETING =
  "Ask a contract question and I'll answer using retrieved evidence from the supported CUAD clauses.";

const STATUS_LABELS: Record<string, string> = {
  routing: "Identifying clause type",
  contextualizing: "Resolving follow-up",
  retrieved: "Evidence retrieved",
  reranked: "Evidence reranked",
  generating: "Drafting grounded answer",
};

const SUGGESTED_QUESTIONS = [
  {
    category: "Assignment",
    question:
      "Which TomOnline Co-Branding Agreement anti-assignment passage allows sublicensing only after Online BVI's prior written approval?",
  },
  {
    category: "Liability",
    question:
      "Which NETGEAR Distributor Agreement passages cap total liability for damages under the agreement?",
  },
  {
    category: "Licensing",
    question:
      "Which Cerence Intellectual Property Agreement passage grants SpinCo a worldwide license under the Nuance patents?",
  },
  {
    category: "Audit",
    question:
      "Which TomOnline Co-Branding Agreement passage gives each party quarterly audit rights over books and records on fifteen days notice?",
  },
  {
    category: "Termination",
    question:
      "Which Cardlytics Maintenance Agreement passage lets Bank of America terminate the agreement, an order, or customization schedules for convenience on forty-five days notice?",
  },
];

function normalizeChats(chats: SavedChat[]): SavedChat[] {
  return chats.map((chat) => ({
    ...chat,
    clauseType: "",
    limit: Math.min(Math.max(chat.limit || DEFAULT_EVIDENCE_LIMIT, 1), DEFAULT_EVIDENCE_LIMIT),
  }));
}

function newChat(): SavedChat {
  return {
    id: crypto.randomUUID(),
    title: "New chat",
    messages: [{ id: crypto.randomUUID(), role: "assistant", content: GREETING }],
    clauseType: "",
    limit: DEFAULT_EVIDENCE_LIMIT,
    updatedAt: new Date().toISOString(),
  };
}

function chatSnippet(chat: SavedChat): string {
  const latest = [...chat.messages].reverse().find((message) => message.role === "user");
  return latest?.content ?? "Start a new review";
}

function displayEvidenceForMessage(message: Message): {
  citedOnly: boolean;
  items: Array<{ citationNumber: number; result: Evidence }>;
} {
  const results = message.response?.results ?? [];
  const citedNumbers = Array.from(
    new Set(
      [...message.content.matchAll(/\[(\d+)\]/g)]
        .map((match) => Number(match[1]))
        .filter((value) => Number.isInteger(value) && value >= 1 && value <= results.length),
    ),
  );
  const numbers = citedNumbers.length
    ? citedNumbers
    : results.map((_, index) => index + 1);
  return {
    citedOnly: citedNumbers.length > 0,
    items: numbers.map((number) => ({
      citationNumber: number,
      result: results[number - 1],
    })),
  };
}

export default function App() {
  const [chats, setChats] = useState<SavedChat[]>(() => normalizeChats(loadChats()));
  const [active, setActive] = useState<SavedChat>(() => {
    const initial = loadChats()[0] ?? newChat();
    return { ...initial, clauseType: "", limit: DEFAULT_EVIDENCE_LIMIT };
  });
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [historyQuery, setHistoryQuery] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bootstrapSession().catch((caught) => {
      setError(caught instanceof Error ? caught.message : "Session setup failed");
    });
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [active.messages, status]);

  const recentChats = useMemo(
    () => chats.slice().sort((a, b) => b.updatedAt.localeCompare(a.updatedAt)),
    [chats],
  );

  const visibleChats = useMemo(() => {
    const clean = historyQuery.trim().toLowerCase();
    if (!clean) return recentChats;
    return recentChats.filter((chat) => {
      const haystack = `${chat.title} ${chatSnippet(chat)}`.toLowerCase();
      return haystack.includes(clean);
    });
  }, [historyQuery, recentChats]);

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
    setStatus("Starting retrieval...");
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
      await streamChat(
        pending.messages.filter(
          (message, index) => !(index === 0 && message.content === GREETING),
        ),
        pending.clauseType,
        pending.limit,
        (streamEvent) => {
          if (streamEvent.event === "status") {
            setStatus(STATUS_LABELS[streamEvent.stage] ?? "Working");
          } else if (streamEvent.event === "token") {
            setStatus("");
            streamed += streamEvent.delta;
            setActive({
              ...pending,
              messages: [
                ...pending.messages,
                {
                  id: "streaming",
                  role: "assistant",
                  content: streamed,
                },
              ],
            });
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
    if (active.id === id) {
      setActive(
        updated[0]
          ? { ...updated[0], limit: DEFAULT_EVIDENCE_LIMIT }
          : newChat(),
      );
    }
  }

  function useSuggestedQuestion(suggestedQuestion: string) {
    setQuestion(suggestedQuestion);
    textareaRef.current?.focus();
  }

  async function copySuggestedQuestion(suggestedQuestion: string) {
    await navigator.clipboard.writeText(suggestedQuestion);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar sidebar-left">
        <div className="brand">
          <span className="brand-mark">QF</span>
          <div className="brand-copy">
            <strong>QFind</strong>
            <small>Contract answers with cited evidence</small>
          </div>
        </div>

        <button className="new-chat" onClick={() => setActive(newChat())} type="button">
          <span className="new-chat-icon">+</span>
          <span className="new-chat-copy">
            <strong>New chat</strong>
          </span>
        </button>

        <label className="left-search">
          <span>Search chats</span>
          <input
            type="search"
            value={historyQuery}
            onChange={(event) => setHistoryQuery(event.target.value)}
            placeholder="Search chats"
          />
        </label>

        <nav className="history-mini-list" aria-label="Chat history">
          {visibleChats.map((chat) => (
            <div
              className={`history-mini-row ${chat.id === active.id ? "active" : ""}`}
              key={chat.id}
            >
              <button
                className="history-mini-item"
                type="button"
                onClick={() => setActive({ ...chat, clauseType: "", limit: DEFAULT_EVIDENCE_LIMIT })}
              >
                <span>{chat.title === "New chat" ? chatSnippet(chat) : chat.title}</span>
              </button>
              <button
                className="history-delete"
                type="button"
                aria-label={`Delete ${chat.title}`}
                title="Delete chat"
                onClick={() => deleteChat(chat.id)}
              >
                x
              </button>
            </div>
          ))}
        </nav>

        <p className="disclaimer">Research prototype. Not legal advice.</p>
      </aside>

      <main className="main">
        <section className="messages">
          {active.messages.map((message) => {
            const evidence = displayEvidenceForMessage(message);
            return (
              <article className={`message ${message.role}`} key={message.id}>
                <div className="avatar">{message.role === "user" ? "You" : "QF"}</div>
                <div className="bubble">
                  <p>{message.content}</p>
                  {evidence.items.length ? (
                    <details>
                      <summary>
                        {evidence.citedOnly ? "Show cited evidence" : "Show retrieved evidence"} (
                        {evidence.items.length})
                      </summary>
                      <div className="evidence-list">
                        {evidence.items.map(({ citationNumber, result }) => (
                          <section className="evidence" key={result.id}>
                            <div className="evidence-header">
                              <strong>
                                [{citationNumber}] {result.clause_type}
                              </strong>
                              <span>{result.source_pdf}</span>
                            </div>
                            <p>{result.text}</p>
                          </section>
                        ))}
                      </div>
                    </details>
                  ) : null}
                </div>
              </article>
            );
          })}
          {status ? (
            <article className="message assistant pending-message" aria-live="polite">
              <div className="avatar">QF</div>
              <div className="bubble pending-bubble">
                <div className="status">
                  <span />
                  {status}
                </div>
              </div>
            </article>
          ) : null}
          <div ref={bottomRef} />
        </section>

        <form className="composer" onSubmit={submit}>
          {error ? <div className="error">{error}</div> : null}
          <div className="input-row">
            <textarea
              ref={textareaRef}
              maxLength={1000}
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  event.currentTarget.form?.requestSubmit();
                }
              }}
              placeholder="Ask about assignment, liability, licensing, audit rights, or termination..."
              disabled={busy}
            />
            <button disabled={busy || !question.trim()}>{busy ? "Working..." : "Send"}</button>
          </div>
          <small>Grounded retrieval over the indexed CUAD subset. Not legal advice.</small>
        </form>
      </main>

      <aside className="sidebar-right" aria-label="Suggested questions">
        <div className="suggestions-header">
          <span>Suggested questions</span>
        </div>
        <div className="suggestion-list">
          {SUGGESTED_QUESTIONS.map((item) => (
            <div className="suggestion-item" key={item.question}>
              <button
                className="suggestion-fill"
                type="button"
                onClick={() => useSuggestedQuestion(item.question)}
                disabled={busy}
                title="Use this question"
              >
                <span>{item.category}</span>
                <strong>{item.question}</strong>
              </button>
              <button
                className="suggestion-copy"
                type="button"
                onClick={() => copySuggestedQuestion(item.question)}
                title="Copy question"
                aria-label={`Copy ${item.category} question`}
              />
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}
