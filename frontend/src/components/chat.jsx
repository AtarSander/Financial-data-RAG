import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import "./chat.css";

function uid() {
  return Math.random().toString(16).slice(2) + Date.now().toString(16);
}

function normalizeSources(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  return [];
}

export default function Chat() {
  const [messages, setMessages] = useState([
    {
      id: uid(),
      role: "assistant",
      content: "Hey — ask me something. I'll answer using your docs (RAG).",
      sources: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const listRef = useRef(null);
  const bottomRef = useRef(null);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, loading]);

  async function send() {
    const text = input.trim();
    if (!text || loading) return;

    const userMsg = { id: uid(), role: "user", content: text, sources: [] };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      const res = await api.get("/answer", {
        params: { query: text },
      });
      const answer = res.data?.answer ?? "";
      console.log(answer)
      const sources = normalizeSources(res.data?.sources);

      const botMsg = {
        id: uid(),
        role: "assistant",
        content: answer || "(empty response)",
        sources,
      };

      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      const msg =
        err?.response?.data?.detail ||
        err?.message ||
        "Request failed";

      setMessages((prev) => [
        ...prev,
        { id: uid(), role: "assistant", content: `⚠️ ${msg}`, sources: [] },
      ]);
    } finally {
      setLoading(false);
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  }

  return (
    <div className="page">
      <header className="topbar">
        <div className="brand">
          <div className="dot" />
          <div>
            <div className="title">RAG Chat</div>
            <div className="subtitle">{loading ? "Thinking…" : "Ready"}</div>
          </div>
        </div>

       
      </header>

      <main className="shell">
        <div className="chat">
          <div className="messages" ref={listRef}>
            {messages.map((m) => (
              <Message key={m.id} msg={m} />
            ))}

            {loading && (
              <div className="row assistant">
                <div className="avatar">AI</div>
                <div className="bubble">
                  <div className="typing">
                    <span />
                    <span />
                    <span />
                  </div>
                </div>
              </div>
            )}

            <div ref={bottomRef} />
          </div>

          <div className="composer">
            <textarea
              className="input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Message your RAG bot… (Enter to send, Shift+Enter newline)"
              rows={1}
              disabled={loading}
            />
            <button className="send" onClick={send} disabled={!canSend}>
              Send
            </button>
          </div>

        </div>
      </main>
    </div>
  );
}

function Message({ msg }) {
  const isUser = msg.role === "user";
  const sources = msg.sources || [];

  return (
    <div className={`row ${isUser ? "user" : "assistant"}`}>
      <div className="avatar">{isUser ? "You" : "AI"}</div>

      <div className="stack">
        <div className={`bubble ${isUser ? "userBubble" : "aiBubble"}`}>
          <div className="content">{msg.content}</div>
        </div>

        {!isUser && sources.length > 0 && (
          <details className="sources">
            <summary>Sources ({sources.length})</summary>
            <div className="sourcesBody">
              {sources.map((s, i) => (
                <div key={i} className="sourceCard">
                  <div className="sourceTitle">
                    {s.url ? (
                      <a href={s.url} target="_blank" rel="noreferrer">
                        {s.title || s.url}
                      </a>
                    ) : (
                      <span>{s.title || `Source ${i + 1}`}</span>
                    )}
                  </div>
                  {s.snippet && <div className="sourceSnippet">{s.snippet}</div>}
                </div>
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  );
}
