const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send-button");
const statusEl = document.querySelector("#status");
const statusDot = document.querySelector("#status-dot");
const sessionSummary = document.querySelector("#session-summary");
const draftCount = document.querySelector("#draft-count");
const historyList = document.querySelector("#history-list");
const historyCount = document.querySelector("#history-count");
const copyLastButton = document.querySelector("#copy-last");
const clearChatButton = document.querySelector("#clear-chat");
const insertContextButton = document.querySelector("#insert-context");
const modeButtons = [...document.querySelectorAll(".mode-button")];
const promptButtons = [...document.querySelectorAll("[data-prompt]")];

let activeMode = "ask";
let sessionNumber = Number(localStorage.getItem("agenticLayerSessionNumber") || "1");
let currentSessionId = localStorage.getItem("agenticLayerCurrentSessionId") || createSessionId();
let questionCount = 0;
let lastAssistantAnswer = "";
let sessionTitle = "Untitled analysis";
let sessionHistory = loadSessionHistory();
let sessionMessages = loadSessionMessages();

const starterMessage =
  "Ask about workflow, forecast logic, report generation, APIs, state, or implementation strategy. I will use the local workspace context and keep the thread focused.";

const workflowContextPrompt =
  "Use the confirmed application workflow context while answering: login, data upload, data understanding, EDA, data cleaning, time series forecast, machine learning forecast, loss forecast, profit forecast, ML assistant, prediction, and report.";
const sendIcon =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4.4 19.4 20.9 12 4.4 4.6 4 10.4l9.1 1.6L4 13.6l.4 5.8Z"/></svg>';
const botIcon =
  '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5a4 4 0 0 1 8 0h1a3 3 0 0 1 3 3v7a4 4 0 0 1-4 4H8a4 4 0 0 1-4-4V8a3 3 0 0 1 3-3h1Zm2 0h4a2 2 0 0 0-4 0Zm-3 2a1 1 0 0 0-1 1v7a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V8a1 1 0 0 0-1-1H7Zm2.5 7a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3Zm5 0a1.5 1.5 0 1 1 0-3 1.5 1.5 0 0 1 0 3Z"/></svg>';

function createSessionId() {
  return `session-${Date.now()}-${Math.random().toString(16).slice(2, 8)}`;
}

function loadSessionHistory() {
  try {
    const stored = JSON.parse(localStorage.getItem("agenticLayerSessionHistory") || "[]");
    return Array.isArray(stored) ? stored : [];
  } catch (error) {
    return [];
  }
}

function saveSessionHistory() {
  localStorage.setItem("agenticLayerSessionHistory", JSON.stringify(sessionHistory.slice(0, 12)));
}

function loadSessionMessages() {
  try {
    const stored = JSON.parse(localStorage.getItem("agenticLayerSessionMessages") || "{}");
    return stored && typeof stored === "object" ? stored : {};
  } catch (error) {
    return {};
  }
}

function saveSessionMessages() {
  localStorage.setItem("agenticLayerSessionMessages", JSON.stringify(sessionMessages));
}

function persistMessage(role, text, meta = "") {
  const items = sessionMessages[currentSessionId] || [];
  sessionMessages[currentSessionId] = [...items, { role, text, meta }].slice(-40);
  saveSessionMessages();
}

function hydrateCurrentSession() {
  const current = sessionHistory.find((item) => item.id === currentSessionId);
  if (!current) {
    return;
  }
  sessionTitle = current.title || sessionTitle;
  questionCount = Number(current.count || 0);
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdownLite(value) {
  const codeBlocks = [];
  const withoutCodeBlocks = value.replace(/```([a-zA-Z0-9_-]+)?\n?([\s\S]*?)```/g, (_match, language, code) => {
    const label = language ? `<div class="code-label">${escapeHtml(language)}</div>` : "";
    codeBlocks.push(`<pre>${label}<code>${escapeHtml(code.trimEnd())}</code></pre>`);
    return `\n\n__CODE_BLOCK_${codeBlocks.length - 1}__\n\n`;
  });

  const escaped = escapeHtml(withoutCodeBlocks);
  return escaped
    .split(/\n{2,}/)
    .map((block) => {
      const codeMatch = block.match(/^__CODE_BLOCK_(\d+)__$/);
      if (codeMatch) {
        return codeBlocks[Number(codeMatch[1])] || "";
      }

      const lines = block.split("\n").filter(Boolean);
      const isUnorderedList = lines.every((line) => /^\s*[-*]\s+/.test(line));
      const isOrderedList = lines.every((line) => /^\s*\d+\.\s+/.test(line));

      if (isUnorderedList || isOrderedList) {
        const tag = isOrderedList ? "ol" : "ul";
        const marker = isOrderedList ? /^\s*\d+\.\s+/ : /^\s*[-*]\s+/;
        const items = lines
          .map((line) => line.replace(marker, ""))
          .map((line) => `<li>${formatInlineMarkdown(line)}</li>`)
          .join("");
        return `<${tag}>${items}</${tag}>`;
      }

      return `<p>${formatInlineMarkdown(block).replaceAll("\n", "<br>")}</p>`;
    })
    .join("");
}

function formatInlineMarkdown(value) {
  return value
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function formatModeLabel(mode) {
  return (mode || "ask").charAt(0).toUpperCase() + (mode || "ask").slice(1);
}

function setMode(mode) {
  activeMode = mode || "ask";
  modeButtons.forEach((item) => item.classList.toggle("active", item.dataset.mode === activeMode));
  recordActivity("mode_changed", "Mode changed", formatModeLabel(activeMode));
}

async function recordActivity(type, title, detail = "") {
  try {
    await fetch("/api/activity", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        type,
        title,
        detail,
        session: `Session ${String(sessionNumber).padStart(2, "0")}`,
        mode: activeMode,
      }),
    });
  } catch (error) {
    console.warn("Activity logging unavailable", error);
  }
}

function updateSessionSummary() {
  sessionSummary.textContent = questionCount
    ? `${questionCount} request${questionCount === 1 ? "" : "s"} in current session`
    : "Ready for workspace analysis";
}

function compactTitle(value) {
  const cleaned = value.replace(/\s+/g, " ").trim();
  if (!cleaned) {
    return "Untitled analysis";
  }
  return cleaned.length > 54 ? `${cleaned.slice(0, 51)}...` : cleaned;
}

function formatHistoryTime(timestamp) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) {
    return "Recent";
  }
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function upsertCurrentSession(message = "") {
  if (message && sessionTitle === "Untitled analysis") {
    sessionTitle = compactTitle(message);
  }

  const entry = {
    id: currentSessionId,
    title: sessionTitle,
    mode: formatModeLabel(activeMode),
    count: questionCount,
    updatedAt: new Date().toISOString(),
  };

  sessionHistory = [entry, ...sessionHistory.filter((item) => item.id !== currentSessionId)].slice(0, 12);
  saveSessionHistory();
  renderSessionHistory();
}

function renderSessionHistory() {
  if (!historyList || !historyCount) {
    return;
  }

  historyCount.textContent = String(sessionHistory.length);

  if (!sessionHistory.length) {
    historyList.innerHTML = '<div class="empty-history">No saved sessions yet</div>';
    return;
  }

  historyList.innerHTML = sessionHistory
    .map((item) => {
      const active = item.id === currentSessionId ? " active" : "";
      return `
        <button class="history-item${active}" type="button" data-session-id="${escapeHtml(item.id)}">
          <span class="history-title">${escapeHtml(item.title)}</span>
          <span class="history-meta">${escapeHtml(item.mode)} - ${item.count} request${item.count === 1 ? "" : "s"} - ${formatHistoryTime(item.updatedAt)}</span>
        </button>
      `;
    })
    .join("");
}

function selectHistoryItem(sessionId) {
  const item = sessionHistory.find((entry) => entry.id === sessionId);
  if (!item) {
    return;
  }

  currentSessionId = item.id;
  sessionTitle = item.title;
  questionCount = item.count;
  localStorage.setItem("agenticLayerCurrentSessionId", currentSessionId);
  updateSessionSummary();
  renderSessionHistory();
  renderStoredMessages();
  recordActivity("session_selected", "Session selected", item.title);
}

function renderStoredMessages() {
  messages.innerHTML = "";
  const stored = sessionMessages[currentSessionId] || [];
  if (!stored.length) {
    addMessage("assistant", starterMessage, "", false);
    return;
  }
  stored.forEach((item) => addMessage(item.role, item.text, item.meta || "", false));
  const lastAssistant = [...stored].reverse().find((item) => item.role === "assistant");
  lastAssistantAnswer = lastAssistant?.text || "";
}

function addMessage(role, text, meta = "", persist = true) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  if (role === "user") {
    avatar.textContent = "You";
  } else {
    avatar.classList.add("bot-avatar");
    avatar.innerHTML = botIcon;
  }

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderMarkdownLite(text);

  if (meta) {
    const metaEl = document.createElement("div");
    metaEl.className = "meta";
    metaEl.textContent = meta;
    bubble.appendChild(metaEl);
  }

  article.append(avatar, bubble);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
  if (persist) {
    persistMessage(role, text, meta);
  }
  return article;
}

function resetConversation() {
  messages.innerHTML = "";
  questionCount = 0;
  lastAssistantAnswer = "";
  sessionNumber += 1;
  currentSessionId = createSessionId();
  sessionTitle = "Untitled analysis";
  lastAssistantAnswer = "";
  sessionMessages[currentSessionId] = [];
  saveSessionMessages();
  localStorage.setItem("agenticLayerSessionNumber", String(sessionNumber));
  localStorage.setItem("agenticLayerCurrentSessionId", currentSessionId);
  updateSessionSummary();
  renderSessionHistory();
  addMessage("assistant", starterMessage, "", false);
  recordActivity("session_reset", "New session", "Conversation workspace reset");
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    await response.json();
    statusEl.textContent = "Assistant ready";
    statusEl.classList.remove("error");
    statusDot.style.background = "var(--success)";
  } catch (error) {
    statusEl.textContent = "Assistant offline";
    statusEl.classList.add("error");
    statusDot.style.background = "var(--danger)";
  }
}

function setPromptFromButton(button) {
  const mode = button.dataset.mode;
  if (mode) {
    setMode(mode);
  }
  input.value = button.dataset.prompt || "";
  updateDraftCount();
  input.focus();
  recordActivity("prompt_selected", "Prompt selected", button.querySelector("strong")?.textContent || input.value);
}

function updateDraftCount() {
  const count = input.value.length;
  draftCount.textContent = `${count} character${count === 1 ? "" : "s"}`;
}

modeButtons.forEach((button) => {
  button.addEventListener("click", () => setMode(button.dataset.mode));
});

promptButtons.forEach((button) => {
  button.addEventListener("click", () => setPromptFromButton(button));
});

historyList?.addEventListener("click", (event) => {
  const item = event.target.closest("[data-session-id]");
  if (!item) {
    return;
  }
  selectHistoryItem(item.dataset.sessionId);
});

insertContextButton.addEventListener("click", () => {
  const prefix = input.value.trim() ? `${input.value.trim()}\n\n` : "";
  input.value = `${prefix}${workflowContextPrompt}`;
  updateDraftCount();
  input.focus();
  recordActivity("context_inserted", "Workflow context inserted", "Confirmed workflow context added to draft");
});

copyLastButton.addEventListener("click", async () => {
  if (!lastAssistantAnswer) {
    recordActivity("copy_skipped", "Copy skipped", "No assistant answer available yet");
    return;
  }

  try {
    await navigator.clipboard.writeText(lastAssistantAnswer);
    recordActivity("answer_copied", "Copied answer", "Latest assistant response copied");
  } catch (error) {
    recordActivity("copy_unavailable", "Copy unavailable", "Browser clipboard access was blocked");
  }
});

clearChatButton.addEventListener("click", resetConversation);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) {
    return;
  }

  addMessage("user", message);
  questionCount += 1;
  updateSessionSummary();
  upsertCurrentSession(message);
  recordActivity("request_submitted", "Request submitted", `${formatModeLabel(activeMode)} mode`);

  input.value = "";
  updateDraftCount();
  sendButton.disabled = true;
  sendButton.textContent = "...";
  const pending = addMessage("assistant", "Analyzing workspace context, relevant files, and confirmed workflow knowledge...", "", false);

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        mode: activeMode,
        provider: "auto",
      }),
    });

    const data = await response.json();
    pending.remove();

    if (!response.ok) {
      addMessage("assistant", data.error || "The agentic layer returned an error.");
      recordActivity("request_failed", "Request failed", "The local API returned an error");
      return;
    }

    lastAssistantAnswer = data.answer;
    addMessage("assistant", data.answer, "Workspace intelligence response");
    recordActivity("response_completed", "Response completed", "Answer generated with workspace context");
  } catch (error) {
    pending.remove();
    addMessage("assistant", "The local agentic API could not be reached. Check that the server is running.");
    recordActivity("connection_issue", "Connection issue", "Local assistant API was unreachable");
  } finally {
    sendButton.disabled = false;
    sendButton.innerHTML = sendIcon;
    input.focus();
  }
});

input.addEventListener("input", updateDraftCount);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    form.requestSubmit();
  }
});

setMode(activeMode);
localStorage.setItem("agenticLayerCurrentSessionId", currentSessionId);
hydrateCurrentSession();
updateSessionSummary();
updateDraftCount();
renderSessionHistory();
if (sessionMessages[currentSessionId]?.length) {
  renderStoredMessages();
}
loadHealth();
