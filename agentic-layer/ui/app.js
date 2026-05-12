const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send-button");
const statusEl = document.querySelector("#status");
const statusDot = document.querySelector("#status-dot");
const activeModeLabel = document.querySelector("#active-mode-label");
const sessionSummary = document.querySelector("#session-summary");
const sessionPill = document.querySelector("#session-pill");
const activityFeed = document.querySelector("#activity-feed");
const activityCount = document.querySelector("#activity-count");
const draftCount = document.querySelector("#draft-count");
const copyLastButton = document.querySelector("#copy-last");
const clearChatButton = document.querySelector("#clear-chat");
const insertContextButton = document.querySelector("#insert-context");
const modeButtons = [...document.querySelectorAll(".mode-button")];
const promptButtons = [...document.querySelectorAll("[data-prompt]")];

let activeMode = "ask";
let sessionNumber = Number(localStorage.getItem("agenticLayerSessionNumber") || "1");
let questionCount = 0;
let lastAssistantAnswer = "";
let activityItems = 1;

const starterMessage =
  "Tell me what you want to understand or plan. I can map workflows, locate implementation files, summarize modules, review risks, and produce structured next steps.";

const workflowContextPrompt =
  "Use the confirmed application workflow context while answering: login, data upload, data understanding, EDA, data cleaning, time series forecast, machine learning forecast, loss forecast, profit forecast, ML assistant, prediction, and report.";

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderMarkdownLite(value) {
  const escaped = escapeHtml(value);
  const withCodeBlocks = escaped.replace(/```([\s\S]*?)```/g, "<pre><code>$1</code></pre>");

  return withCodeBlocks
    .split(/\n{2,}/)
    .map((block) => {
      if (block.startsWith("<pre>")) {
        return block;
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

function setMode(mode) {
  activeMode = mode || "ask";
  modeButtons.forEach((item) => item.classList.toggle("active", item.dataset.mode === activeMode));
  activeModeLabel.textContent = activeMode.charAt(0).toUpperCase() + activeMode.slice(1);
}

function addActivity(title, detail) {
  activityItems += 1;
  const item = document.createElement("li");
  item.innerHTML = `<strong>${escapeHtml(title)}</strong><span>${escapeHtml(detail)}</span>`;
  activityFeed.prepend(item);
  activityCount.textContent = String(activityItems - 1);

  while (activityFeed.children.length > 7) {
    activityFeed.lastElementChild.remove();
  }
}

function updateSessionSummary() {
  sessionPill.textContent = `Session ${String(sessionNumber).padStart(2, "0")}`;
  sessionSummary.textContent = questionCount
    ? `${questionCount} question${questionCount === 1 ? "" : "s"} in this session`
    : "No questions yet";
}

function addMessage(role, text, meta = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "You" : "AI";

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
  return article;
}

function resetConversation() {
  messages.innerHTML = "";
  questionCount = 0;
  lastAssistantAnswer = "";
  sessionNumber += 1;
  localStorage.setItem("agenticLayerSessionNumber", String(sessionNumber));
  updateSessionSummary();
  addMessage("assistant", starterMessage);
  addActivity("New session", "Conversation workspace reset");
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    await response.json();
    statusEl.textContent = "Workspace assistant ready";
    statusEl.classList.remove("error");
    statusDot.style.background = "var(--success)";
  } catch (error) {
    statusEl.textContent = "Local assistant is not reachable.";
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

insertContextButton.addEventListener("click", () => {
  const prefix = input.value.trim() ? `${input.value.trim()}\n\n` : "";
  input.value = `${prefix}${workflowContextPrompt}`;
  updateDraftCount();
  input.focus();
});

copyLastButton.addEventListener("click", async () => {
  if (!lastAssistantAnswer) {
    addActivity("Copy skipped", "No assistant answer available yet");
    return;
  }

  try {
    await navigator.clipboard.writeText(lastAssistantAnswer);
    addActivity("Copied answer", "Latest assistant response copied");
  } catch (error) {
    addActivity("Copy unavailable", "Browser clipboard access was blocked");
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
  addActivity("Request submitted", `${activeModeLabel.textContent} mode`);

  input.value = "";
  updateDraftCount();
  sendButton.disabled = true;
  sendButton.textContent = "Thinking";
  const pending = addMessage("assistant", "Analyzing workspace context, relevant files, and confirmed workflow knowledge...");

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
      addActivity("Request failed", "The local API returned an error");
      return;
    }

    lastAssistantAnswer = data.answer;
    addMessage("assistant", data.answer, "Workspace intelligence response");
    addActivity("Response completed", "Answer generated with workspace context");
  } catch (error) {
    pending.remove();
    addMessage("assistant", "The local agentic API could not be reached. Check that the server is running.");
    addActivity("Connection issue", "Local assistant API was unreachable");
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Send";
    input.focus();
  }
});

input.addEventListener("input", updateDraftCount);

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    form.requestSubmit();
  }
});

setMode(activeMode);
updateSessionSummary();
updateDraftCount();
loadHealth();
