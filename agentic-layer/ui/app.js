const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send-button");
const statusEl = document.querySelector("#status");
const modeButtons = [...document.querySelectorAll(".mode-button")];
const quickPromptButtons = [...document.querySelectorAll("[data-prompt]")];

let activeMode = "ask";

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
      const isList = lines.every((line) => /^\s*[-*]\s+/.test(line));
      if (isList) {
        const items = lines
          .map((line) => line.replace(/^\s*[-*]\s+/, ""))
          .map((line) => `<li>${formatInlineMarkdown(line)}</li>`)
          .join("");
        return `<ul>${items}</ul>`;
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

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    await response.json();
    statusEl.textContent = "Workspace assistant ready";
    statusEl.classList.remove("error");
  } catch (error) {
    statusEl.textContent = "Local assistant is not reachable.";
    statusEl.classList.add("error");
  }
}

modeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeMode = button.dataset.mode;
    modeButtons.forEach((item) => item.classList.toggle("active", item === button));
  });
});

quickPromptButtons.forEach((button) => {
  button.addEventListener("click", () => {
    input.value = button.dataset.prompt || "";
    input.focus();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message) {
    return;
  }

  addMessage("user", message);
  input.value = "";
  sendButton.disabled = true;
  sendButton.textContent = "Thinking";
  const pending = addMessage("assistant", "Working through the workspace context...");

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
      return;
    }

    const meta = "Workspace answer.";
    addMessage("assistant", data.answer, meta);
  } catch (error) {
    pending.remove();
    addMessage("assistant", "The local agentic API could not be reached. Check that the server is running.");
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "Send";
    input.focus();
  }
});

input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    form.requestSubmit();
  }
});

loadHealth();
