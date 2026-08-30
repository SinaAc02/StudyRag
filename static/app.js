const state = document.querySelector("#state");
const summary = document.querySelector("#summary");
const documents = document.querySelector("#documents");
const error = document.querySelector("#error");
const answerCard = document.querySelector("#answer-card");

async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Request failed");
  return data;
}

function showStatus(status) {
  state.textContent = status.state;
  summary.textContent = `${status.documents.length} PDF(s), ${status.chunks} searchable chunks. ${status.message}`;
  documents.replaceChildren(
    ...status.documents.map((name) => {
      const item = document.createElement("span");
      item.textContent = name;
      return item;
    }),
  );
}

async function loadStatus() {
  try {
    showStatus(await request("/api/status"));
  } catch (exception) {
    summary.textContent = exception.message;
  }
}

document.querySelector("#reindex").addEventListener("click", async (event) => {
  event.currentTarget.disabled = true;
  summary.textContent = "Checking the documents folder…";
  try {
    showStatus(await request("/api/reindex", { method: "POST" }));
  } catch (exception) {
    summary.textContent = exception.message;
  } finally {
    event.currentTarget.disabled = false;
  }
});

document.querySelector("#question-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = event.currentTarget.querySelector("button");
  button.disabled = true;
  button.textContent = "Thinking…";
  error.hidden = true;
  answerCard.hidden = true;
  try {
    const result = await request("/api/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: document.querySelector("#question").value }),
    });
    document.querySelector("#answer").textContent = result.text;
    document.querySelector("#sources").replaceChildren(
      ...result.sources.map((source) => {
        const item = document.createElement("article");
        const title = document.createElement("strong");
        title.textContent = `${source.document} · page ${source.page}`;
        const excerpt = document.createElement("p");
        excerpt.textContent = source.excerpt;
        item.append(title, excerpt);
        return item;
      }),
    );
    answerCard.hidden = false;
  } catch (exception) {
    error.textContent = exception.message;
    error.hidden = false;
  } finally {
    button.disabled = false;
    button.textContent = "Explain";
  }
});

loadStatus();
