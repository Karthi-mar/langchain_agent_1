# Study Buddy — a LangChain + LangGraph Agent

A conversational study assistant built with [LangChain](https://www.langchain.com/) and [LangGraph](https://www.langchain.com/langgraph), built as a hands-on follow-up to LangChain's course. It goes beyond a Q&A bot by giving the agent tools to call and persistent memory across turns in a conversation.

Ask it a math question or a factual question, and it decides which tool (if any) it needs, calls it, and answers using the result — while remembering earlier turns in the same session.

## Example session

```
Study buddy is ready! type 'quit or exit'.

You: what's 12*(3+4)?
Study Buddy: 12 * (3 + 4) = 84!

You: who was Alan Turing?
Study Buddy: Alan Turing (1912–1954) was an English mathematician, computer scientist, logician, and cryptanalyst. He is widely considered the father of theoretical computer science and artificial intelligence, famous for formalizing concepts of algorithm and computation with the Turing machine and for his pivotal codebreaking work during World War II.

You: exit
Good Bye!
```

## What it does

- **`calculator`** — evaluates arithmetic expressions (`+`, `-`, `*`, `/`, `**`, parentheses) so the model never has to "guess" at math in its head.
- **`wikipedia_search`** — looks up a topic on Wikipedia and returns a short summary, for factual questions about people, places, or events.
- **Persistent memory** — conversation history is kept per session via a LangGraph checkpointer, so you can ask a follow-up question that refers back to something earlier in the chat and the agent still has the context.

The agent decides on its own, per message, whether it needs a tool at all, and if so, which one — that routing logic isn't hand-written, it comes from `create_agent`'s tool-calling loop.

## Tech stack

| Piece | What it's for |
|---|---|
| [LangChain](https://python.langchain.com/) (`create_agent`) | Builds the tool-calling agent loop |
| [LangGraph](https://www.langchain.com/langgraph) (`InMemorySaver`) | Persists conversation state across turns, keyed by a thread ID |
| [Google Gemini](https://ai.google.dev/) (`gemini-flash-latest`) | The underlying LLM, via `langchain-google-genai` |
| `python-dotenv` | Loads the API key from a local `.env` file instead of hardcoding it |
| `wikipedia` (PyPI package) | Backs the `wikipedia_search` tool |

**Why `gemini-flash-latest` instead of a pinned version like `gemini-2.5-flash`:** Google periodically deprecates dated model names — `gemini-flash-latest` is an alias Google maintains to always point at their current, non-deprecated Flash model, so the project doesn't silently break every few months.

## Project structure

```
langchain_agent_1/
├── agent.py            # Tool definitions, agent setup, and the CLI chat loop
├── requirements.txt    # Python dependencies
├── .env                # Local API key — not committed, see Setup below
├── .gitignore
└── README.md
```

## Setup

1. Clone the repo and move into it:
   ```
   git clone https://github.com/Karthi-mar/langchain_agent_1.git
   cd langchain_agent_1
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate      # Windows
   source .venv/bin/activate   # macOS/Linux
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the project root with your [Google AI Studio](https://aistudio.google.com/) API key:
   ```
   GOOGLE_API_KEY=your_actual_key_here
   ```

## Usage

```
python agent.py
```

Chat with it in the terminal. Type `quit` or `exit` to end the session.

## Concepts this project demonstrates

- **Tool calling** — defining Python functions as agent tools with the `@tool` decorator, and letting the model choose when to invoke them based on their docstrings.
- **Multi-turn conversational memory** — using a LangGraph checkpointer (`InMemorySaver`) keyed by a `thread_id` so the agent retains context across an entire session, not just a single request/response.
- **Model provider abstraction** — using `init_chat_model` so the model provider (here, `google-genai`) is a configuration detail, not something wired throughout the code.
- **System prompt / persona design** — steering the agent's behavior and tool-use preferences through the system prompt.
- **Secrets management** — API key loaded from a local `.env` file (via `python-dotenv`), excluded from version control via `.gitignore`.
- **Per-tool error handling** — each tool catches its own failure modes (e.g. an ambiguous or missing Wikipedia page) and returns a clear message instead of crashing the whole agent.




Originally hardcoded `"gemini-2.5-flash"`. The very first time I ran the agent, Google's API rejected every request with a 404, saying that model was "no longer available to new users." It turns out it still shows up in Google's model list, but new API keys are quietly blocked from actually using it.

Fix — Switched to `gemini-flash-latest`, an alias Google maintains that always points at whatever their current, non-deprecated Flash model is:
```python
# before
model = init_chat_model("gemini-2.5-flash", model_provider="google-genai", temperature=0.3)

# after
model = init_chat_model("gemini-flash-latest", model_provider="google-genai", temperature=0.3)
```

Asking about "Alan Turing" failed every time, with the tool returning a vague "wikipedia lookup failed" message. Digging into the actual exception (not just the message the agent showed the user) revealed the real cause: the `wikipedia` package's `auto_suggest` feature — meant to autocorrect typos — had "corrected" **"Alan Turing" into "alan tuning"**, then failed to find a page with that name.

Fix — turned auto-suggest off, so the tool searches for exactly what was asked:
```python
# before
return wikipedia.summary(query, sentences=3, auto_suggest=True)

# after
return wikipedia.summary(query, sentences=3, auto_suggest=False)
```

After fixing the syntax errors, the agent technically worked, but the calculator answer printed like this instead of plain text:
```
Study Buddy: [{'type': 'text', 'text': 'The answer to $12 \\times (3 + 4)$ is **84**.', 'extras': {'signature': 'ErYBCrMB...'}}]
```
The cause: newer Gemini models return their answer as a *list of content blocks* (each with a `type`, the actual `text`, and an internal `signature` used for multi-turn tool-call verification) instead of a plain string. My code assumed `message.content` was always a plain string and printed it as-is — so it printed the whole Python list, signature and all.

Fix — check whether `content` is a list, and if so, pull out just the `text` field from each block:
```python
# before
reply = result["messages"][-1].content
print(f"Study Buddy: {reply} \n")

# after
reply = result["messages"][-1].content
if isinstance(reply, list):
    reply = "".join(block.get("text", "") for block in reply if isinstance(block, dict))
print(f"Study Buddy: {reply} \n")
```
- Memory is in-process only (`InMemorySaver`) — conversation history is lost when the script exits. Swapping in a persistent checkpointer (e.g. LangGraph's SQLite or Postgres saver) would let sessions survive restarts.
- No automated tests yet — the `calculator` tool is a pure function and would be an easy first target for a small `pytest` .
- Responses currently print all at once; a token-by-token streaming version (via `agent.stream(..., stream_mode="messages")`) would give a more interactive, ChatGPT-style typing effect.
- Only two tools so far — a natural extension would be a live web-search tool alongside the static Wikipedia lookup, or a note-saving tool to persist study notes to disk.
