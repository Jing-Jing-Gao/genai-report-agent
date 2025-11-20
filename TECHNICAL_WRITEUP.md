# Technical Write-Up – Generative AI Report Agent

## 1. Problem Understanding

The task is to build a **Generative AI Report Agent** that:

1. Automatically collects recent information from trusted websites (e.g. BBC, UK Government).
2. Periodically (e.g. every hour) summarises new content for a specified topic.
3. Produces a short, structured report including:
   - A 100–150 word summary,
   - 3–5 key takeaways,
   - Mentioned organizations / key terms.
4. Exposes a simple conversational interface where a user can:
   - Ask for the latest updates, and
   - Ask vague questions such as “What’s happening nowadays?”

The implementation should demonstrate:

- Clear logic and modularity,
- A working hourly report (or realistic mock),
- Use of summarisation techniques,
- Basic conversational context engineering,
- Good documentation.

My solution uses **Python**, **LangChain**, and a **local open-source LLM** served via **Ollama**, so it runs entirely on a local machine without paid APIs.

---

## 2. Solution Overview

The system is implemented in `app_langchain.py` with three main components:

1. **Data Collection** – fetching and cleaning news articles
2. **Automated Reporting** – summarisation and report creation
3. **Conversational Interface** – a chat loop grounded in the latest report

### 2.1 Data Collection – `NewsCollector`

**Goal:** Fetch recent, topic-related news from a trusted source.

- Source: **BBC Technology** RSS feed  
  `https://feeds.bbci.co.uk/news/technology/rss.xml`
- Libraries used:
  - `feedparser` to parse RSS feeds.
  - `beautifulsoup4` to clean HTML content.

**Logic:**

- Parse the RSS feed into entries.
- For each entry, read:
  - `title`
  - `summary` (HTML snippet)
  - `link` (URL)
  - `published_parsed` (date/time, if available)
- Filter entries by a **topic keyword** (e.g. `"AI"`, `"AI regulation"`, `"UK economy"`) appearing in the title or summary.
- Clean the HTML summary to plain text using `BeautifulSoup`.
- Convert each relevant entry into an `Article` dataclass with fields:
  - `source`: which feed it came from (e.g. `"bbc_technology"`)
  - `url`
  - `title`
  - `published_at` (ISO timestamp if available)
  - `text` (title + cleaned summary)

This gives a small, well-structured corpus ready for summarisation without fragile HTML scraping.

---

### 2.2 Automated Reporting – `ReportGenerator`

**Goal:** Turn a set of articles into a concise, structured report.

**Corpus construction:**

- `ReportGenerator` receives a list of `Article` objects.
- It builds a corpus string containing for each article:
  - Title
  - URL
  - Publication time
  - Text content
- Articles are separated with `---` and the corpus is truncated if it becomes too long (simple safety against very large prompts).

**LLM summarisation (LangChain + Ollama):**

- The project uses **LangChain’s `ChatOllama`** as the LLM interface.
- A local model such as `llama3` is run by Ollama.

The **system prompt** instructs the model to:

- Read the corpus about a specific topic,
- Produce a report that contains:
  1. A 100–150 word summary paragraph,
  2. 3–5 concise key takeaways,
  3. A list of mentioned organizations, entities, or important terms,
- Only use information present in the corpus (avoid inventing facts),
- Output **strict JSON** with keys:
  - `"summary"`,
  - `"key_takeaways"`,
  - `"organizations_and_terms"`.

The **user message** contains:

- The topic name,
- The constructed corpus,
- A reminder to output only valid JSON.

The LangChain `llm.invoke()` call returns the model response, which is then parsed as JSON:

- `summary` → string (100–150 word paragraph)
- `key_takeaways` → list of short bullet points
- `organizations_and_terms` → list of strings

If JSON parsing fails, there is a simple fallback that treats the full response as a free-form summary and fills in minimal placeholders, so the pipeline is robust even if the model output is imperfect.

**Report output:**

A full report dictionary is created with:

- `generated_at` (UTC timestamp),
- `topic`,
- `article_count`,
- `summary`,
- `key_takeaways`,
- `organizations_and_terms`,
- `articles` (the original article metadata).

This report is saved in two formats:

1. `reports/report_<timestamp>.json` – machine-readable.
2. `reports/report_<timestamp>.md` – human-readable Markdown with sections:
   - Summary
   - Key takeaways
   - Organizations / terms
   - List of article titles and URLs

The function `run_report_cycle(...)` orchestrates one complete cycle: collect → summarise → save.

---

### 2.3 Conversational Interface – `ChatAgent`

**Goal:** Let the user ask questions like “What’s new in AI today?” or “What’s happening nowadays?” and answer based on the latest report.

**Context construction:**

- The agent loads the **latest** JSON report from `reports/`.
- Builds a compact context string containing:
  - Topic and generation time
  - The summary
  - The key takeaways
  - The organizations / terms list

**Prompt design:**

- A system message describes the assistant’s behaviour:
  - It must answer questions using **only** the information in the provided report.
  - For vague questions like “What’s happening nowadays?” or “Any news?”, it should summarise the most important points from the report.
  - If the information is not in the report, it should say so and steer the user back to the topic.
- Another system message injects the text of the latest report (context).
- User queries are passed as `HumanMessage`, and responses are returned as `AIMessage`.

The agent keeps a simple `history` list of messages so follow-up questions can be interpreted in context, while still grounding everything in the latest report.

**Interface:**

- `chat_loop()` implements a simple command-line interface:
  - Prints a prompt (“Chatting about the latest report”).
  - Reads user input in a loop.
  - Calls `ask(question)` and prints the model’s answer.
  - Exits on `exit` or `quit`.

This satisfies the requirement for a conversational interface while keeping dependencies minimal.

---

### 2.4 Scheduling

To support **hourly reporting**, the project uses the `schedule` library:

- `start_hourly_loop(...)` configures:

  ```python
  schedule.every().hour.do(run_report_cycle, topic, collector, report_generator, max_articles)

- Then it enters a loop calling `schedule.run_pending()` every few seconds.

- This means a new report is generated roughly every hour, and each one is saved to `reports/`.

For convenience and easier testing, the CLI supports multiple modes:

- `--mode report` – single report generation.

- `--mode chat` – chat over the latest report.

- `--mode hourly` – hourly loop.

- `--mode demo` – generate one report and then immediately start chat.

## 3. Key GenAI Concepts and Design Decisions

### 3.1 Local LLM via LangChain + Ollama
Instead of cloud APIs, the project uses:

- LangChain as the orchestration layer,

- Ollama to run an open-source LLM (e.g. llama3) locally.

Motivations:

- Zero cost / no billing setup required.

- Works entirely on the user’s machine (good for privacy and offline demos).

- Still powerful enough to perform summarisation and simple Q&A.

This also demonstrates that the architecture is model-agnostic: swapping to a hosted model is possible by changing the LangChain LLM backend, while keeping the rest of the code unchanged.

### 3.2 Structured Summarisation with JSON

The report generation heavily relies on **structured output**:

- The prompt tells the model to output JSON with specific keys.

- This structure maps directly to the problem requirements:

    - 100–150 word summary,

    - 3–5 key_takeaways,

    - organizations_and_terms.

Benefits:

- Simpler downstream processing (no complex parsing).

- Easy to render to Markdown and reuse in the chat context.

- Easy to store and potentially consume by other systems.

### 3.3 Grounded Conversation and Prompt Engineering

The chat agent uses simple but effective **prompt engineering**:

- It makes the latest report explicit in a system message.

- It instructs the model to:

    - Only answer based on that report,

    - Handle vague questions by summarising the report,

    - Acknowledge when information isn’t available.

This meets the requirement that questions like “What’s happening nowadays?” should be answered based on the **latest collected data**, not general world knowledge.

## 4. Alternatives Considered

1. Direct HTML Scraping vs RSS

    - Alternative: Use requests + BeautifulSoup directly on BBC or GOV.UK HTML pages.

    - Pros: Access to full article text, layout, and richer metadata.

    - Cons: More brittle (HTML structure changes), more parsing code, slower to implement.

    - Decision: Use RSS via feedparser because:

        - It’s stable and standardised.

        - Already includes the key fields (title, summary, URL, date).

        - Sufficient for the summarisation task.

2. Multiple Sources vs Single Source

    - Alternative: Combine BBC with UK Government feeds (e.g. press releases).

    - Pros: Broader set of documents and viewpoints.

    - Cons: Higher implementation complexity for limited added value within the time budget.

    - Decision: Focus on BBC Technology as a clear, trusted example. The NewsCollector abstraction keeps the door open to adding other sources later.

3. Cloud LLM (OpenAI, Anthropic) vs Local LLM

    - Alternative: Use OpenAI (e.g. gpt-4o-mini) through LangChain or the official SDK.

    - Pros: Stronger models, better JSON adherence.

    - Cons: Requires API key, billing, and network access.

    - Decision: Use Ollama + local model to keep the project entirely free and easy to run in offline or restricted environments.

## 5. Conclusion

The `ai_report_agent` system:

- **Collects** topic-specific news from a trusted RSS source (BBC Technology).

- **Summarises** the collected text into a well-structured report (summary, key takeaways, organizations/terms).

- **Automates** this process on an hourly schedule using a lightweight scheduler.

- **Exposes** a conversational interface that can answer:

    - Focused questions (“What’s new in AI regulation?”),

    - Vague questions (“What’s happening nowadays?”) based purely on the latest generated report.

By using **LangChain** with a local **Ollama** model, the solution demonstrates practical GenAI patterns (summarisation, structured output, grounded Q&A) while staying cost-free, reproducible, and easy to run on a typical developer machine.