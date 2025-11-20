# Generative AI Report Agent (LangChain + Ollama)

This project is a **Generative AI Report Agent** that:

1. **Collects news** from BBC Technology (via RSS) for a given topic (e.g. "AI", "AI regulation").
2. **Generates an hourly-ready report** using a local LLM through LangChain:
   - 100–150 word summary
   - 3–5 key takeaways
   - Mentioned organizations / key terms
3. Provides a simple **conversational CLI interface** to chat about the latest report.

It is designed to be **fully local and free**: it uses an open-source model via **Ollama**, and does not require any paid API keys.

---

## 1. Setup

### 1.1 Prerequisites

- macOS / Linux (tested on macOS)
- Python 3.9+
- [Ollama](https://ollama.com) installed and running
- A pulled local model, e.g. `llama3`

### 1.2 Install Ollama and model

1. Download and install Ollama from the website.
2. In a terminal, pull the model:

   ```bash
   ollama pull llama3
   ```
3. Quick sanity check:
    ```
    ollama run llama3 "Say hello in one sentence."
    ```
If this prints a response, Ollama is working.

## 2. Project installation
### 2.1 Clone / unzip
If using Git:

    git clone <your-repo-url> ai_report_agent
    cd ai_report_agent

If using a ZIP, unzip it and `cd' into the `ai_report_agent` folder.

### 2.2 Create virtual environment
    
    python3 -m venv .venv
    source .venv/bin/activate  
    
### 2.3 Install dependencies
    pip install -r requirements.txt
Key dependencies:

## 3. Usage

The main entry point is `app_langchain.py`.
### 3.1 One-off report generation

Generate a single report for a topic (e.g. "AI"):

    python app_langchain.py --mode report --topic "AI"


This will:

- Fetch recent BBC Technology articles mentioning "AI".

- Generate a structured report using a local LLM (LangChain + Ollama).

- Save the report into reports/.

Example:

    python app_langchain.py --mode report --topic "AI regulation"

### 3.2 Chat about the latest report

After at least one report exists in reports/:

    python app_langchain.py --mode chat


You can now ask questions such as:

- What's new in AI today?

- What’s happening nowadays?

- Which organizations are mentioned?

- What are the main risks or opportunities mentioned?

Type exit or quit to leave the chat.

### 3.3 Hourly reporting loop

To run an hourly reporting job:

    python app_langchain.py --mode hourly --topic "AI" --max-articles 5


This will:

- Periodically fetch new BBC Technology items for the given topic.

- Generate and save a new report every hour (using the schedule library).

- Append new JSON/Markdown files to reports/.

Stop with Ctrl+C.

### 3.4 Quick demo (report + chat)

The easiest way to demonstrate the system end-to-end is:

    python app_langchain.py --mode demo --topic "AI"


This will:

- Run a single report generation for the topic.

- Immediately start the chat interface based on that new report.

## 4. Project structure

Typical layout of the ai_report_agent folder:

    ai_report_agent/
    ├── app_langchain.py          # Main code: collector, report generator, chat agent
    ├── requirements.txt          # Python dependencies
    ├── README.md                 # This file
    ├── TECHNICAL_WRITEUP.md      # Short technical explanation (design + decisions)
    ├── example_outputs/          # Example outputs (for the assignment)
    │   ├── sample_report.md
    │   └── sample_chat.txt
    └── reports/                  # Generated at runtime (reports saved here)
- reports/ is created automatically when you run the app.

- example_outputs/ can contain pre-generated sample outputs for submission.

## 5. Example outputs

For the assignment, example outputs are included in example_outputs/ (or can be created like this):

1. Run:

    python app_langchain.py --mode demo --topic "AI"


2. After it generates a report, open the latest reports/report_*.md and copy its contents into:

    - example_outputs/sample_report.md

3. While in the chat loop, ask a few questions:

    - What's new in AI today?

    - What’s happening nowadays?

    - Which organizations are mentioned?

    Copy the Q&A into:

    - example_outputs/sample_chat.txt

These files show:

- The report generation output.

- The conversational interface and how it uses the latest report.

## 6. Design notes and extensions

- Data collection currently uses the BBC Technology RSS feed, filtered by a topic keyword.

- Summarisation is done by a local LLM (e.g. llama3) via LangChain’s ChatOllama.

- Hourly automation uses the lightweight schedule library.

- Conversation is grounded in the latest report, with a prompt that:

    - Encourages using only the report information,

    - Handles vague questions like “What’s happening nowadays?”.

Possible extensions:

- Add a second collector for UK Government press releases.

- Use a vector store (FAISS / Chroma) to support querying older reports.

- Replace the CLI chat with a Streamlit or Gradio web UI using the same ChatAgent logic.

