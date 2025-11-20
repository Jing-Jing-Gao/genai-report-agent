import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import feedparser
from bs4 import BeautifulSoup
import schedule

from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


# -------------------------
# Data model
# -------------------------

@dataclass
class Article:
    source: str
    url: str
    title: str
    published_at: Optional[str]
    text: str


# -------------------------
# News collector (BBC Technology RSS)
# -------------------------

class NewsCollector:
    """
    Collects news articles from BBC Technology RSS and filters them by topic keyword.
    """

    def __init__(self, feed_url: str, source_name: str = "bbc_technology"):
        self.feed_url = feed_url
        self.source_name = source_name

    def fetch_articles(self, topic: str, max_articles: int = 10) -> List[Article]:
        topic_lower = topic.lower()
        feed = feedparser.parse(self.feed_url)

        articles: List[Article] = []

        for entry in feed.entries:
            title = getattr(entry, "title", "")
            summary_html = getattr(entry, "summary", "")
            url = getattr(entry, "link", "")

            # Filter by topic keyword appearing in title or summary
            if topic_lower not in title.lower() and topic_lower not in summary_html.lower():
                continue

            summary_text = self._clean_html(summary_html)

            published_at = None
            published_parsed = getattr(entry, "published_parsed", None)
            if published_parsed:
                try:
                    dt = datetime(
                        year=published_parsed.tm_year,
                        month=published_parsed.tm_mon,
                        day=published_parsed.tm_mday,
                        hour=published_parsed.tm_hour,
                        minute=published_parsed.tm_min,
                        second=published_parsed.tm_sec,
                        tzinfo=timezone.utc,
                    )
                    published_at = dt.isoformat()
                except Exception:
                    published_at = None

            text = f"{title}\n\n{summary_text}"

            articles.append(
                Article(
                    source=self.source_name,
                    url=url,
                    title=title,
                    published_at=published_at,
                    text=text,
                )
            )

            if len(articles) >= max_articles:
                break

        return articles

    @staticmethod
    def _clean_html(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(" ", strip=True)


# -------------------------
# Report generator (LangChain + Ollama)
# -------------------------

class ReportGenerator:
    """
    Uses a LangChain ChatOllama model to create a structured report
    from collected articles and saves it.
    """

    def __init__(self, llm: ChatOllama, reports_dir: Path):
        self.llm = llm
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(self, topic: str, articles: List[Article]) -> dict:
        if not articles:
            raise ValueError("No articles available to generate a report.")

        corpus = self._build_corpus(articles)

        system_prompt = (
            "You are an AI assistant that writes concise news reports.\n"
            "You will receive a corpus of recent news items about a specific topic.\n"
            "Your job is to produce a short report with:\n"
            "1) A 100–150 word summary paragraph.\n"
            "2) 3–5 concise key takeaways.\n"
            "3) A list of mentioned organizations, entities, or important terms.\n"
            "Only use the information in the corpus. Do not invent facts.\n"
            "Output JSON with keys: 'summary', 'key_takeaways', 'organizations_and_terms'."
        )

        user_prompt = (
            f"Topic: {topic}\n\n"
            f"News corpus (separated by ---):\n\n{corpus}\n\n"
            "Remember: output only valid JSON."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        content = response.content if hasattr(response, "content") else str(response)

        try:
            data = json.loads(content)
            summary = data.get("summary", "").strip()
            key_takeaways = data.get("key_takeaways", [])
            organizations = data.get("organizations_and_terms", [])
        except json.JSONDecodeError:
            # Fallback: make up a simple structure from the raw text
            summary = content.strip()[:600]
            key_takeaways = []
            organizations = []

        generated_at = datetime.now(timezone.utc).isoformat()

        report = {
            "generated_at": generated_at,
            "topic": topic,
            "article_count": len(articles),
            "summary": summary,
            "key_takeaways": key_takeaways,
            "organizations_and_terms": organizations,
            "articles": [asdict(a) for a in articles],
        }

        self._save_report_files(report)
        return report

    @staticmethod
    def _build_corpus(articles: List[Article], max_chars: int = 8000) -> str:
        parts = []
        for a in articles:
            parts.append(
                f"Title: {a.title}\nURL: {a.url}\n"
                f"Published: {a.published_at}\n\n{a.text}"
            )
        corpus = "\n\n---\n\n".join(parts)
        if len(corpus) > max_chars:
            corpus = corpus[:max_chars]
        return corpus

    def _save_report_files(self, report: dict) -> None:
        timestamp = report["generated_at"].replace(":", "-")
        json_path = self.reports_dir / f"report_{timestamp}.json"
        md_path = self.reports_dir / f"report_{timestamp}.md"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        with md_path.open("w", encoding="utf-8") as f:
            f.write(self._build_markdown(report))

        print(f"Saved report: {json_path}")
        print(f"Saved human-readable report: {md_path}")

    @staticmethod
    def _build_markdown(report: dict) -> str:
        lines = []
        lines.append(f"# Topic: {report['topic']}")
        lines.append("")
        lines.append(f"Generated at: {report['generated_at']}")
        lines.append(f"Articles: {report['article_count']}")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(report["summary"])
        lines.append("")
        lines.append("## Key takeaways")
        lines.append("")
        for item in report["key_takeaways"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## Organizations / Terms")
        lines.append("")
        for item in report["organizations_and_terms"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("## Articles")
        lines.append("")
        for a in report["articles"]:
            lines.append(f"- {a['title']} ({a['source']}) - {a['url']}")
        lines.append("")
        return "\n".join(lines)


# -------------------------
# Chat agent (LangChain + Ollama)
# -------------------------

class ChatAgent:
    """
    Simple conversational interface grounded in the latest report.
    Uses LangChain's ChatOllama model.
    """

    def __init__(self, llm: ChatOllama, report: dict):
        self.llm = llm
        self.report = report
        self.history: List = []  # list of HumanMessage / AIMessage

    def _build_context(self) -> str:
        lines = []
        lines.append(f"Topic: {self.report['topic']}")
        lines.append(f"Generated at: {self.report['generated_at']}")
        lines.append("")
        lines.append("Summary:")
        lines.append(self.report["summary"])
        lines.append("")
        lines.append("Key takeaways:")
        for item in self.report["key_takeaways"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("Organizations / Terms:")
        for item in self.report["organizations_and_terms"]:
            lines.append(f"- {item}")
        return "\n".join(lines)

    def ask(self, question: str) -> str:
        system_prompt = (
            "You are a helpful AI news assistant.\n"
            "You answer questions about recent updates on a specific topic using "
            "ONLY the information in the provided report.\n"
            "If the user asks a vague question like 'What's happening nowadays?' "
            "or 'Any news?', summarise the most important points from the report.\n"
            "If the user asks about something not covered by the report, say you "
            "don't know and gently steer them back to the topic."
        )

        context = self._build_context()

        messages = [
            SystemMessage(content=system_prompt),
            SystemMessage(content=f"Here is the latest report:\n\n{context}"),
        ] + self.history + [
            HumanMessage(content=question),
        ]

        response = self.llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))
        return answer

    def chat_loop(self) -> None:
        print("\nChatting about the latest report.")
        print("Type 'exit' or 'quit' to leave.\n")

        while True:
            try:
                question = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting chat.")
                break

            if question.lower() in {"exit", "quit"}:
                print("Goodbye.")
                break

            if not question:
                continue

            answer = self.ask(question)
            print(f"Agent: {answer}\n")


# -------------------------
# Utility: load latest report
# -------------------------

def load_latest_report(reports_dir: Path) -> Optional[dict]:
    if not reports_dir.exists():
        return None

    json_files = sorted(
        reports_dir.glob("report_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not json_files:
        return None

    latest = json_files[0]
    with latest.open("r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Scheduling helpers
# -------------------------

def run_report_cycle(
    topic: str,
    collector: NewsCollector,
    report_generator: ReportGenerator,
    max_articles: int,
) -> None:
    print(f"\n[{datetime.now().isoformat()}] Running report cycle for topic '{topic}'")

    articles = collector.fetch_articles(topic=topic, max_articles=max_articles)
    if not articles:
        print("No matching articles found; report not generated.")
        return

    report = report_generator.generate_report(topic, articles)
    print(f"Report generated at {report['generated_at']}")


def start_hourly_loop(
    topic: str,
    collector: NewsCollector,
    report_generator: ReportGenerator,
    max_articles: int,
) -> None:
    schedule.every().hour.do(run_report_cycle, topic, collector, report_generator, max_articles)

    print("Started hourly reporting loop (every 1 hour). Press Ctrl+C to stop.")

    try:
        while True:
            schedule.run_pending()
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nStopped hourly scheduler.")


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generative AI Report Agent - BBC Technology (LangChain + Ollama)"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=os.getenv("TOPIC", "AI"),
        help="Topic keyword to monitor (default: 'AI')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["report", "chat", "hourly", "demo"],
        default="demo",
        help=(
            "Mode:\n"
            "  report  - run a single report generation\n"
            "  chat    - start chat using the latest report\n"
            "  hourly  - run report generation every hour\n"
            "  demo    - generate one report and then start chat (default)"
        ),
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=5,
        help="Maximum number of articles to use per report",
    )
    parser.add_argument(
        "--ollama-model",
        type=str,
        default=os.getenv("OLLAMA_MODEL", "llama3"),
        help="Ollama model name to use (default: 'llama3')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Configuration
    reports_dir = Path("reports")
    feed_url = os.getenv(
        "BBC_TECH_FEED_URL",
        "https://feeds.bbci.co.uk/news/technology/rss.xml",
    )

    # LangChain + Ollama LLM
    llm = ChatOllama(model=args.ollama_model)

    collector = NewsCollector(feed_url=feed_url)
    report_generator = ReportGenerator(llm=llm, reports_dir=reports_dir)

    if args.mode == "report":
        run_report_cycle(args.topic, collector, report_generator, args.max_articles)

    elif args.mode == "chat":
        report = load_latest_report(reports_dir)
        if not report:
            print("No reports found. Run with --mode report first.")
            return
        agent = ChatAgent(llm=llm, report=report)
        agent.chat_loop()

    elif args.mode == "hourly":
        start_hourly_loop(args.topic, collector, report_generator, args.max_articles)

    elif args.mode == "demo":
        # One-shot report then chat
        run_report_cycle(args.topic, collector, report_generator, args.max_articles)
        report = load_latest_report(reports_dir)
        if not report:
            print("Demo: report generation appears to have failed; no report to chat about.")
            return
        agent = ChatAgent(llm=llm, report=report)
        agent.chat_loop()


if __name__ == "__main__":
    main()
