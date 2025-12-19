import os
import sqlite3
import requests
import feedparser
import gradio as gr
from datetime import datetime
from PyPDF2 import PdfReader
from openai import OpenAI

# ===============================
# CONFIGURATION
# ===============================

DB_FILE = "papers.db"

ARXIV_FEEDS = [
    "http://export.arxiv.org/rss/q-bio.CV",
    "http://export.arxiv.org/rss/q-bio.ONC",
    "http://export.arxiv.org/rss/q-bio.PE"
]

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY environment variable not set")

client = OpenAI(api_key=OPENAI_API_KEY)

# ===============================
# DATABASE INITIALIZATION
# ===============================

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT UNIQUE,
    abstract TEXT,
    pdf_url TEXT,
    source TEXT,
    added_on TEXT
)
""")

conn.commit()

# ===============================
# FETCH PAPERS FROM ARXIV
# ===============================

def fetch_arxiv_papers():
    new_papers = []

    for feed_url in ARXIV_FEEDS:
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            title = entry.title.strip()
            abstract = entry.summary.strip()
            pdf_url = entry.link.replace("abs", "pdf")
            source = "arXiv"
            added_on = datetime.utcnow().isoformat()

            # Skip duplicates
            c.execute("SELECT 1 FROM papers WHERE title = ?", (title,))
            if c.fetchone():
                continue

            c.execute("""
                INSERT INTO papers (title, abstract, pdf_url, source, added_on)
                VALUES (?, ?, ?, ?, ?)
            """, (title, abstract, pdf_url, source, added_on))

            conn.commit()

            new_papers.append(f"• {title}\n  {pdf_url}")

    if not new_papers:
        return "No new papers found."

    return "\n\n".join(new_papers)

# ===============================
# PDF TEXT EXTRACTION (OPTIONAL)
# ===============================

def extract_pdf_text(pdf_url):
    try:
        r = requests.get(pdf_url, timeout=20)
        with open("temp.pdf", "wb") as f:
            f.write(r.content)

        reader = PdfReader("temp.pdf")
        text = ""
        for page in reader.pages[:5]:  # limit pages
            text += page.extract_text() or ""

        os.remove("temp.pdf")
        return text[:4000]

    except Exception:
        return ""

# ===============================
# AI ANSWERING (OPENAI)
# ===============================

def answer_question(question):
    c.execute("""
        SELECT title, abstract, pdf_url
        FROM papers
        ORDER BY added_on DESC
        LIMIT 10
    """)
    rows = c.fetchall()

    if not rows:
        return "⚠️ No research papers available yet. Please fetch papers first."

    context = ""
    for title, abstract, pdf_url in rows:
        context += f"""
Title: {title}
Abstract: {abstract}
Link: {pdf_url}

"""

    prompt = f"""
You are a medical research assistant.

RULES:
- Answer ONLY using the research context
- If evidence is limited, say so clearly
- Do NOT give medical diagnosis
- Cite paper titles in references

RESEARCH CONTEXT:
{context}

USER QUESTION:
{question}

FORMAT:
- Clear explanation
- Bullet points if useful
- References section
"""

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
        max_output_tokens=700
    )

    return response.output_text.strip()

# ===============================
# GRADIO CHAT INTERFACE
# ===============================

def chat_interface(question):
    if not question.strip():
        return "Please enter a valid health-related question."
    return answer_question(question)

with gr.Blocks(title="Health Research Chatbot") as demo:
    gr.Markdown("""
# 🩺 Health Research Chatbot

Ask health-related questions and receive **evidence-based answers**
derived from **real research papers (arXiv)**.

⚠️ **Disclaimer:**  
This tool provides research summaries only and does **not** replace
professional medical advice.
""")

    with gr.Row():
        question_input = gr.Textbox(
            label="Your Question",
            placeholder="e.g. What does research say about cancer immunotherapy?",
            scale=4
        )
        ask_btn = gr.Button("Ask", scale=1)

    answer_output = gr.Textbox(
        label="Answer",
        lines=14,
        interactive=False
    )

    ask_btn.click(
        fn=chat_interface,
        inputs=question_input,
        outputs=answer_output
    )

    gr.Markdown("## 📄 Research Papers")

    fetch_btn = gr.Button("Fetch Latest Papers from arXiv")
    fetch_output = gr.Textbox(
        label="Newly Added Papers",
        lines=10,
        interactive=False
    )

    fetch_btn.click(
        fn=fetch_arxiv_papers,
        outputs=fetch_output
    )

# ===============================
# LAUNCH
# ===============================

demo.launch(server_name="0.0.0.0", server_port=7860)
