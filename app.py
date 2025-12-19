import os
import sys

# ---------- Python 3.13 cgi compatibility ----------
if sys.version_info >= (3, 13):
    import types
    sys.modules["cgi"] = types.ModuleType("cgi")

# ---------- Disable Render proxy injection ----------
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)

import requests
import feedparser
from flask import Flask, render_template, request
from openai import OpenAI

app = Flask(__name__)

# ---------- OpenAI ----------
client = OpenAI()  # API key read from env automatically

# ---------- APIs ----------
ARXIV_API = "http://export.arxiv.org/api/query"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ---------- Paper Fetchers ----------
def fetch_arxiv_papers(query, max_results=5):
    feed = feedparser.parse(
        f"{ARXIV_API}?search_query=all:{query}&start=0&max_results={max_results}"
    )

    return [{
        "title": e.title,
        "summary": e.summary,
        "link": e.link
    } for e in feed.entries]

def fetch_pubmed_papers(query, max_results=5):
    search = requests.get(PUBMED_API, params={
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }).json()

    ids = search.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    abstracts = requests.get(PUBMED_FETCH, params={
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "text",
        "rettype": "abstract"
    }).text

    return [{
        "title": f"PubMed Article {pid}",
        "summary": abstracts[:1200],
        "link": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
    } for pid in ids]

# ---------- AI Answer ----------
def answer_with_research(question):
    papers = fetch_arxiv_papers(question) + fetch_pubmed_papers(question)

    context = "\n\n".join(
        f"Title: {p['title']}\nSummary: {p['summary']}\nSource: {p['link']}"
        for p in papers
    )

    prompt = f"""
You are a medical research assistant.
Answer ONLY using the research below.
Cite sources.

RESEARCH:
{context}

QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600
    )

    return response.choices[0].message.content

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    q = request.form.get("user_input")
    a = answer_with_research(q)
    return render_template("index.html", user_input=q, chatbot_response=a)

# ---------- Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
