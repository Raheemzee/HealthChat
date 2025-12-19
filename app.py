import os
import sys
from urllib.parse import quote_plus

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

# ---------- Flask ----------
app = Flask(__name__)

# ---------- OpenAI ----------
# API key is read automatically from environment variable OPENAI_API_KEY
client = OpenAI()

# ---------- External APIs ----------
ARXIV_API = "http://export.arxiv.org/api/query"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# =================================================
#                 PAPER FETCHERS
# =================================================

def fetch_arxiv_papers(query, max_results=5):
    """
    Fetch research papers from arXiv safely (URL-encoded)
    """
    safe_query = quote_plus(query)

    url = (
        f"{ARXIV_API}"
        f"?search_query=all:{safe_query}"
        f"&start=0&max_results={max_results}"
    )

    feed = feedparser.parse(url)

    papers = []
    for entry in feed.entries:
        papers.append({
            "title": entry.title,
            "summary": entry.summary,
            "link": entry.link
        })

    return papers


def fetch_pubmed_papers(query, max_results=5):
    """
    Fetch research papers from PubMed
    """
    search_resp = requests.get(
        PUBMED_API,
        params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results
        },
        timeout=10
    ).json()

    ids = search_resp.get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    abstracts = requests.get(
        PUBMED_FETCH,
        params={
            "db": "pubmed",
            "id": ",".join(ids),
            "retmode": "text",
            "rettype": "abstract"
        },
        timeout=10
    ).text

    return [{
        "title": f"PubMed Article {pid}",
        "summary": abstracts[:1200],
        "link": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
    } for pid in ids]

# =================================================
#                 AI ANSWERING
# =================================================

def answer_with_research(question):
    """
    Build research context and ask OpenAI
    """
    papers = fetch_arxiv_papers(question) + fetch_pubmed_papers(question)

    if not papers:
        return (
            "I couldn't find sufficient research evidence for this question. "
            "Please consult a qualified medical professional."
        )

    context = "\n\n".join(
        f"Title: {p['title']}\n"
        f"Summary: {p['summary']}\n"
        f"Source: {p['link']}"
        for p in papers
    )

    prompt = f"""
You are a medical research assistant.

Answer the health question STRICTLY using the research evidence below.
If evidence is limited or inconclusive, state it clearly.
DO NOT give personal medical advice.
Always cite sources at the end.

RESEARCH EVIDENCE:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=600
    )

    return response.choices[0].message.content.strip()

# =================================================
#                     ROUTES
# =================================================

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form.get("user_input", "").strip()

    if not user_input:
        return render_template(
            "index.html",
            chatbot_response="Please enter a health-related question."
        )

    try:
        answer = answer_with_research(user_input)
    except Exception as e:
        print("ERROR:", e)
        answer = "An internal error occurred while processing your request."

    return render_template(
        "index.html",
        user_input=user_input,
        chatbot_response=answer
    )

# =================================================
#                     MAIN
# =================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
