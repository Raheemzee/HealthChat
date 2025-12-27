import os
import sys
from urllib.parse import quote_plus

# ---------- Python 3.13 cgi compatibility ----------
if sys.version_info >= (3, 13):
    import types
    sys.modules["cgi"] = types.ModuleType("cgi")

# ---------- Disable proxy injection ----------
for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
    os.environ.pop(k, None)

import requests
import feedparser
from flask import Flask, render_template, request, session, jsonify
from openai import OpenAI

# ---------- Flask ----------
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

# ---------- OpenAI ----------
client = OpenAI()

# ---------- APIs ----------
ARXIV_API = "http://export.arxiv.org/api/query"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# =================================================
#                 PAPER FETCHERS
# =================================================

def fetch_arxiv_papers(query, max_results=5):
    url = f"{ARXIV_API}?search_query=all:{quote_plus(query)}&start=0&max_results={max_results}"
    feed = feedparser.parse(url)

    return [{
        "title": e.title,
        "summary": e.summary,
        "link": e.link
    } for e in feed.entries]


def fetch_pubmed_papers(query, max_results=5):
    search = requests.get(
        PUBMED_API,
        params={
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results
        },
        timeout=10
    ).json()

    ids = search.get("esearchresult", {}).get("idlist", [])
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
#                 AI ANSWER
# =================================================

def answer_with_research(question):
    papers = fetch_arxiv_papers(question) + fetch_pubmed_papers(question)

    if not papers:
        return "I couldn’t find strong research evidence for this question."

    context = "\n\n".join(
        f"Title: {p['title']}\nSummary: {p['summary']}\nSource: {p['link']}"
        for p in papers
    )

    prompt = f"""
You are a medical research assistant.
Answer STRICTLY using the research evidence.
Always cite sources.

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

    return response.choices[0].message.content.strip()

# =================================================
#                     ROUTES
# =================================================

@app.route("/")
def home():
    session.setdefault("chat_history", [])
    return render_template("index.html", chat_history=session["chat_history"])


@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.json
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    answer = answer_with_research(user_input)

    session["chat_history"].append({
        "user": user_input,
        "bot": answer
    })
    session.modified = True

    return jsonify({
        "user": user_input,
        "bot": answer
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
