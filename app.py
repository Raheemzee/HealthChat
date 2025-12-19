# --- Python 3.13 cgi compatibility patch ---
import sys
if sys.version_info >= (3, 13):
    import types
    sys.modules["cgi"] = types.ModuleType("cgi")
# ------------------------------------------

import os
import requests
import feedparser
from flask import Flask, render_template, request
from openai import OpenAI

app = Flask(__name__)

# ================= CONFIG =================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

ARXIV_API = "http://export.arxiv.org/api/query"
PUBMED_API = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ================= PAPER FETCHERS =================
def fetch_arxiv_papers(query, max_results=5):
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results
    }
    feed = feedparser.parse(ARXIV_API, params=params)
    papers = []

    for entry in feed.entries:
        papers.append({
            "title": entry.title,
            "summary": entry.summary,
            "link": entry.link
        })
    return papers


def fetch_pubmed_papers(query, max_results=5):
    search_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results
    }
    search_resp = requests.get(PUBMED_API, params=search_params).json()
    ids = search_resp.get("esearchresult", {}).get("idlist", [])

    if not ids:
        return []

    fetch_params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "text",
        "rettype": "abstract"
    }
    abstracts = requests.get(PUBMED_FETCH, params=fetch_params).text

    return [{
        "title": f"PubMed Article {pid}",
        "summary": abstracts[:1000],
        "link": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
    } for pid in ids]


# ================= AI ANSWERING =================
def answer_with_research(question):
    arxiv_papers = fetch_arxiv_papers(question)
    pubmed_papers = fetch_pubmed_papers(question)

    context = ""
    for p in arxiv_papers + pubmed_papers:
        context += f"""
Title: {p['title']}
Summary: {p['summary']}
Source: {p['link']}

"""

    prompt = f"""
You are a medical research assistant.

Answer the health question strictly using the research context below.
If evidence is limited, say so clearly.
Always cite papers at the end.

RESEARCH CONTEXT:
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

    return response.choices[0].message.content


# ================= ROUTES =================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form.get("user_input")
    answer = answer_with_research(user_input)
    return render_template(
        "index.html",
        user_input=user_input,
        chatbot_response=answer
    )


# ================= MAIN =================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
