import os
import tempfile
import subprocess
import sqlite3
import requests
import feedparser
import pandas as pd
from datetime import datetime

from flask import Flask, render_template, request, jsonify, send_file

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from gtts import gTTS
from faster_whisper import WhisperModel

# =============================
# App Setup
# =============================

app = Flask(__name__)
DB_FILE = "papers.db"

ARXIV_FEEDS = [
    "http://export.arxiv.org/rss/q-bio.CV",
    "http://export.arxiv.org/rss/q-bio.ONC",
]

# =============================
# Database
# =============================

conn = sqlite3.connect(DB_FILE, check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT UNIQUE,
    abstract TEXT,
    pdf_url TEXT,
    added_on TEXT
)
""")
conn.commit()

# =============================
# Fetch arXiv Papers
# =============================

def fetch_arxiv_papers():
    added = 0

    for feed_url in ARXIV_FEEDS:
        feed = feedparser.parse(feed_url)

        for entry in feed.entries:
            title = entry.title.strip()
            abstract = entry.summary.strip()
            pdf_url = entry.link.replace("abs", "pdf")
            added_on = datetime.utcnow().isoformat()

            c.execute("SELECT id FROM papers WHERE title=?", (title,))
            if c.fetchone():
                continue

            c.execute("""
                INSERT INTO papers (title, abstract, pdf_url, added_on)
                VALUES (?, ?, ?, ?)
            """, (title, abstract, pdf_url, added_on))
            conn.commit()
            added += 1

    return added

# =============================
# Load Papers into TF-IDF
# =============================

def load_vector_index():
    df = pd.read_sql("SELECT * FROM papers", conn)

    if df.empty:
        return None, None, None

    documents = (
        df["title"] + ". " + df["abstract"]
    ).tolist()

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    vectors = vectorizer.fit_transform(documents)

    return df, vectorizer, vectors

# =============================
# Research-Based Answering
# =============================

def get_best_answer(question):
    df, vectorizer, vectors = load_vector_index()

    if df is None:
        return "No research papers available yet. Please fetch papers first."

    user_vec = vectorizer.transform([question])
    sims = cosine_similarity(user_vec, vectors).flatten()

    top_idx = sims.argmax()
    score = sims[top_idx]

    if score < 0.25:
        return (
            "I could not find strong evidence in the current research papers. "
            "Please consult a medical professional."
        )

    paper = df.iloc[top_idx]

    return (
        f"📄 **Based on research findings:**\n\n"
        f"{paper['abstract']}\n\n"
        f"🔗 **Source:** {paper['title']}\n{paper['pdf_url']}"
    )

# =============================
# Text-to-Speech
# =============================

def text_to_speech(text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    gTTS(text).save(tmp.name)
    return tmp.name

# =============================
# Whisper STT
# =============================

whisper_model = WhisperModel("tiny", device="cpu")

@app.route("/stt", methods=["POST"])
def stt():
    if "audio" not in request.files:
        return jsonify({"error": "No audio uploaded"}), 400

    audio = request.files["audio"]
    tmp_webm = tempfile.NamedTemporaryFile(delete=False, suffix=".webm").name
    audio.save(tmp_webm)

    tmp_wav = tmp_webm.replace(".webm", ".wav")

    subprocess.run([
        "ffmpeg", "-i", tmp_webm,
        "-ar", "16000", "-ac", "1",
        tmp_wav, "-y", "-loglevel", "quiet"
    ], check=True)

    segments, _ = whisper_model.transcribe(tmp_wav)
    transcript = " ".join(seg.text for seg in segments).strip()

    response = get_best_answer(transcript) if transcript else "I didn’t catch that."

    return jsonify({
        "transcript": transcript,
        "response": response
    })

# =============================
# Routes
# =============================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.form["user_input"]
    response = get_best_answer(user_input)
    return render_template(
        "index.html",
        user_input=user_input,
        chatbot_response=response
    )

@app.route("/speak_response", methods=["POST"])
def speak_response():
    text = request.get_json().get("text", "")
    audio_path = text_to_speech(text)
    return send_file(audio_path, mimetype="audio/mpeg")

@app.route("/fetch_papers", methods=["POST"])
def fetch_papers():
    count = fetch_arxiv_papers()
    return jsonify({"added": count})

# =============================
# Run
# =============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
