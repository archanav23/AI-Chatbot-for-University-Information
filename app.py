import pyodbc
import streamlit as st
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Connect to AZURE SQL DATABASE

conn = pyodbc.connect(
    "Driver={ODBC Driver 17 for SQL Server};"
    f"Server={st.secrets['DB_SERVER']};"
    f"Database={st.secrets['DB_NAME']};"
    f"Uid={st.secrets['DB_USER']};"
    f"Pwd={st.secrets['DB_PASSWORD']};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)
cursor = conn.cursor()

# 2. Fetch Questions and Answers

cursor.execute("SELECT question, answer FROM faq WHERE question IS NOT NULL AND answer IS NOT NULL")
data = cursor.fetchall()

questions = [str(row[0]).strip() for row in data]
answers = [str(row[1]).strip() for row in data]

# 3. Normalize Text

def normalize_text(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

questions_norm = [normalize_text(q) for q in questions]
questions_tokens = [set(q.split()) for q in questions_norm]

# 4. TF-IDF Vectorization

vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(questions_norm)

# 5. Chatbot Logic

def chatbot_response(user_input):
    u_norm = normalize_text(user_input)
    u_tokens = set(u_norm.split())

    strong_phrases = [
        'exam timetable','exam schedule','exam date','semester exam','exam duration',
        'admit card','hall ticket','syllabus','time table',
        'exam result','marksheet','marks','cgpa','gpa',
        'exam registration','exam fee',
        'revaluation','rechecking','backlog','supplementary',
        'degree certificate','provisional certificate','migration certificate',
        'attendance shortage','minimum attendance',
        'practical exam','lab exam','viva',
        'error in marks','discrepancy',
        'calculator','exam hall','mobile phone',
        'late for exam','missed exam',
        'online exam portal','student login',
        'duplicate mark sheet','duplicate degree',
        'convocation','graduation ceremony',
        'fee receipt','fee payment issues'
    ]

    for phrase in strong_phrases:
        if phrase in u_norm:
            for i, q_norm in enumerate(questions_norm):
                if phrase in q_norm:
                    return answers[i]

    best_overlap_idx = -1
    best_overlap = 0.0
    for i, qtoks in enumerate(questions_tokens):
        if len(u_tokens) == 0:
            overlap = 0.0
        else:
            overlap = len(u_tokens & qtoks) / len(u_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
            best_overlap_idx = i

    if best_overlap > 0.4:
        return answers[best_overlap_idx]

    user_vec = vectorizer.transform([u_norm])
    sims = cosine_similarity(user_vec, X).flatten()
    top_idx = sims.argmax()

    if sims[top_idx] > 0.25:
        return answers[top_idx]

    return "I'm not sure about that."

# 6. STREAMLIT UI

st.set_page_config(page_title="University AI Chatbot")

st.title("AI Chatbot for University Information")

user_question = st.text_input("Ask your question:")

if user_question:
    response = chatbot_response(user_question)
    st.write("**Chatbot:**", response)