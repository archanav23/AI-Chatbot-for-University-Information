import pandas as pd
import streamlit as st
import re
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

# 1. Load Data

@st.cache_data
def load_data():
    df = pd.read_csv('Chatbot Data.csv')
    questions = [str(q).strip() for q in df['question'].tolist()]
    answers   = [str(a).strip() for a in df['answer'].tolist()]
    return questions, answers

# 2. Load Best Semantic Model

@st.cache_resource
def load_model():
    # all-mpnet-base-v2 is the highest accuracy
    # sentence-transformer model available
    return SentenceTransformer('all-mpnet-base-v2')

# 3. Pre-compute Embeddings for Questions + Answers

@st.cache_resource
def get_embeddings(_model, questions, answers):
    q_embeddings = _model.encode(questions, convert_to_tensor=True)
    a_embeddings = _model.encode(answers,   convert_to_tensor=True)
    return q_embeddings, a_embeddings

# 4. Clean Text

def clean(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


# 5. Keyword Boost

def keyword_boost(user_input: str, questions: list) -> np.ndarray:
    user_words  = set(clean(user_input).split())
    boost       = np.zeros(len(questions))
    stopwords   = {'what','when','where','how','why','who','is',
                   'the','a','an','i','my','can','do','did','are',
                   'in','of','for','to','get','me','will','be','it'}
    # Only keep meaningful words
    user_words -= stopwords

    for i, q in enumerate(questions):
        q_words = set(clean(q).split())
        overlap = user_words & q_words
        if overlap:
            # Boost proportional to overlap
            boost[i] = len(overlap) / max(len(user_words), 1) * 0.15

    return boost

# 6. Main Response Function

def chatbot_response(user_input, questions, answers, model, q_embeddings, a_embeddings):
    if not user_input or len(user_input.strip()) < 3:
        return "Please ask a complete question."

    # Encode user question
    user_embedding = model.encode(user_input, convert_to_tensor=True)

    # Semantic similarity against questions (primary)
    q_scores = util.cos_sim(user_embedding, q_embeddings)[0].cpu().numpy()

    # Semantic similarity against answers (secondary)
    a_scores = util.cos_sim(user_embedding, a_embeddings)[0].cpu().numpy()

    # Keyword boost
    boost = keyword_boost(user_input, questions)

    # Final score:
    # 70% question match + 20% answer match + 10% keyword boost
    final_scores = (0.70 * q_scores) + (0.20 * a_scores) + boost

    # Get top 3 candidates
    top3_idx = np.argsort(final_scores)[::-1][:3]
    best_idx  = top3_idx[0]
    best_score = final_scores[best_idx]

    # Dynamic threshold — if best score is decent, return it
    THRESHOLD = 0.25

    if best_score >= THRESHOLD:
        return answers[best_idx]

    # Last resort — if semantic fails, return closest question's answer anyway
    # (because user is clearly asking about university topics)
    if q_scores[best_idx] > 0.15:
        return answers[best_idx]

    return (
        "Sorry, I could not find relevant information for your question. "
        "Please rephrase or contact the university administration."
    )

# 7. Streamlit UI

st.set_page_config(page_title="University AI Chatbot", layout="centered")
st.title("🎓 University Information Chatbot")
st.write("Ask anything about exams, results, attendance, fees, and more!")

# Load everything
questions, answers       = load_data()
model                    = load_model()
q_embeddings, a_embeddings = get_embeddings(model, questions, answers)

# Chat input
user_question = st.text_input("Your Question:", placeholder="e.g. When will my results come?")

if user_question:
    with st.spinner("Finding best answer..."):
        response = chatbot_response(
            user_question, questions, answers,
            model, q_embeddings, a_embeddings
        )
    st.markdown(f"**🤖 Chatbot:** {response}")

# Sample questions
st.markdown("---")
st.markdown("💡 **What are your queries?**")
samples = [
    "Tell me about exam dates",
    "How do I get my admit card?",
    "What score do I need to pass?",
    "My marksheet has wrong marks",
    "How many days to get duplicate marksheet?",
]
for s in samples:
    st.write(f"• {s}")
