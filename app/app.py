import streamlit as st
import fitz  # PyMuPDF
import pickle
import os
import re

# Load saved components
with open("model/resume_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("model/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Text cleaning function
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

# Extract text from PDF
def extract_text_from_pdf(file):
    with fitz.open(stream=file.read(), filetype="pdf") as doc:
        return "\n".join([page.get_text() for page in doc])

# Streamlit UI
st.title("📄 Resume Shortlister")
st.write("Upload your resume in PDF format to see which job roles it matches.")

uploaded_file = st.file_uploader("Upload a PDF resume", type=["pdf"])

if uploaded_file:
    resume_text = extract_text_from_pdf(uploaded_file)
    st.subheader("Extracted Resume Text:")
    st.text_area("Resume Preview", resume_text, height=200)

    cleaned = clean_text(resume_text)
    vector = vectorizer.transform([cleaned])
    probs = model.predict_proba(vector)[0]

    # Decode class labels
    categories = label_encoder.inverse_transform(range(len(probs)))

    # Show categories above a threshold
    threshold = 0.15
    eligible = [(cat, prob) for cat, prob in zip(categories, probs) if prob >= threshold]
    eligible.sort(key=lambda x: x[1], reverse=True)

    if eligible:
        st.success("🎯 You may be eligible for:")
        for cat, prob in eligible:
            st.write(f"- {cat} ({prob:.2%} confidence)")
    else:
        st.warning("No strong job category match found.")
