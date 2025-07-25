# train_model.py

import os
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Simulated dataset with job role categories
data = {
    'resume_text': [
        "Experienced Python developer with ML background",
        "Marketing specialist with strong SEO knowledge",
        "Data scientist skilled in Python, pandas, and deep learning",
        "Front-end developer experienced in HTML, CSS, JS",
        "Business analyst with project management experience",
        "Java backend developer with Spring Boot experience",
        "Cybersecurity expert with knowledge of firewalls and network protocols",
        "Machine learning engineer experienced in TensorFlow and PyTorch",
        "SEO content writer with WordPress and Google Analytics skills",
        "Full-stack developer with experience in MERN stack"
    ],
    'label': [
        "Python Developer",
        "Marketing Specialist",
        "Data Scientist",
        "Web Developer",
        "Business Analyst",
        "Java Developer",
        "Security Analyst",
        "ML Engineer",
        "Content Writer",
        "Full Stack Developer"
    ]
}

df = pd.DataFrame(data)

# Vectorize resume text
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['resume_text'])

# Encode job roles
le = LabelEncoder()
y = le.fit_transform(df['label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save all components
os.makedirs("model", exist_ok=True)

with open("model/resume_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("model/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("✅ Model, vectorizer, and encoder saved successfully.")
