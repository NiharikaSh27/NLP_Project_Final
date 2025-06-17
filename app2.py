import re
import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
from transformers import pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import requests
from evaluate import load


# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')

# Create FastAPI app
app = FastAPI(title=" HealthCare Assistant")

# Device setup
device = "mps" if torch.backends.mps.is_available() else "cpu"

# --- 1. Text Cleaning ---
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)

# --- 2. Load and Prepare Data ---
print("Loading MedQuad dataset...")
medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset")
medquad_df = pd.DataFrame(medquad['train'])
medquad_df['Answer'] = medquad_df['Answer'].apply(clean_text)
docs = medquad_df['Answer'].dropna().tolist()

# --- 3. Build FAISS Index ---
print("Building FAISS index...")
embedding_model = SentenceTransformer(
    'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb',
    device=device
)
doc_embeddings = embedding_model.encode(docs, show_progress_bar=True)
index = faiss.IndexFlatL2(doc_embeddings.shape[1])
index.add(np.array(doc_embeddings))

def retrieve_context(query, k=3):
    query_vec = embedding_model.encode([clean_text(query)])
    D, I = index.search(np.array(query_vec), k=k)
    return "\n".join([docs[i] for i in I[0]])

# --- 4. Emotion Classifier ---
print("Loading emotion detection model...")
emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=0 if torch.cuda.is_available() else -1
)

emotion_intro = {
    "sadness": "I'm here for you. I understand this might be worrying.",
    "fear": "Don't worry, I'm here to explain things calmly.",
    "joy": "That's great to hear! Let's keep that positivity going.",
    "anger": "I understand you're upset. Let's work through this together.",
    "neutral": "Happy to help you with this.",
}

def detect_emotion(text):
    result = emotion_model(clean_text(text))[0]
    return result['label']

# --- 5. LLaMA API Query ---
def query_llama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3", "prompt": prompt, "stream": False}
        )
        return response.json()['response'].strip()
    except Exception as e:
        return f"Error: Cannot connect to LLaMA API - {str(e)}"
# --- 8. Evaluation Metrics ---
rouge = load("rouge")
bleu = load("bleu")

def evaluate_response(prediction, reference):
    rouge_score = rouge.compute(predictions=[prediction], references=[reference])
    bleu_score = bleu.compute(predictions=[prediction.split()], references=[[reference.split()]])
    return {"rouge": rouge_score, "bleu": bleu_score}

# --- 6. Pydantic Schema ---
class UserQuery(BaseModel):
    query: str

# --- 7. API Endpoints ---
@app.post("/symptom-check")
def symptom_check(data: UserQuery):
    context = retrieve_context(data.query)
    emotion = detect_emotion(data.query)
    prompt = f"""You are HealthCare, a virtual health assistant.

Based on the symptoms described, provide possible causes and suggest when to seek medical help, using only the provided context. Start with an empathetic response based on the user's emotional state.

### EMOTION:
{emotion_intro[emotion]}

### CONTEXT:
{context}

### SYMPTOMS:
{data.query}

### RESPONSE:"""
    return {"response": query_llama(prompt)}

@app.post("/explain-prescription")
def explain_prescription(data: UserQuery):
    context = retrieve_context(data.query)
    prompt = f"""You are HealthCare Assistant, a virtual pharmacist assistant.

Explain the dosage, common side effects, and important interactions of the drug or prescription term given below, using only the provided context.

### CONTEXT:
{context}

### DRUG NAME / INSTRUCTION:
{data.query}

### RESPONSE:"""
    return {"response": query_llama(prompt)}

@app.post("/health-literacy")
def health_literacy(data: UserQuery):
    context = retrieve_context(data.query)
    prompt = f"""You are HealthCare Assistant, a patient-friendly health tutor.

Explain the concept in very simple, easy-to-understand language suitable for someone with no medical background.

### CONTEXT:
{context}

### QUESTION:
{data.query}

### RESPONSE:"""
    return {"response": query_llama(prompt)}

@app.post("/evaluate-response")
def evaluate_response_api(data: dict):
    prediction = data.get("prediction", "")
    reference = data.get("reference", "")
    return evaluate_response(prediction, reference)
