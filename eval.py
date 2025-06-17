import random
import requests
import pandas as pd
from datasets import load_dataset
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

# Load dataset
print("Loading MedQuAD...")
dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
df = pd.DataFrame(dataset["train"])
df = df.dropna(subset=["Question", "Answer"]).reset_index(drop=True)

# Scorers
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Evaluation config
NUM_QUESTIONS = 20  
base_url = "http://127.0.0.1:8000"

# Store scores
bleu_scores = []
rouge_scores = []

# Evaluate chatbot
print(f"Evaluating on {NUM_QUESTIONS} questions...\n")
for i in tqdm(random.sample(range(len(df)), NUM_QUESTIONS)):
    question = df.loc[i, "Question"]
    reference = df.loc[i, "Answer"]

    try:
        response = requests.post(f"{base_url}/symptom-check", json={"query": question}, timeout=30)
        generated = response.json()["response"]
    except Exception as e:
        print(f"Error on index {i}: {e}")
        continue

    # BLEU
    bleu = sentence_bleu([reference.split()], generated.split())
    bleu_scores.append(bleu)

    # ROUGE-L
    rouge = scorer.score(reference, generated)["rougeL"].fmeasure
    rouge_scores.append(rouge)

# Results
print("\nEvaluation complete:")
print(f"Average BLEU score:  {sum(bleu_scores) / len(bleu_scores):.4f}")
print(f"Average ROUGE-L score: {sum(rouge_scores) / len(rouge_scores):.4f}")
