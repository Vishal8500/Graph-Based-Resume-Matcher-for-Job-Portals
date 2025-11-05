import pandas as pd
import json
import traceback
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import time
import google.generativeai as genai

# -----------------------------------------------------
# STEP 0: Configure Gemini API Key
# -----------------------------------------------------
# IMPORTANT: Set your API key.
# Best practice is to use an environment variable.
API_KEY = os.environ.get("GEMINI_API_KEY")

# If not using an environment variable, uncomment and paste your key here:
# API_KEY = "YOUR_API_KEY_HERE"

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it or paste your key in the script.")

genai.configure(api_key=API_KEY)


# -----------------------------------------------------
# STEP 1: Load Ground Truth Ontology
# -----------------------------------------------------
print("Loading ground truth 'synthetic skill ontology.csv'...")
try:
    ground_truth = pd.read_csv("synthetic skill ontology.csv")
    # Convert related skills to list form
    ground_truth["Related Skills"] = ground_truth["Related Skills"].apply(
        lambda x: [s.strip().capitalize() for s in str(x).split(",")]
    )
    print("✅ Ground truth loaded.")
except FileNotFoundError:
    print("❌ ERROR: 'synthetic skill ontology.csv' not found.")
    print("Please make sure the file is in the same directory as this script.")
    exit()


# -----------------------------------------------------
# STEP 2: Normalization Helper
# -----------------------------------------------------
def normalize(skill):
    """
    Normalize skill names for fair comparison.
    """
    return re.sub(r'[^a-zA-Z0-9 ]+', '', skill.strip().lower())

# -----------------------------------------------------
# STEP 3: Semantic Similarity Setup
# -----------------------------------------------------
print("Loading SentenceTransformer model 'all-MiniLM-L6-v2'...")
# Optional — helps catch near-synonyms like "ML" ≈ "Machine Learning"
model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ SentenceTransformer model loaded.")

def semantic_match(true_skills, predicted_skills, threshold=0.7):
    """
    Count true positives based on semantic similarity above threshold.
    """
    tp = 0
    matched_pred = set()
    
    # Pre-encode all skills to avoid redundant encoding
    true_embeddings = model.encode(list(true_skills))
    pred_embeddings = model.encode(list(predicted_skills))
    
    # Compute cosine similarity matrix
    if true_embeddings.shape[0] == 0 or pred_embeddings.shape[0] == 0:
        return 0
        
    cos_sim_matrix = util.cos_sim(true_embeddings, pred_embeddings)

    for i in range(len(true_skills)):
        for j in range(len(predicted_skills)):
            if j in matched_pred:
                continue
            
            sim = cos_sim_matrix[i][j].item()
            
            if sim >= threshold:
                tp += 1
                matched_pred.add(j)
                break # Move to the next true skill
    return tp

# -----------------------------------------------------
# STEP 4: Gemini Ontology Prompt
# -----------------------------------------------------
def build_ontology_prompt(skill_name: str) -> str:
    """
    Build the prompt for ontology expansion.
    """
    return f"""
    Act as a technical HR ontology builder for resume parsing and job matching.
    For the *single* skill "{skill_name}", find its most important related skills and parent categories.
    Return a JSON object with a list of relations (max 5).
    Each relation must include:
    - "from": "{skill_name}"
    - "to": a valid, recognized technical skill or category (e.g., "Web Framework", "Containerization").
    - "relation_type": "IS_A" or "RELATED_TO".
    - "confidence": a float between 0.0 and 1.0.

    Example for input "Flask":
    {{
        "relations": [
            {{"from": "Flask", "to": "Web Framework", "relation_type": "IS_A", "confidence": 0.95}},
            {{"from": "Flask", "to": "Python", "relation_type": "RELATED_TO", "confidence": 0.9}},
            {{"from": "Flask", "to": "Django", "relation_type": "RELATED_TO", "confidence": 0.8}}
        ]
    }}

    Focus on practical, resume-relevant skills and avoid overly generic links.
    Return only JSON.
    """

# -----------------------------------------------------
# STEP 5: Fetch Relations via Gemini
# -----------------------------------------------------
def _get_relations_for_single_skill(skill_name: str):
    """
    Fetch related skills from Gemini using the structured prompt.
    """
    prompt = build_ontology_prompt(skill_name)
    raw_text = ""
    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash",  # <-- CORRECTED model name
            generation_config={"response_mime_type": "application/json"}
        )
        print(f"Sending prompt to Gemini for '{skill_name}'...")
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        data = json.loads(raw_text)
        relations = data.get("relations", [])
        print(f"✅ Received {len(relations)} relations for '{skill_name}'.")
        return relations

    except json.JSONDecodeError as json_err:
        print(f"❌ JSON Parsing Error for skill '{skill_name}': {json_err}")
        print(f"   Raw text: {raw_text[:300]}")
        return []
    except Exception as e:
        print(f"❌ Gemini API Error for '{skill_name}': {type(e).__name__} - {e}")
        traceback.print_exc(limit=1)
        return []

# -----------------------------------------------------
# STEP 6: Evaluation Function
# -----------------------------------------------------
def evaluate_skill(skill_name, predicted_relations, ground_truth_df, semantic=True, threshold=0.7):
    """
    Compare system-generated relations with ground truth relations for one skill.
    Supports both exact and semantic matching.
    """
    try:
        true_skills_raw = ground_truth_df.loc[
            ground_truth_df["Skill"].str.lower() == skill_name.lower(),
            "Related Skills"
        ].values[0]
        
        true_skills = set(map(normalize, true_skills_raw))
        
    except IndexError:
        print(f"⚠️ Warning: Skill '{skill_name}' not found in ground truth. Skipping.")
        return None # Return None to filter out later

    # Normalize predicted
    predicted_skills_raw = [rel["to"] for rel in predicted_relations]
    predicted_skills = set(map(normalize, predicted_skills_raw))
    
    # Remove self-references from comparison
    norm_skill_name = normalize(skill_name)
    true_skills.discard(norm_skill_name)
    predicted_skills.discard(norm_skill_name)

    # Count true positives
    if semantic:
        tp = semantic_match(true_skills, predicted_skills, threshold)
    else:
        # Exact match
        tp = len(true_skills & predicted_skills)

    fp = len(predicted_skills - true_skills)
    fn = len(true_skills - predicted_skills)
    
    # Semantic match can sometimes over-count. 
    # TP should not be greater than the number of items in either set.
    tp = min(tp, len(true_skills), len(predicted_skills))
    
    # Recalculate FP and FN based on semantic TP
    if semantic:
        fp = len(predicted_skills) - tp
        fn = len(true_skills) - tp


    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    
    # Accuracy is not a great metric for this imbalanced task, but included
    # (Jaccard Index might be better)
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

    return {
        "Skill": skill_name, "TP": tp, "FP": fp, "FN": fn,
        "Precision": precision, "Recall": recall,
        "F1": f1, "Accuracy (Jaccard)": accuracy
    }

# -----------------------------------------------------
# STEP 7: Skills to Evaluate
# -----------------------------------------------------
# Per your request, here are 10 skills to iterate over.
skills_to_evaluate = [
    "Python",
    "Machine Learning",
    "Tensorflow",
    "FastAPI",
    "Docker",
    "React",
    "MongoDB",
    "CI/CD",
    "Linux",
    "Git"
]
print(f"\nStarting evaluation for {len(skills_to_evaluate)} skills...")


# -----------------------------------------------------
# STEP 8: Evaluate Ontology Generation per Skill
# -----------------------------------------------------
predicted_results = {}
# This loop runs 10 times (once for each skill)
for skill in skills_to_evaluate:
    relations = _get_relations_for_single_skill(skill)
    predicted_results[skill] = relations
    print("--- Waiting 1s to avoid rate limits ---")
    time.sleep(1) # <-- Added to avoid API rate limits

# Evaluate each skill
metrics = []
for skill, preds in predicted_results.items():
    result = evaluate_skill(skill, preds, ground_truth, semantic=True, threshold=0.7)
    if result: # Only append if skill was found in ground truth
        metrics.append(result)

if not metrics:
    print("\n❌ No valid metrics were calculated. Did any skills match the ground truth?")
    exit()

metrics_df = pd.DataFrame(metrics)

# -----------------------------------------------------
# STEP 9: Aggregated Final Metrics
# -----------------------------------------------------
overall = {
    "Precision": metrics_df["Precision"].mean(),
    "Recall": metrics_df["Recall"].mean(),
    "F1": metrics_df["F1"].mean(),
    "Accuracy (Jaccard)": metrics_df["Accuracy (Jaccard)"].mean(),
}

print("\n" + "="*40)
print("🔹 FINAL AGGREGATED METRICS (10 SKILLS) 🔹")
print("="*40)
for k, v in overall.items():
    print(f"{k}: {v:.3f}")

print("\n🔹 Per-Skill Breakdown:")
print(metrics_df.to_string(index=False, float_format="%.3f"))

# Save results
metrics_df.to_csv("enhanced_resume_ontology_evaluation.csv", index=False)
print(f"\n✅ Full results saved to 'enhanced_resume_ontology_evaluation.csv'")