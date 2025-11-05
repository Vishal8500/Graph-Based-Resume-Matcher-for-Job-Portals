"""
ontology_metrics_fixed.py
-------------------------
Evaluates standard and ontology-based metrics for your Neo4j Resume Matcher.

✅ Works with your schema:
   (:Resume)-[:HAS]->(:Skill)
   (:Job)-[:REQUIRES]->(:Skill)
   (:Skill)-[:RELATED_TO|:IS_A]->(:Skill)
   (:Resume)-[:MATCHED_TO {method:'direct'/'expanded'}]->(:Job)
   (:Resume)-[:ACTUAL_MATCH]->(:Job)   # ground truth pairs

Metrics:
  - Precision, Recall, F1, Accuracy
  - Skill Expansion Rate (SER)
  - Semantic Match Rate (SMR)
  - Ontology Coverage (OC)
  - Ranking metric (ΔNDCG)
"""
from math import log2
from neo4j import GraphDatabase
from prettytable import PrettyTable
from sklearn.metrics import ndcg_score
from dotenv import load_dotenv
import numpy as np
import os

# -------------------------------------------------------------
# 1️⃣ Connect to Neo4j
# -------------------------------------------------------------
load_dotenv()
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "vishal2004")

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# -------------------------------------------------------------
# 2️⃣ Basic metrics (Precision, Recall, F1, Accuracy)
# -------------------------------------------------------------
# -------------------------- Utility: Safe Single Value -------------------------

def safe_run(session, query, params=None, key="c", default=0):
    """Run a Cypher query and safely extract a single numeric result."""
    try:
        rec = session.run(query, params or {}).single()
        if rec and key in rec:
            return rec[key]
    except Exception as e:
        print(f"⚠️ Query failed: {e}")
    return default

# ------------------------ STEP 1: Classification Metrics -----------------------

def compute_classification_metrics():
    metrics = {}
    with driver.session() as s:
        for method in ["direct", "expanded"]:
            # True Positives (TP)
            tp = safe_run(s, """
                MATCH (r:Resume)-[:MATCHED_TO {method:$m}]->(j:Job)
                WHERE (r)-[:ACTUAL_MATCH]->(j)
                RETURN count(*) AS c
            """, {"m": method})

            # False Positives (FP)
            fp = safe_run(s, """
                MATCH (r:Resume)-[:MATCHED_TO {method:$m}]->(j:Job)
                WHERE NOT (r)-[:ACTUAL_MATCH]->(j)
                RETURN count(*) AS c
            """, {"m": method})

            # False Negatives (FN)
            fn = safe_run(s, """
                MATCH (r:Resume)-[:ACTUAL_MATCH]->(j:Job)
                WHERE NOT (r)-[:MATCHED_TO {method:$m}]->(j)
                RETURN count(*) AS c
            """, {"m": method})

            # True Negatives (TN)
            tn = safe_run(s, """
                MATCH (r:Resume), (j:Job)
                WHERE NOT (r)-[:MATCHED_TO {method:$m}]->(j)
                  AND NOT (r)-[:ACTUAL_MATCH]->(j)
                RETURN count(*) AS c
            """, {"m": method})

            # Derived metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

            metrics[method] = {
                "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                "Precision": round(precision, 3),
                "Recall": round(recall, 3),
                "F1": round(f1, 3),
                "Accuracy": round(accuracy, 3)
            }

    return metrics

# ---------------------------- STEP 2: Ontology Metrics -------------------------

def compute_ontology_metrics():
    with driver.session() as s:
        # --- Skill Expansion Rate (SER)
        # Measures how many related skills are added via ontology
        q1 = s.run("""
            MATCH (r:Resume)-[:HAS]->(s1:Skill)
            OPTIONAL MATCH (s1)-[:RELATED_TO]->(s2:Skill)
            WITH r, count(DISTINCT s1) AS explicit, count(DISTINCT s2) AS related
            RETURN coalesce(avg(related*1.0/explicit),0) AS ser
        """).single()
        ser = q1["ser"] if q1 else 0.0

        # --- Semantic Match Rate (SMR)
        # Measures how much ontology contributes to resume-job matching
        q2 = s.run("""
            MATCH (r:Resume)-[:HAS]->(s1:Skill)-[:RELATED_TO]->(s2:Skill)<-[:REQUIRES]-(j:Job)
            WITH count(DISTINCT j) AS ont_matches
            MATCH (r:Resume)-[:HAS]->(:Skill)<-[:REQUIRES]-(j:Job)
            WITH count(DISTINCT j) AS total_matches, ont_matches
            RETURN coalesce(ont_matches*1.0/total_matches,0) AS smr
        """).single()
        smr = q2["smr"] if q2 else 0.0

        # --- Ontology Coverage (OC)
        # Measures how many skills have at least one related link
        q3 = s.run("""
            MATCH (s:Skill)
            WITH count(s) AS total
            MATCH (s:Skill)-[:RELATED_TO]->()
            WITH count(DISTINCT s) AS connected, total
            RETURN coalesce(connected*1.0/total,0) AS oc
        """).single()
        oc = q3["oc"] if q3 else 0.0

    return {"SER": round(ser, 3), "SMR": round(smr, 3), "OC": round(oc, 3)}

# --------------------------- STEP 3: Ranking Metric (NDCG) ---------------------

def compute_ndcg():
    ndcg_results = {}
    with driver.session() as s:
        for method in ["direct", "expanded"]:
            # Get predicted scores for all resume-job pairs
            recs = s.run("""
                MATCH (r:Resume)-[m:MATCHED_TO {method:$m}]->(j:Job)
                OPTIONAL MATCH (r)-[:ACTUAL_MATCH]->(j)
                RETURN r.id AS resume, j.id AS job,
                       coalesce(m.score, 0.0) AS pred_score,
                       CASE WHEN (r)-[:ACTUAL_MATCH]->(j) THEN 1 ELSE 0 END AS true_label
                ORDER BY r.id, pred_score DESC
            """, {"m": method}).data()

            if not recs:
                ndcg_results[method] = 0
                continue

            # Group predictions per resume
            grouped = {}
            for r in recs:
                grouped.setdefault(r["resume"], []).append(r)

            ndcgs = []
            for resume, items in grouped.items():
                y_true = np.array([i["true_label"] for i in items])[np.newaxis, :]
                y_score = np.array([i["pred_score"] for i in items])[np.newaxis, :]
                try:
                    ndcg = ndcg_score(y_true, y_score)
                except Exception:
                    ndcg = 0
                ndcgs.append(ndcg)

            ndcg_results[method] = round(float(np.mean(ndcgs)), 3)
    return ndcg_results


# ----------------------------- STEP 4: Main Execution --------------------------

if __name__ == "__main__":
    print("🏁 Running Ontology + Performance Metrics Evaluation...")

    # Step 1 – Classification metrics
    cls = compute_classification_metrics()

    # Step 2 – Ontology metrics
    ont = compute_ontology_metrics()

    # Step 3 – Ranking metric
    ndcg = compute_ndcg()

    # -------------------- RESULTS TABLE: Classification --------------------
    print("\n📊 STANDARD METRICS COMPARISON")
    t = PrettyTable(["Metric", "Direct", "Expanded"])
    for key in ["Precision", "Recall", "F1", "Accuracy"]:
        t.add_row([key, cls["direct"][key], cls["expanded"][key]])
    print(t)

    # -------------------- RESULTS TABLE: Ontology -------------------------
    print("\n🧠 ONTOLOGY METRICS")
    o = PrettyTable(["Metric", "Value"])
    for k, v in ont.items():
        o.add_row([k, v])
    print(o)

    # -------------------- RESULTS TABLE: Ranking --------------------------
    print("\n📈 RANKING METRIC (NDCG)")
    print(f"NDCG (Direct):   {ndcg.get('direct',0)}")
    print(f"NDCG (Expanded): {ndcg.get('expanded',0)}")
    print(f"ΔNDCG: {round(ndcg.get('expanded',0) - ndcg.get('direct',0), 3)}")

    driver.close()