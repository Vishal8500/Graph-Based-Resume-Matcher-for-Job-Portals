<div align="center">
<br>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Graph%20Resume%20Matcher&fontSize=42&fontColor=ffffff&fontAlignY=38&desc=Semantic%20%C2%B7%20Explainable%20%C2%B7%20Graph-Powered&descAlignY=58&descSize=18" width="100%"/>

<br>

[![Neo4j](https://img.shields.io/badge/Neo4j-Graph_DB-00bfff?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-00d4aa?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-Frontend-61dafb?style=flat-square&logo=react&logoColor=black)](https://reactjs.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Storage-4db33d?style=flat-square&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Gemini](https://img.shields.io/badge/Gemini_LLM-Powered-4285f4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google)
[![Python](https://img.shields.io/badge/Python-3.10+-ffd343?style=flat-square&logo=python&logoColor=black)](https://python.org)

<br>

> **Stop losing great candidates to keyword filters.**
> This system understands what skills *mean* — not just what they're called.

<br>

</div>

---

## 🧠 What Is This?

**Graph Resume Matcher** is an AI-powered recruitment system that goes beyond traditional ATS keyword matching. Instead of asking *"does this resume contain the exact word?"*, it asks *"does this person actually have the right skills?"*

At its core, the system builds a living **knowledge graph** of skills, candidates, and jobs inside Neo4j — and uses **Google Gemini LLM** to continuously understand how skills relate to each other. Every match comes with a plain-English explanation of *why* it was made.

---

## 🔴 The Problem with Traditional ATS

```
Job requires:  "Django"
Resume says:   "Flask, FastAPI, Python Web Frameworks"

Traditional ATS result:  ❌  No match found
This system's result:    ✅  Matched via related skills
```

Conventional systems eliminate qualified candidates because:
- They rely on **exact keyword overlap** — synonyms are invisible
- They treat skills as isolated words, not a connected web of knowledge
- They provide **zero explanation** — a ranked list with no reasoning

This project solves both problems.

---

## ✨ How It Works

### 🔷 Step 1 — Candidates & Recruiters Upload

Candidates upload their **PDF resume**. Recruiters enter a **job description**. Both go through Google Gemini, which extracts structured skill data with deep contextual understanding.

### 🔷 Step 2 — The Knowledge Graph Grows

Skills, candidates, and jobs become **nodes** in Neo4j. When a new skill appears, Gemini automatically maps it to the rest of the graph:

```
"Flask"  ──RELATED_TO──►  "Django"
"Keras"  ──IS_A──────────► "Deep Learning Framework"
"React"  ──RELATED_TO──►  "Vue.js"
```

The ontology **builds itself** — no manual curation, no static taxonomy files.

### 🔷 Step 3 — Semantic Matching

Matching isn't just counting overlapping words. The system traverses the graph to find both **direct** and **semantically related** skill connections, scoring each candidate-job pair on a normalized 0–1 scale.

### 🔷 Step 4 — Explainable Results

Every recommendation comes with a human-readable reason:

```
✅  Direct match    →  "Python" matches "Python"
🔗  Related match  →  "Flask" is related to required skill "Django"
📦  Parent match   →  "Keras" is a type of "Deep Learning Framework"
```

No black box. No mystery scores.

---

## 📊 Performance

<div align="center">

### Model Comparison

| Metric | Static Ontology (Baseline) | **Our System** |
|:------:|:--------------------------:|:--------------:|
| Accuracy | 0.915 | 0.905 |
| **Precision** | 0.919 | **0.925 ↑** |
| **Recall** | 0.910 | **0.913 ↑** |
| **F1 Score** | 0.915 | **0.918 ↑** |

### Ontology Quality

| Metric | Meaning | Score |
|:------:|:-------:|:-----:|
| **Skill Extraction Recall** | Relationships found vs. ground truth | **3.185×** |
| **Skill Match Rate** | Correct skill-to-job alignments | **100%** |
| **Ontology Coverage** | True relationships recreated | **84%** |

</div>

> The system discovers **3× more skill relationships** per resume than a static baseline — capturing hidden semantic connections that keyword systems completely miss.
>
> Higher precision + higher recall means **fewer false matches AND fewer missed candidates** simultaneously.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│          React JS  ·  Dual-Role Portal           │
│        Candidate Upload  |  Recruiter JD         │
└────────────────────┬────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────┐
│     FastAPI Backend  ·  PyMuPDF  ·  Gemini LLM  │
│      Parse → Extract → Structure → Store         │
└──────────────┬──────────────┬───────────────────┘
               │              │
        ┌──────▼──────┐ ┌─────▼───────────────────┐
        │   MongoDB   │ │      Neo4j Graph DB       │
        │  · resumes  │ │  Candidate ──HAS──► Skill │
        │  · JD data  │ │  Job ──REQUIRES──► Skill  │
        │  · GridFS   │ │  Skill ──IS_A──► Skill    │
        └─────────────┘ │  Skill ──RELATED_TO──►    │
                        └─────────┬───────────────┘
                                  │
                     ┌────────────▼────────────────┐
                     │   Matching Engine + XAI      │
                     │  Score → Rank → Explain      │
                     └─────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React JS — Responsive dual-portal UI |
| Backend API | FastAPI (Python) — Async REST |
| Graph Database | Neo4j — Knowledge graph & traversal |
| Document Store | MongoDB + GridFS |
| LLM | Google Gemini — Skill parsing & ontology |
| PDF Parsing | PyMuPDF (fitz) |
| Auth | bcrypt — Cryptographic password hashing |

---

## 🚀 Getting Started

### Prerequisites

```
Python 3.10+   |   Node.js 18+   |   Neo4j 5.x   |   MongoDB 6.x
```

### 1 · Clone

```bash
git clone https://github.com/Vishal8500/Graph-Based-Resume-Matcher-for-Job-Portals.git
cd Graph-Based-Resume-Matcher-for-Job-Portals
```

### 2 · Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env        # fill in your API keys
uvicorn main:app --reload
```

### 3 · Frontend

```bash
cd frontend
npm install
npm start
```

### 4 · Environment Variables

```env
GEMINI_API_KEY    = your_google_gemini_key
MONGO_URI         = mongodb://localhost:27017
NEO4J_URI         = bolt://localhost:7687
NEO4J_USER        = neo4j
NEO4J_PASSWORD    = your_password
```

---

## 👥 Team

@Vishal8500
@Abishek7952


---

<div align="center">

<br>

*Built to make recruitment smarter, fairer, and more transparent.*

<br>

⭐ **Star this repo** if you found it useful!

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer" width="100%"/>

</div>
