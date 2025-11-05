from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from pymongo import MongoClient
import gridfs
from bson.objectid import ObjectId
from fastapi.middleware.cors import CORSMiddleware
from passlib.context import CryptContext
from neo4j import GraphDatabase
import google.generativeai as genai
import fitz  # PyMuPDF
import os
import json
from dotenv import load_dotenv
import tempfile
import traceback
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import time

# ---------------------------------------------------------------------------
# 1️⃣ Load environment & configure Gemini + MongoDB + Neo4j
# ---------------------------------------------------------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise Exception("❌ Missing GEMINI_API_KEY in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# MongoDB Config
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["Resume_Matcher"]
fs = gridfs.GridFS(db)

# Neo4j Config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "vishal2004")
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ---------------------------------------------------------------------------
# 2️⃣ Initialize FastAPI + CORS
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Resume & JD Analyzer API with Neo4j Integration",
    description="FastAPI + Gemini + MongoDB + Neo4j + Dynamic Ontology + XAI",
    version="4.5"  # Version bump for Cypher syntax fix
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# 3️⃣ Helper Functions — Normalization & Neo4j Sync Logic
# ---------------------------------------------------------------------------

def normalize_parsed_resume(parsed):
    """
    Normalize different possible Gemini outputs into a consistent schema.
    (This function is unchanged and is working correctly)
    """
    try:
        if not isinstance(parsed, dict):
            return {"error": "parsed is not a dict", "parsed_raw": parsed}

        out = {}
        out['parsed_raw'] = parsed  # keep original for debugging

        # --- Name/email/phone extraction ---
        name = parsed.get("name") or parsed.get("full_name") or None

        personal = parsed.get("personal_information") or parsed.get("personalInfo") or parsed.get("personal") or {}
        email = parsed.get("email") or ""
        phone = parsed.get("phone") or ""

        if isinstance(personal, dict):
            if not name:
                name = personal.get("name") or personal.get("full_name")
            contact = personal.get("contact_details") or personal.get("contact") or {}
            if isinstance(contact, dict):
                email = email or contact.get("email") or ""
                phone = phone or contact.get("phone") or ""
            elif isinstance(personal.get("Email"), str): # Fallback for other formats
                 email = email or personal.get("Email")
            elif isinstance(personal.get("Phone"), str):
                 phone = phone or personal.get("Phone")


        # fallback searches
        if not email:
            def find_email(obj):
                if isinstance(obj, str):
                    if "@" in obj and "." in obj.split("@")[-1]:
                        return obj
                if isinstance(obj, dict):
                    for v in obj.values():
                        e = find_email(v)
                        if e:
                            return e
                if isinstance(obj, list):
                    for item in obj:
                        e = find_email(item)
                        if e:
                            return e
                return None
            email = find_email(parsed) or ""

        if not phone:
            def find_phone(obj):
                if isinstance(obj, str):
                    digits = "".join(ch for ch in obj if ch.isdigit() or ch == '+')
                    if 7 <= len(digits) <= 15:
                        return obj
                if isinstance(obj, dict):
                    for v in obj.values():
                        p = find_phone(v)
                        if p:
                            return p
                if isinstance(obj, list):
                    for item in obj:
                        p = find_phone(item)
                        if p:
                            return p
                return None
            phone = find_phone(parsed) or ""

        out['name'] = name or ""
        out['email'] = email or ""
        out['phone'] = phone or ""

        # --- Summary / career objective ---
        summary = parsed.get("summary") or parsed.get("career_objective") or parsed.get("objective") or ""
        if not summary and isinstance(personal, dict):
            summary = personal.get("summary") or personal.get("career_objective") or ""
        out['summary'] = summary or ""

        # --- Skills --- normalize to list[str]
        skills = []
        raw_skills = parsed.get("skills") or parsed.get("skillset") or parsed.get("technical_skills") or {}
        if isinstance(raw_skills, dict):
            for v in raw_skills.values():
                if isinstance(v, list):
                    skills.extend([str(x).strip() for x in v if x])
                elif isinstance(v, str):
                    skills.extend([s.strip() for s in v.split(",") if s.strip()])
        elif isinstance(raw_skills, list):
            for s in raw_skills:
                if isinstance(s, str):
                    skills.append(s.strip())
                elif isinstance(s, dict):
                    skills.append(s.get("name") or s.get("skill") or str(s))
                else:
                    skills.append(str(s))
        elif isinstance(raw_skills, str):
            skills = [s.strip() for s in raw_skills.split(",") if s.strip()]

        seen = set()
        clean_skills = []
        for s in skills:
            # Simple normalization: capitalize first letter, rest lower
            s_normalized = s.strip().capitalize()
            # 🚀 Avoid empty strings after normalization
            if not s_normalized:
                continue
            key = s_normalized.lower()
            if key not in seen and s_normalized:
                seen.add(key)
                clean_skills.append(s_normalized)
        out['skills'] = clean_skills

        # --- Professional Experience normalization ---
        pro = parsed.get("professional_experience") or parsed.get("work_experience") or parsed.get("experience") or []
        normalized_exp = []
        if isinstance(pro, dict):
            for k, v in pro.items():
                if isinstance(v, dict):
                    normalized_exp.append({
                        "title": v.get("title") or v.get("role") or "",
                        "company": k,
                        "dates": v.get("dates") or v.get("duration") or "",
                        "responsibilities": v.get("responsibilities") or v.get("responsibility") or v.get("tasks") or []
                    })
                elif isinstance(v, list):
                    for ent in v:
                        if isinstance(ent, dict):
                            normalized_exp.append({
                                "title": ent.get("title") or ent.get("role") or "",
                                "company": k,
                                "dates": ent.get("dates") or ent.get("duration") or "",
                                "responsibilities": ent.get("responsibilities") or ent.get("tasks") or []
                            })
        elif isinstance(pro, list):
            for item in pro:
                if isinstance(item, dict):
                    title = item.get("title") or item.get("role") or item.get("position") or ""
                    company = item.get("company") or item.get("employer") or ""
                    dates = item.get("dates") or item.get("duration") or item.get("period") or ""
                    resp = item.get("responsibilities") or item.get("responsibility") or item.get("tasks") or item.get("description") or []
                    if isinstance(resp, str):
                        resp = [r.strip() for r in resp.split(".") if r.strip()]
                    normalized_exp.append({
                        "title": title,
                        "company": company,
                        "dates": dates,
                        "responsibilities": resp if isinstance(resp, list) else [str(resp)]
                    })
                elif isinstance(item, str):
                    normalized_exp.append({
                        "title": "",
                        "company": "",
                        "dates": "",
                        "responsibilities": [item]
                    })
        out['professional_experience'] = normalized_exp

        # --- Projects normalization ---
        proj = parsed.get("projects") or parsed.get("personal_projects") or parsed.get("project") or []
        normalized_projects = []
        if isinstance(proj, dict):
            for pname, pdetail in proj.items():
                if isinstance(pdetail, dict):
                    details = pdetail.get("details") or pdetail.get("description") or pdetail.get("points") or []
                    if isinstance(details, str):
                        details = [d.strip() for d in details.split(".") if d.strip()]
                    normalized_projects.append({
                        "title": pname,
                        "details": details if isinstance(details, list) else [str(details)]
                    })
        elif isinstance(proj, list):
            for p in proj:
                if isinstance(p, dict):
                    title = p.get("title") or p.get("name") or ""
                    details = p.get("details") or p.get("description") or p.get("points") or [] # Corrected variable name
                    if isinstance(details, str):
                        details = [d.strip() for d in details.split(".") if d.strip()]
                    normalized_projects.append({
                        "title": title,
                        "details": details if isinstance(details, list) else [str(details)]
                    })
                elif isinstance(p, str):
                    normalized_projects.append({"title": p, "details": []})
        out['projects'] = normalized_projects

        out['personal_information'] = personal if isinstance(personal, dict) else {}

        return out
    except Exception:
        traceback.print_exc()
        return {"error": "normalization_failed", "parsed_raw": parsed}


# 🚀 --- ROBUST ONTOLOGY BUILDER with ENHANCED LOGGING --- 🚀
def _get_relations_for_single_skill(skill_name: str):
    """
    Internal helper to get relations for *one* skill.
    Includes detailed logging.
    """
    prompt = f"""
    Act as an expert knowledge graph builder.
    For the *single* skill "{skill_name}", find its most important related skills and parent categories.
    Return a JSON object with a list of relations (max 5).
    Each relation must include:
    - "from": "{skill_name}"
    - "to": A *new* related skill or parent category (e.g., "Web Framework", "Containerization"). Ensure this 'to' skill is a valid, recognized technical skill or category.
    - "relation_type": "IS_A" (parent category) or "RELATED_TO" (sibling/related tech).
    - "confidence": a float between 0.0 and 1.0

    Example for input "Flask":
    {{
        "relations": [
            {{"from": "Flask", "to": "Web Framework", "relation_type": "IS_A", "confidence": 0.95}},
            {{"from": "Flask", "to": "Python", "relation_type": "RELATED_TO", "confidence": 0.9}},
            {{"from": "Flask", "to": "Django", "relation_type": "RELATED_TO", "confidence": 0.8}}
        ]
    }}

    Return only industry-relevant links. Normalize all skill names to 'Capitalized' format.
    If no relations are found, return {{"relations": []}}. Return only JSON.
    """
    raw_text = "" # Initialize raw_text
    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        print(f"      - Sending prompt to Gemini for '{skill_name}'...") # LOGGING
        response = model.generate_content(prompt)
        print(f"      - Received response from Gemini for '{skill_name}'.") # LOGGING

        # LOGGING: Get raw response text for debugging
        raw_text = response.text.strip()
        # print(f"      - Raw Gemini Response for '{skill_name}': {raw_text[:200]}...") # Uncomment for deep debug

        # Attempt to parse the JSON
        data = json.loads(raw_text)
        relations = data.get("relations", [])

        # LOGGING: Report if Gemini returned empty relations
        if not relations:
             print(f"      - Gemini returned 0 relations for '{skill_name}'.")
        else:
             print(f"      - Gemini returned {len(relations)} relations for '{skill_name}'.")

        return relations
    # LOGGING: Catch and print specific errors
    except json.JSONDecodeError as json_err:
        print(f"❌ JSON Parsing Error for skill '{skill_name}': {json_err}")
        print(f"   - Failed Response Text: {raw_text[:500]}...") # Print problematic text
        return []
    except Exception as e:
        print(f"❌ Gemini API Error for skill '{skill_name}': {type(e).__name__} - {e}")
        traceback.print_exc(limit=1) # Print concise traceback
        return []

def expand_skill_ontology_with_gemini(skills: list):
    """
    MODIFIED (ROBUST): Calls Gemini to find related skills *one by one*.
    Includes detailed logging and status tracking ('failed').
    🚀 FIXED: Corrected Cypher syntax for finding unprocessed skills.
    """
    if not skills:
        return {"status": "no_skills_provided"}

    unprocessed_skills = []
    with neo4j_driver.session() as session:
        #CYPHER QUERY SYNTAX
        result = session.run("""
            UNWIND $skills AS skillName
            MERGE (s:Skill {name: skillName})
            WITH s // Pass the node to the WHERE clause
            WHERE s.ontology_processed IS NULL
               OR s.ontology_processed = false
               OR s.ontology_processed = 'failed'
            RETURN s.name AS skillName
        """, skills=skills)
        unprocessed_skills = [record["skillName"] for record in result]

    if not unprocessed_skills:
        print("✅ Ontology: All listed skills already processed successfully.")
        return {"status": "all_skills_already_processed"}

    print(f"🛠️ Expanding ontology for {len(unprocessed_skills)} skills (one by one)...")

    total_relations_added = 0
    successful_skills = 0
    failed_skills = 0

    with neo4j_driver.session() as session:
        for skill_name in unprocessed_skills:
            print(f"  -> Processing skill: '{skill_name}'")
            relations = _get_relations_for_single_skill(skill_name)
            processed_status = 'failed' # Default status unless relations found & processed

            relations_found_count = len(relations)
            relations_added_count = 0

            if relations_found_count > 0:
                for rel in relations:
                    # Basic validation
                    confidence = rel.get("confidence", 0)
                    rel_type = rel.get("relation_type")
                    from_skill = rel.get("from", "").strip().capitalize()
                    to_skill = rel.get("to", "").strip().capitalize()

                    if confidence < 0.6:
                        print(f"      - Skipping relation due to low confidence ({confidence}): {rel}")
                        continue
                    if rel_type not in ["IS_A", "RELATED_TO"]:
                         print(f"      - Skipping relation due to invalid type ({rel_type}): {rel}")
                         continue
                    if not from_skill or not to_skill:
                         print(f"      - Skipping relation due to missing 'from' or 'to': {rel}")
                         continue
                    # Ensure 'from' matches the skill we are processing
                    if from_skill != skill_name:
                         print(f"      - Skipping relation where 'from' ({from_skill}) doesn't match processed skill ({skill_name}): {rel}")
                         continue
                    # Avoid self-loops
                    if from_skill == to_skill:
                         print(f"      - Skipping self-loop relation: {rel}")
                         continue


                    # If validation passes, attempt to write to Neo4j
                    try:
                        session.run(f"""
                            MERGE (s1:Skill {{name: $from_skill}})
                            MERGE (s2:Skill {{name: $to_skill}})
                            MERGE (s1)-[r:{rel_type}]->(s2)
                            SET r.source = 'LLM',
                                r.confidence = $confidence,
                                r.updated_at = $timestamp
                            MERGE (s2)-[r_inv:{rel_type}]->(s1)
                            SET r_inv.source = 'LLM',
                                r_inv.confidence = $confidence,
                                r_inv.updated_at = $timestamp
                        """,
                        from_skill=from_skill,
                        to_skill=to_skill,
                        confidence=confidence, # Use validated confidence
                        timestamp=datetime.now().isoformat())
                        relations_added_count += 1
                    except Exception as neo_err:
                        print(f"      - ❌ Neo4j Error writing relation {rel}: {neo_err}")
                        # Don't mark the whole skill as failed just for one bad relation write

                # If at least one relation was successfully added, mark skill as success
                if relations_added_count > 0:
                     processed_status = True # Use boolean true for success
                     total_relations_added += relations_added_count
                     print(f"      - Successfully added {relations_added_count} relations for '{skill_name}'.")
                else:
                     # Gemini returned relations, but none were valid or writable
                     print(f"      - No valid relations added for '{skill_name}' despite Gemini returning {relations_found_count}.")
                     processed_status = 'failed' # Mark as failed if no relations actually got written

            else:
                 # Gemini returned [] or API failed
                 processed_status = 'failed' # Mark as failed


            # Mark this *one* skill as processed (True, False, or 'failed')
            try:
                session.run("""
                    MATCH (s:Skill {name: $skillName})
                    SET s.ontology_processed = $status, s.last_processed = $timestamp
                """, skillName=skill_name, status=processed_status, timestamp=datetime.now().isoformat())
            except Exception as neo_err:
                 print(f"      - ❌ Neo4j Error updating processed status for '{skill_name}': {neo_err}")


            if processed_status is True:
                successful_skills += 1
            else:
                failed_skills += 1

            # Keep rate limit
            time.sleep(1.1) # Slightly increased delay

    print(f"✅ Ontology expansion attempt finished.")
    print(f"   - Total Relations Added: {total_relations_added}")
    print(f"   - Skills Marked Successful: {successful_skills}")
    print(f"   - Skills Marked Failed/No Relations: {failed_skills}")

    return {
        "status": "finished",
        "relations_added": total_relations_added,
        "skills_processed_successfully": successful_skills,
        "skills_failed": failed_skills
    }
# --- END OF ONTOLOGY BUILDER ---


def rebuild_ontology():
    """
    Fetches ALL skills from Neo4j and re-runs the ontology expansion.
    """
    all_skills = []
    with neo4j_driver.session() as session:
        # MODIFIED: Clear existing processing flags AND failed flags
        session.run("MATCH (s:Skill) SET s.ontology_processed = false, s.last_processed = null")

        result = session.run("MATCH (s:Skill) RETURN s.name AS skillName")
        all_skills = [record["skillName"] for record in result]

    if all_skills:
        # This will now call the new, robust, one-by-one function
        expand_skill_ontology_with_gemini(all_skills)

    return {"status": "rebuild_complete", "total_skills": len(all_skills)}


def push_jobs_to_neo4j():
    jobs = list(db["JD_skills"].find())
    with neo4j_driver.session() as session:
        for job in jobs:
            job_id = str(job["_id"])
            job_title = job.get("job_title", "Unknown Job")
            skills = job.get("skills", [])

            session.run("""
                MERGE (j:Job {id:$job_id})
                SET j.title=$job_title
            """, job_id=job_id, job_title=job_title)

            for skill in skills:
                # Ensure skill is capitalized before merging
                skill_name = skill.strip().capitalize()
                if skill_name:
                    session.run("MERGE (s:Skill {name:$skill})", skill=skill_name)
                    session.run("""
                        MATCH (j:Job {id:$job_id}), (s:Skill {name:$skill})
                        MERGE (j)-[:REQUIRES]->(s)
                    """, job_id=job_id, skill=skill_name)
    print("✅ Jobs pushed to Neo4j.")


def push_resumes_to_neo4j():
    resumes = list(db["resumes"].find())
    with neo4j_driver.session() as session:
        for resume in resumes:
            resume_id = str(resume.get("_id"))
            file_id = resume.get("gridfs_file_id", "")

            # prefer normalized fields when available
            name = resume.get("name") or resume.get("parsed_raw", {}).get("name") or "Unknown"
            email = resume.get("email") or resume.get("parsed_raw", {}).get("email") or "N/A"
            phone = resume.get("phone") or resume.get("parsed_raw", {}).get("phone") or "N/A"
            summary = resume.get("summary") or resume.get("parsed_raw", {}).get("summary") or "No summary available."

            skills = resume.get("skills") or []
            if isinstance(skills, dict): # Should not happen after normalization, but defensive check
                flat_skills = []
                for skill_list in skills.values():
                    if isinstance(skill_list, list):
                        flat_skills.extend(skill_list)
                skills = flat_skills

            session.run("""
                MERGE (r:Resume {id:$resume_id})
                SET r.name=$name, r.file_id=$file_id, r.email=$email, r.phone=$phone, r.summary=$summary
            """, resume_id=resume_id, name=name, file_id=file_id, email=email, phone=phone, summary=summary)

            # Clear existing HAS relationships before adding new ones
            session.run("MATCH (r:Resume {id:$resume_id})-[rel:HAS]->() DELETE rel", resume_id=resume_id)

            for skill in skills:
                # Ensure skill is capitalized before merging
                skill_name = skill.strip().capitalize()
                if isinstance(skill_name, str) and skill_name:
                    session.run("MERGE (s:Skill {name:$skill})", skill=skill_name)
                    session.run("""
                        MATCH (r:Resume {id:$resume_id}), (s:Skill {name:$skill})
                        MERGE (r)-[:HAS]->(s)
                    """, resume_id=resume_id, skill=skill_name)
    print("✅ Resumes (with corrected flat details & HAS rels) pushed to Neo4j.")


def recommend_jobs(resume_id, limit=5, mode: str = "expanded"):
    """
    MODIFIED: Now accepts a 'mode' parameter to toggle scoring logic.
    - 'expanded': (default) Uses weighted scoring (direct=1.0, related=0.5)
    - 'direct': Uses simple direct skill count.
    """
    with neo4j_driver.session() as session:

        if mode == "direct":
            # --- DIRECT-ONLY SCORING (Old Logic) ---
            result = session.run("""
                MATCH (r:Resume {id:$resume_id})-[:HAS]->(s:Skill)<-[:REQUIRES]-(j:Job)
                RETURN j.id AS job_id,
                       j.title AS job_title,
                       count(s) AS directScore
                ORDER BY directScore DESC
                LIMIT $limit
            """, resume_id=resume_id, limit=limit)

            recommendations = []
            for record in result:
                job_id = record["job_id"]
                job_doc = db["JD_skills"].find_one({"_id": ObjectId(job_id)})

                recommendations.append({
                    "job_id": job_id,
                    "job_title": record["job_title"],
                    "company_portal_link": job_doc.get("company_portal_link", "") if job_doc else "",
                    "skills": job_doc.get("skills", []) if job_doc else [],
                    "weightedScore": float(record["directScore"]), # Use float for consistency
                    "directScore": record["directScore"],
                    "relatedScore": 0,
                    "matchedSkills": record["directScore"]
                })

        else:
            # --- EXPANDED SCORING (New Logic) ---
            result = session.run("""
                MATCH (r:Resume {id:$resume_id})-[:HAS]->(rs:Skill) // Candidate's skills
                WITH r, collect(DISTINCT rs) AS candidateSkills
                MATCH (j:Job)-[:REQUIRES]->(js:Skill) // Job's required skills

                // Calculate direct matches
                WITH r, j, candidateSkills, js,
                     CASE WHEN js IN candidateSkills THEN 1 ELSE 0 END AS directMatch

                // Calculate related matches (1-hop)
                WITH r, j, candidateSkills, js, directMatch
                OPTIONAL MATCH (rs_related)-[:RELATED_TO|IS_A]->(js)
                WHERE rs_related IN candidateSkills AND directMatch = 0 // Check for 1-hop relation, only if not a direct match
                WITH r, j, js, directMatch,
                     CASE WHEN rs_related IS NOT NULL THEN 1 ELSE 0 END AS relatedMatch

                // Aggregate scores
                WITH j,
                     SUM(directMatch) AS directScore,
                     SUM(relatedMatch) AS relatedScore
                WHERE directScore + relatedScore > 0

                RETURN j.id AS job_id,
                       j.title AS job_title,
                       (directScore * 1.0) + (relatedScore * 0.5) AS weightedScore,
                       directScore,
                       relatedScore
                ORDER BY weightedScore DESC
                LIMIT $limit
            """, resume_id=resume_id, limit=limit)

            recommendations = []
            for record in result:
                job_id = record["job_id"]
                job_doc = db["JD_skills"].find_one({"_id": ObjectId(job_id)})

                recommendations.append({
                    "job_id": job_id,
                    "job_title": record["job_title"],
                    "company_portal_link": job_doc.get("company_portal_link", "") if job_doc else "",
                    "skills": job_doc.get("skills", []) if job_doc else [],
                    "weightedScore": record["weightedScore"],
                    "directScore": record["directScore"],
                    "relatedScore": record["relatedScore"],
                    "matchedSkills": record["directScore"] + record["relatedScore"] # For frontend compatibility
                })

    return recommendations



def eligible_applicants(job_id):
    """
    Finds applicants based on direct AND related skills (1-hop).
    Implements weighted scoring: direct=1.0, related=0.5
    """
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (j:Job {id:$job_id})-[:REQUIRES]->(js:Skill) // Job's skills
            WITH j, collect(DISTINCT js) AS jobSkills
            MATCH (r:Resume)-[:HAS]->(rs:Skill) // Candidate's skills

            // Calculate direct matches
            WITH j, r, jobSkills, rs,
                 CASE WHEN rs IN jobSkills THEN 1 ELSE 0 END AS directMatch

            // Calculate related matches (1-hop)
            WITH j, r, jobSkills, rs, directMatch
            OPTIONAL MATCH (rs)-[:RELATED_TO|IS_A]->(js_related)
            WHERE js_related IN jobSkills AND directMatch = 0 // Check for 1-hop relation, only if not a direct match
            WITH j, r, rs, directMatch,
                 CASE WHEN js_related IS NOT NULL THEN 1 ELSE 0 END AS relatedMatch

            // Aggregate scores
            WITH r,
                 SUM(directMatch) AS directScore,
                 SUM(relatedMatch) AS relatedScore
            WHERE directScore + relatedScore > 0

            RETURN r.id AS resume_id,
                   r.name AS resume_name,
                   r.file_id AS file_id,
                   r.email AS email,
                   r.phone AS phone,
                   r.summary AS summary,
                   (directScore * 1.0) + (relatedScore * 0.5) AS weightedScore,
                   directScore,
                   relatedScore
            ORDER BY weightedScore DESC
        """, job_id=job_id)

        applicants = []
        for record in result:
            applicants.append({
                "resume_id": record["resume_id"],
                "resume_name": record["resume_name"],
                "file_id": record["file_id"],
                "email": record["email"],
                "phone": record["phone"],
                "summary": record["summary"],
                "weightedScore": record["weightedScore"],
                "directScore": record["directScore"],
                "relatedScore": record["relatedScore"],
                "matchedSkills": record["directScore"] + record["relatedScore"] # For frontend compatibility
            })
    return applicants

# ---------------------------------------------------------------------------
# 4️⃣ Authentication (Signup / Login)
# ---------------------------------------------------------------------------

@app.post("/signup/")
def signup(username: str = Form(...), password: str = Form(...)):
    try:
        existing = db["users"].find_one({"username": username})
        if existing:
            return JSONResponse(content={"status": "failed", "message": "User already exists"}, status_code=400)

        hashed_pwd = pwd_context.hash(password)
        db["users"].insert_one({"username": username, "password": hashed_pwd})
        return {"status": "success", "message": "User registered successfully"}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"status": "error", "details": str(e), "trace": traceback.format_exc()}, status_code=500)


@app.post("/login/")
def login(username: str = Form(...), password: str = Form(...)):
    """User login: admin → /extract_jd_skills, others → /parse_resume."""
    if username == "admin" and password == "admin":
        return {"status": "success", "role": "admin", "redirect": "/extract_jd_skills"}

    user = db["users"].find_one({"username": username})
    if not user:
        return JSONResponse(content={"status": "failed", "message": "User not found"}, status_code=401)

    if not pwd_context.verify(password, user["password"]):
        return JSONResponse(content={"status": "failed", "message": "Invalid password"}, status_code=401)

    return {"status": "success", "role": "user", "redirect": "/parse_resume"}

# ---------------------------------------------------------------------------
# 5️⃣ Resume Parsing (Gemini + MongoDB + Neo4j)
# ---------------------------------------------------------------------------

@app.post("/parse_resume/")
async def parse_resume(file: UploadFile = File(...), username: str = Form(...)):
    """
    Using the correct, complex prompt that matches the normalize_parsed_resume function.
    This will fix the missing name, work experience, and skills.
    Triggers the robust ontology builder.
    """
    try:
        file_content = await file.read()

        with fitz.open(stream=file_content, filetype="pdf") as doc:
            raw_text = "".join(page.get_text() for page in doc)
            if not raw_text.strip():
                return JSONResponse(
                    content={"status": "failed", "error": "No text in PDF"},
                    status_code=400
                )

        # Prompt for resume parsing (matches normalize function)
        prompt = f"""
Act as an expert resume parser. Analyze the text below and return a structured JSON object.
The JSON structure should include (but adapt if necessary):
- "personal_information": {{ "name": "...", "contact_details": {{ "email": "...", "phone": "..." }} }}
- "summary": "..." (or "career_objective")
- "skills": {{ "programming_languages": [...], "frameworks": [...], "tools": [...] }} OR a flat list ["Python","SQL",...]
- "professional_experience": [{{ "title": "...", "company": "...", "dates": "...", "responsibilities": [...] }}]
- "projects": [{{ "title": "...", "details": [...] }}]

Resume Text:
---
{raw_text}
---

Return only valid JSON. If a field is missing, you may omit it. Make the structure JSON-first so it can be parsed programmatically.
"""

        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )

        safety_settings = {
            'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
            'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
            'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
            'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE'
        }

        response = model.generate_content(prompt, safety_settings=safety_settings)

        # Normalize the parsed data
        raw_parsed_data = json.loads(response.text)
        parsed_data = normalize_parsed_resume(raw_parsed_data)

        # Check if normalization failed or essential data is missing
        if parsed_data.get("error") or not parsed_data.get("skills"):
             print(f"❌ Normalization failed or no skills found for {file.filename}.")
             print(f"   Raw parsed data: {raw_parsed_data}") # Log raw data if normalization fails
             # Optionally, return an error to the frontend
             # return JSONResponse(content={"status": "failed", "error": "Could not extract skills"}, status_code=400)


        # Save the resume file to GridFS
        file_id = fs.put(file_content, filename=file.filename)
        parsed_data['gridfs_file_id'] = str(file_id)

        # Associate resume with the logged-in username
        parsed_data['username'] = username

        # Remove any previous resume from this user
        existing = db["resumes"].find_one({"username": username})
        if existing:
            print(f"Deleting existing resume for user {username} (ID: {existing.get('_id')})")
            db["resumes"].delete_one({"username": username})
            try:
                if existing.get("gridfs_file_id"):
                    print(f"Deleting associated GridFS file: {existing['gridfs_file_id']}")
                    fs.delete(ObjectId(existing["gridfs_file_id"]))
            except Exception as gridfs_err:
                 print(f"Warning: Failed to delete old GridFS file {existing.get('gridfs_file_id')}: {gridfs_err}")


        # Insert the new parsed resume
        result = db["resumes"].insert_one(parsed_data)
        parsed_data['_id'] = str(result.inserted_id)
        resume_id = str(result.inserted_id)
        print(f"✅ Successfully inserted new resume for {username} (ID: {resume_id})")


        # Push resume to Neo4j (will clear old :HAS and add new)
        push_resumes_to_neo4j()

        # Trigger the ROBUST skill ontology expansion
        try:
            skill_list = parsed_data.get("skills", [])
            if skill_list:
                print(f"Triggering ontology expansion for {len(skill_list)} skills from resume {resume_id}...")
                # Run this async in a real app, but sync here for simplicity
                expand_skill_ontology_with_gemini(skill_list)
            else:
                 print(f"ℹ️ No skills extracted from resume {resume_id}, skipping ontology expansion.")
        except Exception as e:
            print(f"⚠️ WARNING: Skill ontology expansion failed during trigger: {e}")
            traceback.print_exc(limit=1)

        # Get job recommendations using the expanded logic (default for parse)
        recommendations = recommend_jobs(resume_id, limit=5, mode="expanded")

        return {"status": "success", "data": parsed_data, "recommendations": recommendations}

    except Exception as e:
        print(f"❌ CRITICAL ERROR in /parse_resume: {e}")
        traceback.print_exc()
        return JSONResponse(
            content={"status": "failed", "error": "An internal server error occurred during parsing."},
            status_code=500
        )



@app.get("/my_resume/")
def get_my_resume(username: str):
    """Return saved resume for this username (if any)."""
    try:
        doc = db["resumes"].find_one({"username": username})
        if not doc:
            return {"found": False}
        doc_copy = dict(doc)
        doc_copy["_id"] = str(doc.get("_id"))
        # Ensure 'skills' is always a list for consistency
        if 'skills' not in doc_copy or not isinstance(doc_copy['skills'], list):
            # Attempt to re-normalize if needed (might indicate old data format)
            normalized = normalize_parsed_resume(doc_copy)
            doc_copy['skills'] = normalized.get('skills', [])
        return {"found": True, "data": doc_copy}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"status": "failed", "error": str(e)}, status_code=500)


@app.delete("/my_resume/")
def delete_my_resume(username: str):
    """Delete saved resume and GridFS file for a username."""
    try:
        doc = db["resumes"].find_one({"username": username})
        if not doc:
            return {"status": "failed", "message": "No resume found"}

        resume_id = str(doc.get("_id")) # Get ID for Neo4j deletion

        # Delete from MongoDB
        print(f"Deleting MongoDB resume for user {username} (ID: {resume_id})")
        db["resumes"].delete_one({"_id": doc["_id"]})

        # Delete GridFS file
        try:
            if doc.get("gridfs_file_id"):
                print(f"Deleting associated GridFS file: {doc['gridfs_file_id']}")
                fs.delete(ObjectId(doc["gridfs_file_id"]))
        except Exception as gridfs_err:
             print(f"Warning: Failed to delete GridFS file {doc.get('gridfs_file_id')}: {gridfs_err}")


        # Delete Resume node and its relationships from Neo4j
        try:
            print(f"Deleting Neo4j node and relationships for resume ID: {resume_id}")
            with neo4j_driver.session() as session:
                 session.run("MATCH (r:Resume {id: $resume_id}) DETACH DELETE r", resume_id=resume_id)
            print(f"✅ Neo4j node deleted for resume ID: {resume_id}")
            # No need to call push_resumes_to_neo4j() anymore
        except Exception as neo_err:
            print(f"⚠️ WARNING: Failed to delete Neo4j node for resume {resume_id}: {neo_err}")
            traceback.print_exc(limit=1)

        return {"status": "success", "message": "Resume deleted"}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"status": "failed", "error": str(e)}, status_code=500)

# ---------------------------------------------------------------------------
# 6️⃣ Job Description Skill Extraction (Gemini + MongoDB + Neo4j)
# ---------------------------------------------------------------------------

def extract_skills_with_gemini(job_description: str):
    prompt = f"""
You are an expert career analyst. Extract all technical and soft skills from the job description.
---
{job_description}
---
Output JSON: {{ "skills": [ "Python", "SQL", ... ] }}
"""
    try:
        model = genai.GenerativeModel(
            "gemini-2.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(prompt)
        text = response.text.strip().replace("```json", "").replace("```", "")
        data = json.loads(text)

        # Normalize skills
        skills_raw = data.get("skills", [])
        seen = set()
        clean_skills = []
        for s in skills_raw:
            s_normalized = s.strip().capitalize()
            # Avoid empty strings
            if not s_normalized:
                continue
            key = s_normalized.lower()
            if key not in seen and s_normalized:
                seen.add(key)
                clean_skills.append(s_normalized)
        return clean_skills

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}


@app.post("/extract_jd_skills/")
async def extract_jd_skills(
    job_description: str = Form(...),
    job_title: str = Form(...),
    company_portal_link: str = Form(...)
):
    """
    Recruiter posts JD -> extract skills -> save -> sync Neo4j ->
    TRIGGER ROBUST ONTOLOGY EXPANSION -> return EXPANDED eligible applicants
    """
    try:
        # Extract key skills using Gemini
        skills = extract_skills_with_gemini(job_description)
        if isinstance(skills, dict) and "error" in skills:
            return JSONResponse(content={"status": "failed", "error": skills["error"]}, status_code=400)

        # Prepare JD document for MongoDB
        doc = {
            "job_title": job_title.strip(),
            "company_portal_link": company_portal_link.strip(),
            "job_description": job_description.strip(),
            "skills": skills
        }

        # Save to MongoDB
        result = db["JD_skills"].insert_one(doc)
        job_id = str(result.inserted_id)
        doc["_id"] = job_id

        # Sync to Neo4j
        push_jobs_to_neo4j() # Call helper to handle sync logic

        print(f"✅ JD pushed to Neo4j: {job_title}")

        # Trigger the ROBUST skill ontology expansion
        try:
            if skills:
                print(f"Triggering ontology expansion for {len(skills)} skills from job {job_id}...")
                expand_skill_ontology_with_gemini(skills)
            else:
                print(f"ℹ️ No skills extracted from job {job_id}, skipping ontology expansion.")
        except Exception as e:
            print(f"⚠️ WARNING: Skill ontology expansion failed during trigger: {e}")
            traceback.print_exc(limit=1)

        # Find eligible applicants based on expanded skills
        applicants = eligible_applicants(job_id)

        # Return complete JD info and matched applicants
        return {
            "status": "success",
            "data": {
                "_id": job_id,
                "job_title": job_title.strip(),
                "company_portal_link": company_portal_link.strip(),
                "job_description": job_description.strip(),
                "skills": skills
            },
            "applicants": applicants
        }

    except Exception as e:
        print(f"❌ CRITICAL ERROR in /extract_jd_skills: {e}")
        traceback.print_exc()
        return JSONResponse(
            content={"status": "failed", "error": "An internal server error occurred during JD processing."},
            status_code=500
        )



# ---------------------------------------------------------------------------
# 7️⃣ Public Endpoints for Frontend (Now using expanded matching)
# ---------------------------------------------------------------------------

@app.get("/recommend_jobs/")
def get_recommendations(resume_id: str, mode: str = "expanded"):
    """
    MODIFIED: Gets recommendations using the specified scoring 'mode'.
    'expanded' (default) or 'direct'
    """
    recs = recommend_jobs(resume_id, mode=mode)
    return {"recommendations": recs}


@app.get("/eligible_applicants/")
def get_eligible_applicants(job_id: str):
    """Gets applicants using the MODIFIED expanded/weighted logic."""
    applicants = eligible_applicants(job_id)
    return {"applicants": applicants}


@app.get("/download_resume/{file_id}")
def download_resume(file_id: str):
    """Retrieves a resume from GridFS and streams it for download."""
    try:
        gridfs_file = fs.get(ObjectId(file_id))
        return StreamingResponse(gridfs_file, media_type='application/pdf',
                                 headers={"Content-Disposition": f"attachment; filename=\"{gridfs_file.filename}\""})
    except gridfs.errors.NoFile:
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

# ---------------------------------------------------------------------------
# 8️⃣ Ontology & XAI Endpoints
# ---------------------------------------------------------------------------

class SkillList(BaseModel):
    skills: List[str]

@app.post("/ontology/expand")
def api_expand_ontology(skill_list: SkillList):
    """
    Manual endpoint to trigger ontology expansion for a given list of skills.
    """
    try:
        # Normalize skills before expanding
        seen = set()
        clean_skills = []
        for s in skill_list.skills:
            s_normalized = s.strip().capitalize()
            if not s_normalized: continue
            key = s_normalized.lower()
            if key not in seen and s_normalized:
                seen.add(key)
                clean_skills.append(s_normalized)

        result = expand_skill_ontology_with_gemini(clean_skills)
        return {"status": "success", "result": result}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"status": "failed", "error": str(e)}, status_code=500)

@app.post("/ontology/rebuild")
def api_rebuild_ontology():
    """
    Manual endpoint to trigger a full rebuild of the skill ontology.
    """
    try:
        result = rebuild_ontology()
        return {"status": "success", "result": result}
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"status": "failed", "error": str(e)}, status_code=500)

@app.get("/explain_match/")
def explain_match(resume_id: str, job_id: str):
    """
    XAI Endpoint.
    (This query is correct and will work once the ontology is built)
    """
    try:
        with neo4j_driver.session() as session:
            result = session.run("""
                MATCH (r:Resume {id: $resume_id})-[:HAS]->(rs:Skill)
                MATCH (j:Job {id: $job_id})-[:REQUIRES]->(js:Skill)

                // Find all paths (direct and 1-hop related)
                // 0..1 hops: 0 = direct match (rs == js), 1 = related match
                OPTIONAL MATCH p = shortestPath((rs)-[:RELATED_TO|IS_A*0..1]->(js))
                WHERE p IS NOT NULL

                WITH p, nodes(p)[0] AS candidateSkillNode, nodes(p)[-1] AS jobSkillNode
                RETURN
                    candidateSkillNode.name AS candidateSkill,
                    jobSkillNode.name AS jobSkill,
                    length(p) AS pathLength,
                    [rel in relationships(p) | type(rel)] AS relations
                LIMIT 10 // Limit explanations for brevity
            """, resume_id=resume_id, job_id=job_id)

            paths = []
            explanations = []
            seen_explanations = set()

            for record in result:
                path_data = {
                    "candidateSkill": record["candidateSkill"],
                    "jobSkill": record["jobSkill"],
                    "pathLength": record["pathLength"],
                    "relations": record["relations"]
                }
                paths.append(path_data)

                explanation = ""
                candidate_skill = record['candidateSkill']
                job_skill = record['jobSkill']

                if record["pathLength"] == 0:
                    # Direct match check
                    if candidate_skill == job_skill:
                        explanation = f"Direct match: Your skill **{candidate_skill}** matches the requirement."
                    # else: # Should not happen with *0..1 and WHERE p IS NOT NULL
                    #     explanation = f"Path length 0 but skills differ ({candidate_skill} vs {job_skill}) - Check Cypher query."

                elif record["pathLength"] == 1:
                    # Related match check
                    if record["relations"]: # Ensure relations list is not empty
                        rel_type = record["relations"][0].replace("_", " ").lower()
                        explanation = f"Related match: Your skill **{candidate_skill}** is **{rel_type}** the required skill **{job_skill}**."
                    else: # Should not happen if pathLength is 1
                         explanation = f"Path length 1 but no relation type found for {candidate_skill} -> {job_skill}."

                # Add unique explanations
                if explanation and explanation not in seen_explanations:
                    explanations.append(explanation)
                    seen_explanations.add(explanation)

            if not explanations:
                 # Check if there was ANY overlap, even if paths weren't found (fallback)
                 direct_overlap_check = session.run("""
                     MATCH (r:Resume {id: $resume_id})-[:HAS]->(s:Skill)<-[:REQUIRES]-(j:Job {id: $job_id})
                     RETURN count(s) > 0 AS hasOverlap
                 """, resume_id=resume_id, job_id=job_id)
                 if direct_overlap_check.single()["hasOverlap"]:
                     return {"paths": [], "explanations": ["Direct skill matches exist, but explanation path query failed. Check Cypher/DB state."]}
                 else:
                     return {"paths": [], "explanations": ["No clear skill matches (direct or related) found."]}


            return {"paths": paths, "explanations": explanations}

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"status": "failed", "error": str(e)}, status_code=500)


@app.get("/ontology/explore")
def get_skill_relations(skill: str):
    """
    New endpoint for the frontend SkillGraphExplorer.
    """
    if not skill:
        return JSONResponse(content={"error": "Skill parameter is required"}, status_code=400)

    # Normalize skill name to match DB
    skill_name = skill.strip().capitalize()

    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (s:Skill {name: $skill_name})-[r:RELATED_TO|IS_A]->(s2:Skill)
            RETURN s2.name AS relatedSkill, type(r) AS relationType, r.confidence as confidence
            ORDER BY relationType, confidence DESC
            LIMIT 25
        """, skill_name=skill_name)

        relations = []
        for record in result:
            relations.append({
                "skill": record["relatedSkill"],
                "type": record["relationType"],
                "confidence": record["confidence"]
            })

    return {"skill": skill_name, "relations": relations}

# ---------------------------------------------------------------------------
# 9️⃣ Root Endpoint
# ---------------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": f"🚀 Resume & JD Analyzer API v{app.version} (Enhanced Ontology Logging) Ready!"}

# ---------------------------------------------------------------------------
# 10️⃣ Run:
# uvicorn main:app --reload
# ---------------------------------------------------------------------------