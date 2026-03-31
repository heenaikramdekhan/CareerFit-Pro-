"""
CareerFit Pro — AI Resume & Job Matching System
Semantic matching using sentence-transformers + spaCy NER
"""

import re
import io
import json
import streamlit as st
import pdfplumber
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="CareerFit Pro",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono&display=swap');

  html, body, [class*="css"] { font-family: 'DM Mono', monospace; }
  h1,h2,h3 { font-family: 'Syne', sans-serif !important; }

  .stApp { background: #080c14; color: #e8edf5; }

  /* Score card */
  .score-card {
    background: #0d1421;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    position: relative;
    overflow: hidden;
  }
  .score-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--accent, #00d4ff);
  }
  .score-num {
    font-family: 'Syne', sans-serif;
    font-size: 42px;
    font-weight: 800;
    color: var(--accent, #00d4ff);
    line-height: 1;
  }
  .score-label {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #7a8ba0;
    margin-top: 6px;
  }

  /* Skill chip */
  .chip-match {
    display: inline-block;
    padding: 3px 10px;
    background: rgba(16,185,129,0.12);
    color: #10b981;
    border: 1px solid rgba(16,185,129,0.3);
    border-radius: 4px;
    font-size: 11px;
    margin: 2px;
  }
  .chip-gap {
    display: inline-block;
    padding: 3px 10px;
    background: rgba(239,68,68,0.1);
    color: #ef4444;
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 4px;
    font-size: 11px;
    margin: 2px;
  }
  .chip-neutral {
    display: inline-block;
    padding: 3px 10px;
    background: rgba(0,212,255,0.08);
    color: #00d4ff;
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 4px;
    font-size: 11px;
    margin: 2px;
  }

  /* Section panel */
  .panel {
    background: #0d1421;
    border: 1px solid #1e2d45;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
  }
  .panel-title {
    font-size: 10px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00d4ff;
    margin-bottom: 12px;
  }

  /* Progress bar override */
  .stProgress > div > div > div { background: #00d4ff !important; }

  /* Verdict box */
  .verdict-high { background: rgba(16,185,129,0.1); border-left: 3px solid #10b981; padding: 12px; border-radius: 4px; }
  .verdict-mid  { background: rgba(245,158,11,0.1); border-left: 3px solid #f59e0b; padding: 12px; border-radius: 4px; }
  .verdict-low  { background: rgba(239,68,68,0.08); border-left: 3px solid #ef4444; padding: 12px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# MODEL LOADING (cached)
# ──────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load NLP models once and cache them."""
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        st.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
        st.stop()
    return embedder, nlp


# ──────────────────────────────────────────────
# SKILL TAXONOMY
# ──────────────────────────────────────────────
SKILL_TAXONOMY = {
    "Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "kotlin", "swift", "r", "scala", "ruby", "php", "bash", "sql", "matlab",
    ],
    "Web & APIs": [
        "react", "angular", "vue", "node.js", "fastapi", "django", "flask",
        "spring", "express", "graphql", "rest api", "grpc", "html", "css",
    ],
    "Data & ML": [
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
        "xgboost", "hugging face", "nlp", "computer vision", "spark", "hadoop",
        "tableau", "power bi", "dbt", "airflow", "mlops",
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
        "ci/cd", "github actions", "jenkins", "linux", "helm",
    ],
    "Databases": [
        "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra",
        "dynamodb", "snowflake", "bigquery", "sqlite",
    ],
    "Practices": [
        "agile", "scrum", "tdd", "bdd", "microservices", "system design",
        "code review", "git", "jira", "devops", "sre", "devsecops",
    ],
}

ALL_SKILLS = {skill for skills in SKILL_TAXONOMY.values() for skill in skills}


# ──────────────────────────────────────────────
# TEXT EXTRACTION
# ──────────────────────────────────────────────
def extract_pdf_text(uploaded_file) -> str:
    """Extract plain text from an uploaded PDF using pdfplumber."""
    text_pages = []
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        for page in pdf.pages:
            raw = page.extract_text()
            if raw:
                text_pages.append(raw)
    return "\n".join(text_pages).strip()


def clean_text(text: str) -> str:
    """Normalize whitespace, strip special characters."""
    text = re.sub(r"[^\w\s\.\,\+\#\-\/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


# ──────────────────────────────────────────────
# SKILL EXTRACTION
# ──────────────────────────────────────────────
def extract_skills(text: str) -> dict[str, list[str]]:
    """Return skills found, grouped by taxonomy category."""
    lower = text.lower()
    found: dict[str, list[str]] = defaultdict(list)
    for category, skills in SKILL_TAXONOMY.items():
        for skill in skills:
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, lower):
                found[category].append(skill)
    return dict(found)


def flat_skills(skills_dict: dict) -> set[str]:
    return {s for lst in skills_dict.values() for s in lst}


# ──────────────────────────────────────────────
# EXPERIENCE EXTRACTION
# ──────────────────────────────────────────────
_EXP_PATTERNS = [
    r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s*)?(?:experience|exp)",
    r"experience\s*(?:of\s*)?(\d+)\+?\s*(?:years?|yrs?)",
]

def extract_years(text: str) -> int | None:
    lower = text.lower()
    for pat in _EXP_PATTERNS:
        m = re.search(pat, lower)
        if m:
            return int(m.group(1))
    return None


def classify_seniority(years: int | None, text: str) -> str:
    low = text.lower()
    if years is not None:
        if years >= 8: return "Senior / Principal"
        if years >= 5: return "Mid–Senior"
        if years >= 2: return "Mid-Level"
        return "Junior / Entry-Level"
    for kw, label in [
        ("senior", "Senior / Principal"), ("lead", "Senior / Principal"),
        ("principal", "Senior / Principal"), ("staff", "Senior / Principal"),
        ("mid-level", "Mid-Level"), ("junior", "Junior / Entry-Level"),
        ("entry", "Junior / Entry-Level"), ("intern", "Intern"),
    ]:
        if kw in low:
            return label
    return "Not Specified"


# ──────────────────────────────────────────────
# EDUCATION EXTRACTION
# ──────────────────────────────────────────────
_EDU_KW = {
    "phd": "PhD / Doctorate",
    "doctorate": "PhD / Doctorate",
    "master": "Master's Degree",
    "msc": "Master's Degree",
    "m.s.": "Master's Degree",
    "mba": "MBA",
    "bachelor": "Bachelor's Degree",
    "b.s.": "Bachelor's Degree",
    "b.e.": "Bachelor's Degree",
    "btech": "Bachelor's Degree",
    "b.tech": "Bachelor's Degree",
    "associate": "Associate Degree",
    "diploma": "Diploma",
    "high school": "High School",
}

def extract_education(text: str) -> str:
    low = text.lower()
    for kw, label in _EDU_KW.items():
        if kw in low:
            return label
    return "Not Detected"


# ──────────────────────────────────────────────
# SEMANTIC SIMILARITY ENGINE
# ──────────────────────────────────────────────
def compute_semantic_score(embedder, text_a: str, text_b: str) -> float:
    """Cosine similarity of sentence-transformer embeddings (0–100)."""
    emb_a = embedder.encode(text_a, convert_to_tensor=True)
    emb_b = embedder.encode(text_b, convert_to_tensor=True)
    score = float(util.cos_sim(emb_a, emb_b)[0][0])
    return round(max(0.0, min(score, 1.0)) * 100, 1)


# ──────────────────────────────────────────────
# COMPOSITE SCORING
# ──────────────────────────────────────────────
def compute_composite(
    semantic: float,
    resume_skills: set,
    job_skills: set,
    resume_years: int | None,
    job_years: int | None,
) -> dict:
    """
    Weighted composite:
      45% semantic similarity
      35% skill overlap (Jaccard on matched sets)
      20% experience alignment
    """
    # Skill score
    if job_skills:
        matched = resume_skills & job_skills
        skill_score = len(matched) / len(job_skills) * 100
    else:
        skill_score = 50.0  # no skills listed in JD → neutral

    # Experience score
    if resume_years is not None and job_years is not None:
        diff = resume_years - job_years
        if diff >= 0:
            exp_score = min(100.0, 80 + diff * 4)
        else:
            exp_score = max(0.0, 100 + diff * 15)
    else:
        exp_score = 55.0  # unknown → neutral

    composite = 0.45 * semantic + 0.35 * skill_score + 0.20 * exp_score

    return {
        "semantic": round(semantic, 1),
        "skill": round(skill_score, 1),
        "experience": round(exp_score, 1),
        "composite": round(composite, 1),
    }


# ──────────────────────────────────────────────
# ATS KEYWORD DENSITY
# ──────────────────────────────────────────────
def ats_keyword_density(resume_text: str, job_text: str) -> float:
    """% of JD content words found in resume (simple ATS proxy)."""
    stop = {"a","an","the","and","or","in","on","of","for","to","is","are","be",
            "with","that","this","we","you","our","their","will","have","has"}
    job_words = {w for w in re.findall(r"\b[a-z]{3,}\b", job_text.lower()) if w not in stop}
    res_words  = set(re.findall(r"\b[a-z]{3,}\b", resume_text.lower()))
    if not job_words:
        return 0.0
    return round(len(job_words & res_words) / len(job_words) * 100, 1)


# ──────────────────────────────────────────────
# VERDICT HELPER
# ──────────────────────────────────────────────
def verdict(score: float) -> tuple[str, str, str]:
    if score >= 75:
        return "Strong Match 🟢", "verdict-high", "You are a strong candidate for this role. Focus on tailoring your language to mirror the JD."
    if score >= 55:
        return "Good Potential 🟡", "verdict-mid", "Solid alignment exists. Addressing skill gaps and adding relevant keywords will boost your profile significantly."
    return "Needs Improvement 🔴", "verdict-low", "The current fit is limited. Consider upskilling in the key gap areas before applying."


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎯 CareerFit Pro")
    st.caption("AI-powered semantic resume × job matching")
    st.divider()
    st.markdown("### How it works")
    st.markdown("""
1. Upload your **resume** (PDF) or paste text
2. Add one or more **job descriptions**
3. Hit **Analyze** to get:
   - Semantic match score
   - Skill gap breakdown
   - ATS keyword density
   - Experience alignment
   - Improvement recommendations
""")
    st.divider()
    st.markdown("### Model Info")
    st.caption("Semantic: `all-MiniLM-L6-v2`")
    st.caption("NER: spaCy `en_core_web_sm`")
    st.caption("Skill taxonomy: 80+ technologies")


# ──────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────
st.markdown("# 🎯 CareerFit Pro")
st.caption("Semantic Resume × Job Description Intelligence Engine")
st.divider()

embedder, nlp = load_models()

# ── INPUT SECTION ──
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("### 📄 Resume Input")
    input_mode = st.radio("Input method", ["Upload PDF", "Paste Text"], horizontal=True)

    resume_text = ""
    if input_mode == "Upload PDF":
        pdf_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
        if pdf_file:
            with st.spinner("Extracting text from PDF…"):
                resume_text = extract_pdf_text(pdf_file)
            st.success(f"Extracted {len(resume_text.split())} words from PDF")
            with st.expander("Preview extracted text"):
                st.text(resume_text[:1200] + ("…" if len(resume_text) > 1200 else ""))
    else:
        resume_text = st.text_area(
            "Paste resume text",
            height=280,
            placeholder="Paste your full resume here — skills, experience, education, projects…",
        )

with col_right:
    st.markdown("### 💼 Job Descriptions")
    num_jobs = st.number_input("Number of job descriptions", min_value=1, max_value=5, value=1, step=1)
    job_inputs: list[str] = []
    for i in range(int(num_jobs)):
        jd = st.text_area(
            f"Job Description {i+1}",
            height=120,
            key=f"jd_{i}",
            placeholder=f"Paste job description {i+1} here…",
        )
        job_inputs.append(jd)

st.divider()

# ── ANALYZE BUTTON ──
analyze = st.button("⚡ Analyze Career Fit", type="primary", use_container_width=True)

if analyze:
    # ── VALIDATION ──
    if len(resume_text.strip()) < 80:
        st.error("Please provide more resume content (at least a few sentences).")
        st.stop()

    filled_jobs = [(i + 1, jd.strip()) for i, jd in enumerate(job_inputs) if len(jd.strip()) > 40]
    if not filled_jobs:
        st.error("Please enter at least one job description.")
        st.stop()

    # ── GLOBAL RESUME ANALYSIS ──
    clean_resume = clean_text(resume_text)
    resume_skills_dict = extract_skills(resume_text)
    resume_skills_flat = flat_skills(resume_skills_dict)
    resume_years = extract_years(resume_text)
    resume_edu = extract_education(resume_text)
    resume_seniority = classify_seniority(resume_years, resume_text)

    # ── PROFILE SUMMARY ──
    st.markdown("## 📊 Profile Summary")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.metric("Total Skills Detected", len(resume_skills_flat))
    with p2:
        st.metric("Experience", f"{resume_years} yrs" if resume_years else "—")
    with p3:
        st.metric("Seniority Level", resume_seniority)
    with p4:
        st.metric("Education", resume_edu)

    # Skills by category
    if resume_skills_dict:
        with st.expander("🔬 Your Skill Breakdown by Category", expanded=True):
            for cat, skills in resume_skills_dict.items():
                chips = " ".join(f'<span class="chip-neutral">{s}</span>' for s in skills)
                st.markdown(f"**{cat}:** {chips}", unsafe_allow_html=True)
    else:
        st.warning("No structured skills detected. Check that your resume includes technology keywords.")

    st.divider()

    # ── PER-JOB ANALYSIS ──
    st.markdown(f"## 🎯 Job Match Results ({len(filled_jobs)} role{'s' if len(filled_jobs) > 1 else ''})")

    all_scores: list[float] = []

    for idx, jd_text in filled_jobs:
        clean_jd = clean_text(jd_text)
        jd_skills_dict = extract_skills(jd_text)
        jd_skills_flat = flat_skills(jd_skills_dict)
        jd_years = extract_years(jd_text)

        # Compute scores
        with st.spinner(f"Computing match for Job {idx}…"):
            semantic = compute_semantic_score(embedder, clean_resume, clean_jd)
            scores = compute_composite(semantic, resume_skills_flat, jd_skills_flat, resume_years, jd_years)
            ats = ats_keyword_density(resume_text, jd_text)

        all_scores.append(scores["composite"])

        matched = resume_skills_flat & jd_skills_flat
        gaps = jd_skills_flat - resume_skills_flat
        extras = resume_skills_flat - jd_skills_flat

        verdict_label, verdict_class, verdict_msg = verdict(scores["composite"])

        with st.container():
            st.markdown(f"### Job {idx}")
            st.markdown(
                f'<div class="{verdict_class}"><b>{verdict_label}</b> — {verdict_msg}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            # Score cards
            c1, c2, c3, c4, c5 = st.columns(5)
            card_data = [
                ("Composite Match", scores["composite"], "#00d4ff"),
                ("Semantic Score",  scores["semantic"],  "#7c3aed"),
                ("Skill Overlap",   scores["skill"],     "#10b981"),
                ("Experience Fit",  scores["experience"],"#f59e0b"),
                ("ATS Density",     ats,                 "#06b6d4"),
            ]
            for col, (label, val, color) in zip([c1, c2, c3, c4, c5], card_data):
                with col:
                    st.markdown(
                        f"""<div class="score-card" style="--accent:{color}">
                          <div class="score-num">{val}<span style="font-size:18px">%</span></div>
                          <div class="score-label">{label}</div>
                        </div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown("<br>", unsafe_allow_html=True)

            # Progress bars
            st.markdown("**Score Breakdown**")
            for label, val in [
                ("Composite", scores["composite"]),
                ("Semantic",  scores["semantic"]),
                ("Skill Fit", scores["skill"]),
                ("Experience",scores["experience"]),
                ("ATS",       ats),
            ]:
                st.markdown(f"<small>{label}</small>", unsafe_allow_html=True)
                st.progress(int(val))

            st.markdown("<br>", unsafe_allow_html=True)

            # Skill gap analysis
            skill_col1, skill_col2, skill_col3 = st.columns(3)
            with skill_col1:
                st.markdown("**✅ Matched Skills**")
                if matched:
                    st.markdown(
                        " ".join(f'<span class="chip-match">{s}</span>' for s in sorted(matched)),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("None detected")

            with skill_col2:
                st.markdown("**❌ Skill Gaps**")
                if gaps:
                    st.markdown(
                        " ".join(f'<span class="chip-gap">{s}</span>' for s in sorted(gaps)),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("No gaps — great fit!")

            with skill_col3:
                st.markdown("**➕ Your Extra Skills**")
                if extras:
                    st.markdown(
                        " ".join(f'<span class="chip-neutral">{s}</span>' for s in sorted(extras)),
                        unsafe_allow_html=True,
                    )
                else:
                    st.caption("None")

            # Experience comparison
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**🕐 Experience Alignment**")
            exp_col1, exp_col2, exp_col3 = st.columns(3)
            with exp_col1:
                st.metric("Your Experience", f"{resume_years} yrs" if resume_years else "Not specified")
            with exp_col2:
                st.metric("Job Requires", f"{jd_years} yrs" if jd_years else "Not specified")
            with exp_col3:
                if resume_years and jd_years:
                    delta = resume_years - jd_years
                    st.metric("Delta", f"{delta:+d} yrs", delta=delta)
                else:
                    st.metric("Delta", "—")

            # Recommendations
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("**💡 Recommendations**")
            recs = []
            if gaps:
                top_gaps = list(gaps)[:3]
                recs.append(f"Acquire or highlight: **{', '.join(top_gaps)}** — these are required but missing from your resume.")
            if ats < 50:
                recs.append("Your **ATS keyword density is low**. Mirror the exact phrasing from the job description more closely.")
            if scores["semantic"] < 55:
                recs.append("Rewrite your **summary/objective** section to align closely with this role's language and priorities.")
            if resume_years is not None and jd_years is not None and resume_years < jd_years:
                recs.append(f"The role requires **{jd_years} years** of experience. Emphasize complexity and impact of your {resume_years}-year career.")
            if not recs:
                recs.append("Your profile is well-aligned. Tailor your cover letter to reinforce matched skills and quantify impact.")

            for i, rec in enumerate(recs, 1):
                st.markdown(f"{i}. {rec}")

            st.divider()

    # ── OVERALL SUMMARY ──
    if len(all_scores) > 1:
        st.markdown("## 📈 Cross-Job Summary")
        best_idx = int(np.argmax(all_scores)) + 1
        avg = round(np.mean(all_scores), 1)
        st.success(f"🏆 **Best Match:** Job {best_idx} ({all_scores[best_idx-1]}%) — Average across all roles: **{avg}%**")

        rank_data = sorted(
            [(f"Job {filled_jobs[i][0]}", s) for i, s in enumerate(all_scores)],
            key=lambda x: x[1], reverse=True,
        )
        st.markdown("**Ranking (best to lowest):**")
        for rank, (jname, sc) in enumerate(rank_data, 1):
            bar = "█" * int(sc / 5)
            st.markdown(f"`{rank}.` **{jname}** — {sc}% `{bar}`")

    # ── EXPORT ──
    st.divider()
    st.markdown("### 💾 Export Results")
    export = {
        "resume_profile": {
            "skills": {cat: skills for cat, skills in resume_skills_dict.items()},
            "years_experience": resume_years,
            "seniority": resume_seniority,
            "education": resume_edu,
        },
        "job_results": [
            {
                "job_number": idx,
                "composite_match": all_scores[i],
            }
            for i, (idx, _) in enumerate(filled_jobs)
        ],
    }
    st.download_button(
        label="📥 Download JSON Report",
        data=json.dumps(export, indent=2),
        file_name="careerfit_report.json",
        mime="application/json",
    )
