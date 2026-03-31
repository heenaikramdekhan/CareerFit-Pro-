# 🎯 CareerFit Pro — AI Resume & Job Matching System

A semantic AI-powered system that analyzes your resume against job descriptions and provides a detailed compatibility report — using real **sentence-transformer embeddings** rather than simple keyword overlap.

---

## Overview

CareerFit Pro goes beyond traditional TF-IDF matching. It combines:

- **Semantic similarity** (via `all-MiniLM-L6-v2` embeddings) to understand *meaning*, not just word presence
- **Structured skill taxonomy** extraction across 80+ technologies organized into 6 categories
- **Experience & seniority classification** using regex + heuristics
- **ATS keyword density** scoring (simulates Applicant Tracking System screening)
- **Composite weighted scoring** that blends semantic, skill, and experience signals
- **Multi-job comparison** to rank up to 5 roles simultaneously

---

## How It Differs from TF-IDF Systems

| Feature | TF-IDF Baseline | CareerFit Pro |
|---|---|---|
| Similarity method | Term frequency vectors | Sentence-transformer embeddings |
| Understanding | Keyword overlap | Semantic meaning |
| Skill detection | Bag-of-words | Curated taxonomy (80+ skills) |
| Experience parsing | Simple regex | Regex + seniority classification |
| Output | Single score | 5-dimensional scorecard |
| Multi-job support | ❌ | ✅ Up to 5 JDs simultaneously |
| ATS simulation | ❌ | ✅ |
| Export | ❌ | ✅ JSON report |

---

## Project Structure

```
.
├── app.py              # Main Streamlit application (all logic + UI)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## Installation

### Prerequisites
- Python 3.10+
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/heenaikramdekhan/CareerFit-Pro-.git
cd CareerFit-Pro-

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy language model
python -m spacy download en_core_web_sm

# 5. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

> **Note:** The first launch downloads the `all-MiniLM-L6-v2` model (~90MB) automatically. Subsequent launches use the cached version.

---

## Usage

### Step 1 — Provide Your Resume
Choose between:
- **Upload PDF** — the app uses `pdfplumber` to extract text page by page
- **Paste Text** — paste raw resume text directly

### Step 2 — Add Job Descriptions
Paste 1–5 job descriptions in the provided text areas.

### Step 3 — Analyze
Click **⚡ Analyze Career Fit**. Within seconds you'll see:

#### Profile Summary
- Total skills detected across 6 categories
- Years of experience & seniority level
- Highest education level detected

#### Per-Job Scorecard (5 dimensions)
| Metric | Weight | What it measures |
|---|---|---|
| **Composite Match** | —     | Weighted blend of all signals |
| **Semantic Score**  | 45%   | Embedding-based meaning similarity |
| **Skill Overlap**   | 35%   | Jaccard overlap on matched skill sets |
| **Experience Fit**  | 20%   | Years of experience vs. requirement |
| **ATS Density**     | info  | % of JD keywords present in resume |

#### Skill Gap Analysis
- ✅ **Matched Skills** — skills you have that the JD requires
- ❌ **Skill Gaps** — required skills not found in your resume
- ➕ **Extra Skills** — skills you have beyond the JD requirements

#### Recommendations
Automatically generated, specific action items based on your gaps.

#### Cross-Job Ranking (when multiple JDs provided)
Ranks all roles by composite score and highlights the best fit.

### Step 4 — Export
Download a **JSON report** summarizing your profile and all match scores for record-keeping.

---

## Scoring Logic

```
Composite = 0.45 × Semantic + 0.35 × Skill_Overlap + 0.20 × Experience_Fit

Semantic Score     : cosine similarity of sentence-transformer embeddings
Skill Overlap      : |resume_skills ∩ jd_skills| / |jd_skills| × 100
Experience Fit     : 100% if resume_years ≥ jd_years; decreases linearly if under
ATS Density        : |jd_content_words ∩ resume_words| / |jd_content_words| × 100
```

**Composite interpretation:**
- **≥ 75%** → Strong Match 🟢
- **55–74%** → Good Potential 🟡
- **< 55%** → Needs Improvement 🔴

---

## Skill Taxonomy (80+ technologies)

| Category | Examples |
|---|---|
| Languages | Python, Java, TypeScript, Go, Rust, SQL… |
| Web & APIs | React, FastAPI, Node.js, GraphQL, REST API… |
| Data & ML | PyTorch, scikit-learn, Hugging Face, Airflow, MLOps… |
| Cloud & DevOps | AWS, Docker, Kubernetes, Terraform, CI/CD… |
| Databases | PostgreSQL, MongoDB, Redis, Snowflake, BigQuery… |
| Practices | Agile, TDD, Microservices, System Design, Git… |

---

## Extending the Project

### Add more skills
Edit the `SKILL_TAXONOMY` dict in `app.py` — each category maps to a list of skill strings.

### Swap the embedding model
Replace `"all-MiniLM-L6-v2"` in `load_models()` with any model from [SBERT.net](https://www.sbert.net/docs/pretrained_models.html), e.g., `"all-mpnet-base-v2"` for higher accuracy.

### Add cover letter generation
Integrate an LLM API (OpenAI, Anthropic) to auto-generate tailored cover letters based on matched skills and gaps.

### Batch resume screening (recruiter mode)
Upload multiple resumes and one JD to rank candidates — invert the matching direction.

---

## Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `pdfplumber` | PDF text extraction |
| `sentence-transformers` | Semantic embedding model |
| `spacy` | NLP pipeline / NER |
| `numpy` | Numerical operations |
| `torch` | Backend for sentence-transformers |

---

## License

MIT — free to use, modify, and distribute.
