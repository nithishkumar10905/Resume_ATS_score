import os
import re
import io
import json
import sqlite3
import logging
import datetime
from collections import Counter
from flask import Flask, render_template, request, jsonify

# ── Optional/Heavy imports with graceful fallbacks ──────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
except Exception:
    SPACY_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from pdfminer.high_level import extract_text as pdfminer_extract
    from pdfminer.layout import LAParams
    PDFMINER_AVAILABLE = True
except ImportError:
    PDFMINER_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('app.log'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Gemini API Setup ─────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
logger_init = logging.getLogger(__name__)
if GEMINI_API_KEY:
    logger_init.info(f"GEMINI_API_KEY is set: {GEMINI_API_KEY[:20]}...")
else:
    logger_init.info("GEMINI_API_KEY is not set")
    
gemini_model = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        logger_init.info(f"Configuring Gemini with API key")
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        logger.info("Gemini AI initialized successfully")
    except Exception as e:
        logger_init.warning(f"Gemini init failed: {e}")
        logger.warning(f"Gemini init failed: {e}")

# ── SBERT Model ──────────────────────────────────────────────────────────────
sbert_model = None
if SBERT_AVAILABLE:
    try:
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SBERT model loaded successfully")
    except Exception as e:
        logger.warning(f"SBERT load failed: {e}")

# ── SQLite Database Setup ────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "ats_history.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            resume_snippet TEXT,
            job_snippet TEXT,
            match_score REAL,
            matched_skills TEXT,
            missing_skills TEXT,
            ai_feedback TEXT,
            ats_score REAL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── NLTK Setup ───────────────────────────────────────────────────────────────
for _res, _cat in [('punkt', 'tokenizers'), ('punkt_tab', 'tokenizers'), ('stopwords', 'corpora')]:
    try:
        nltk.data.find(f'{_cat}/{_res}')
    except (LookupError, OSError):
        nltk.download(_res, quiet=True)

# ── Skills Database ──────────────────────────────────────────────────────────
SKILL_SYNONYMS = {
    'js': 'javascript', 'javascript': 'javascript',
    'react': 'react', 'reactjs': 'react', 'react.js': 'react',
    'node': 'nodejs', 'nodejs': 'nodejs', 'node.js': 'nodejs',
    'python': 'python', 'py': 'python',
    'java': 'java', 'golang': 'go', 'go': 'go',
    'ml': 'machine learning', 'machine learning': 'machine learning',
    'ai': 'artificial intelligence', 'artificial intelligence': 'artificial intelligence',
    'deep learning': 'deep learning', 'dl': 'deep learning',
    'nlp': 'nlp', 'natural language processing': 'nlp',
    'cv': 'computer vision', 'computer vision': 'computer vision',
    'aws': 'aws', 'amazon web services': 'aws',
    'gcp': 'gcp', 'google cloud': 'gcp',
    'azure': 'azure', 'microsoft azure': 'azure',
    'docker': 'docker', 'containerization': 'docker',
    'k8s': 'kubernetes', 'kubernetes': 'kubernetes',
    'sql': 'sql', 'mysql': 'sql', 'postgresql': 'sql', 'postgres': 'sql',
    'nosql': 'nosql', 'mongodb': 'nosql',
    'redis': 'redis', 'elasticsearch': 'elasticsearch',
    'ci/cd': 'ci/cd', 'continuous integration': 'ci/cd', 'jenkins': 'ci/cd',
    'git': 'git', 'github': 'git', 'version control': 'git',
    'pytorch': 'pytorch', 'tensorflow': 'tensorflow', 'keras': 'keras',
    'pandas': 'pandas', 'numpy': 'numpy', 'scikit-learn': 'scikit-learn', 'sklearn': 'scikit-learn',
    'flask': 'flask', 'django': 'django', 'fastapi': 'fastapi', 'express': 'express',
    'typescript': 'typescript', 'ts': 'typescript',
    'vue': 'vuejs', 'vuejs': 'vuejs', 'angular': 'angular',
    'graphql': 'graphql', 'rest': 'rest api', 'rest api': 'rest api',
    'agile': 'agile', 'scrum': 'scrum', 'kanban': 'kanban',
    'linux': 'linux', 'unix': 'linux',
    'terraform': 'terraform', 'ansible': 'ansible',
    'microservices': 'microservices', 'serverless': 'serverless',
    'data science': 'data science', 'data analysis': 'data analysis',
    'power bi': 'powerbi', 'powerbi': 'powerbi', 'tableau': 'tableau',
    'excel': 'excel', 'spark': 'apache spark', 'apache spark': 'apache spark',
    'hadoop': 'hadoop', 'kafka': 'kafka',
    'c++': 'c++', 'cpp': 'c++', 'c#': 'c#', 'csharp': 'c#',
    'rust': 'rust', 'swift': 'swift', 'kotlin': 'kotlin',
    'leadership': 'leadership', 'communication': 'communication',
    'teamwork': 'teamwork', 'problem solving': 'problem solving',
    'project management': 'project management',
}

SKILL_RECOMMENDATIONS = {
    'python': {'courses': ['Python for Everybody (Coursera)', 'Complete Python Bootcamp (Udemy)'], 'projects': ['Build a REST API', 'Web scraper project'], 'certifications': ['PCEP - Python Certified Entry-Level']},
    'javascript': {'courses': ['JavaScript Complete Guide (Udemy)', 'freeCodeCamp JS Track'], 'projects': ['Todo app', 'Interactive dashboard'], 'certifications': ['freeCodeCamp JS Certification']},
    'react': {'courses': ['React Complete Guide (Udemy)', 'React Official Docs'], 'projects': ['Portfolio website', 'Weather app with API'], 'certifications': ['Meta Front-End Developer Certificate']},
    'aws': {'courses': ['AWS Solutions Architect', 'AWS Cloud Practitioner Essentials'], 'projects': ['Deploy app on EC2', 'Serverless Lambda app'], 'certifications': ['AWS Cloud Practitioner', 'AWS Solutions Architect']},
    'docker': {'courses': ['Docker Mastery (Udemy)', 'Docker Official Docs'], 'projects': ['Containerize a web app', 'Multi-container app with compose'], 'certifications': ['Docker Certified Associate']},
    'machine learning': {'courses': ['ML by Andrew Ng (Coursera)', 'Fast.ai Practical DL'], 'projects': ['Sentiment classifier', 'Recommendation system'], 'certifications': ['TensorFlow Developer', 'AWS ML Specialty']},
    'kubernetes': {'courses': ['Kubernetes Mastery', 'CKA Prep Course'], 'projects': ['Deploy microservices on K8s', 'Helm chart creation'], 'certifications': ['CKA - Certified Kubernetes Administrator']},
    'sql': {'courses': ['SQL Bootcamp (Udemy)', 'Mode Analytics SQL Tutorial'], 'projects': ['Database design project', 'Analytics dashboard'], 'certifications': ['Oracle SQL Certified Associate']},
    'data science': {'courses': ['IBM Data Science (Coursera)', 'Kaggle Courses'], 'projects': ['EDA on public dataset', 'Predictive model'], 'certifications': ['IBM Data Science Professional']},
    'tensorflow': {'courses': ['TensorFlow Developer Certificate', 'Deep Learning Specialization (Coursera)'], 'projects': ['Image classification', 'Text generation'], 'certifications': ['TensorFlow Developer Certificate']},
}

# ── Text Extraction ──────────────────────────────────────────────────────────
def extract_text_from_pdf(file_obj):
    """Try pdfminer first (better accuracy), fallback to PyPDF2."""
    if PDFMINER_AVAILABLE:
        try:
            file_obj.seek(0)
            text = pdfminer_extract(file_obj, laparams=LAParams())
            if text and text.strip():
                return text
        except Exception as e:
            logger.warning(f"pdfminer failed: {e}")

    if PYPDF2_AVAILABLE:
        try:
            file_obj.seek(0)
            reader = PyPDF2.PdfReader(file_obj)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            logger.warning(f"PyPDF2 failed: {e}")

    return ""

def extract_text_from_docx(file_obj):
    if not DOCX_AVAILABLE:
        return ""
    try:
        doc = Document(file_obj)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception as e:
        logger.warning(f"docx extract failed: {e}")
        return ""

def extract_text_from_file(file):
    filename = file.filename.lower()
    if filename.endswith('.pdf'):
        return extract_text_from_pdf(file)
    elif filename.endswith(('.docx', '.doc')):
        return extract_text_from_docx(file)
    return ""

# ── NLP Preprocessing ────────────────────────────────────────────────────────
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    try:
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        return ' '.join([w for w in tokens if w not in stop_words and len(w) > 2])
    except Exception:
        return ' '.join([w for w in text.split() if len(w) > 2])

def normalize_skill(skill):
    return SKILL_SYNONYMS.get(skill.lower().strip(), skill.lower().strip())

# ── Skill Extraction ─────────────────────────────────────────────────────────
SKILLS_PATTERN = re.compile(
    r'\b(?:python|java(?:script|)|typescript|golang|go|rust|swift|kotlin|c\+\+|c#|'
    r'react(?:js|\.js|)|angular|vue(?:js|)|node(?:js|\.js|)|express|django|flask|fastapi|'
    r'html|css|graphql|rest\s*api|'
    r'machine\s*learning|deep\s*learning|nlp|natural\s*language\s*processing|'
    r'computer\s*vision|data\s*science|data\s*analysis|'
    r'tensorflow|pytorch|keras|scikit-learn|sklearn|pandas|numpy|'
    r'apache\s*spark|hadoop|kafka|elasticsearch|'
    r'sql|mysql|postgresql|postgres|nosql|mongodb|redis|'
    r'aws|gcp|azure|docker|kubernetes|k8s|terraform|ansible|'
    r'ci/cd|continuous\s*integration|jenkins|git(?:hub|)|'
    r'agile|scrum|kanban|microservices|serverless|'
    r'power\s*bi|powerbi|tableau|excel|'
    r'linux|unix|'
    r'leadership|communication|teamwork|problem\s*solving|project\s*management)\b',
    re.IGNORECASE
)

def extract_skills_spacy(text):
    """Use spaCy NER + pattern matching for skill extraction."""
    skills = set()
    # Pattern-based extraction (always applies)
    for m in SKILLS_PATTERN.finditer(text):
        skills.add(normalize_skill(m.group(0)))

    # spaCy entity extraction for ORG/PRODUCT names that could be skills
    if SPACY_AVAILABLE:
        try:
            doc = nlp(text[:5000])  # limit for performance
            for ent in doc.ents:
                if ent.label_ in ('ORG', 'PRODUCT', 'WORK_OF_ART'):
                    candidate = ent.text.lower().strip()
                    if candidate in SKILL_SYNONYMS:
                        skills.add(normalize_skill(candidate))
        except Exception:
            pass

    return skills

# ── Semantic Similarity ──────────────────────────────────────────────────────
def compute_semantic_similarity(text1, text2):
    """SBERT cosine similarity; fallback to TF-IDF if unavailable."""
    if sbert_model:
        try:
            emb1 = sbert_model.encode([text1])
            emb2 = sbert_model.encode([text2])
            sim = cosine_similarity(emb1, emb2)[0][0]
            return float(sim)
        except Exception as e:
            logger.warning(f"SBERT similarity failed: {e}")

    # TF-IDF fallback
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([
            preprocess_text(text1), preprocess_text(text2)
        ])
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
        return float(cos_sim(tfidf[0:1], tfidf[1:2])[0][0])
    except Exception:
        return 0.0

# ── Experience / Seniority ───────────────────────────────────────────────────
def extract_experience_years(text):
    patterns = [
        r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?experience',
        r'experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
        r'(\d+)\+?\s*(?:years?|yrs?)\s+in',
    ]
    for p in patterns:
        m = re.search(p, text.lower())
        if m:
            return int(m.group(1))
    return 0

def extract_seniority_level(text):
    t = text.lower()
    if any(w in t for w in ['intern', 'internship', 'trainee']):
        return 'Intern'
    if any(w in t for w in ['junior', 'entry level', 'entry-level', 'associate']):
        return 'Junior'
    if any(w in t for w in ['senior', 'sr.', 'lead', 'principal', 'staff']):
        return 'Senior'
    if any(w in t for w in ['manager', 'director', 'head of', 'vp', 'chief']):
        return 'Management'
    return 'Mid-Level'

# ── Resume Section Parser ────────────────────────────────────────────────────
def parse_resume_sections(text):
    sections = {k: '' for k in ['skills', 'experience', 'education', 'projects', 'summary']}
    current = None
    for line in text.split('\n'):
        ll = line.lower().strip()
        if any(w in ll for w in ['skill', 'technical skill', 'core competenc', 'technologies']):
            current = 'skills'
        elif any(w in ll for w in ['experience', 'work history', 'employment', 'career']):
            current = 'experience'
        elif any(w in ll for w in ['education', 'academic', 'qualification', 'degree']):
            current = 'education'
        elif any(w in ll for w in ['project', 'personal project', 'portfolio']):
            current = 'projects'
        elif any(w in ll for w in ['summary', 'profile', 'objective', 'about', 'overview']):
            current = 'summary'
        elif current and line.strip():
            sections[current] += line + ' '
    return sections

# ── Skill Priority Classification ────────────────────────────────────────────
def classify_skill_priority(skill, job_text):
    job_lower = job_text.lower()
    sl = re.escape(skill.lower())
    if any(re.search(p, job_lower) for p in [
        rf'required.*{sl}', rf'must have.*{sl}',
        rf'{sl}.*required', rf'{sl}.*mandatory',
        rf'essential.*{sl}', rf'requirements?.*{sl}'
    ]):
        return 'mandatory'
    if any(re.search(p, job_lower) for p in [
        rf'nice to have.*{sl}', rf'preferred.*{sl}',
        rf'bonus.*{sl}', rf'plus.*{sl}'
    ]):
        return 'optional'
    return 'optional'

# ── ATS Compliance Checker ───────────────────────────────────────────────────
def check_ats_compliance(text):
    issues = []
    score = 100
    wc = len(text.split())
    if wc < 200:
        issues.append(f"Resume too short ({wc} words; aim for 300–700)")
        score -= 15
    elif wc > 1000:
        issues.append(f"Resume too long ({wc} words; aim for 300–700)")
        score -= 10
    if not re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text):
        issues.append("Missing email address")
        score -= 10
    if not re.search(r'\b\d[\d\s\-().+]{7,}\d\b', text):
        issues.append("Missing phone number")
        score -= 10
    for sec in ['experience', 'education', 'skill']:
        if sec not in text.lower():
            issues.append(f"Missing '{sec}' section header")
            score -= 10
    if re.search(r'[^\x00-\x7F]+', text):
        issues.append("Contains non-ASCII characters (may confuse older ATS)")
        score -= 5
    table_hints = ['|', '\t\t\t']
    if any(h in text for h in table_hints):
        issues.append("Tables/columns detected — prefer single-column layout")
        score -= 5
    return max(0, score), issues

# ── Gemini AI Feedback ───────────────────────────────────────────────────────
def get_gemini_feedback(resume_text, job_text):
    """Call Gemini API and return structured AI feedback."""
    if not gemini_model:
        return {
            "available": False,
            "message": "Set the GEMINI_API_KEY environment variable to enable AI feedback.",
            "suggestions": [],
            "rewrite_example": "",
            "keyword_gaps": []
        }

    prompt = f"""You are an expert ATS and resume consultant. Analyze the following resume against the job description and return ONLY valid JSON (no markdown, no code fences).

Resume (first 1500 chars):
{resume_text[:1500]}

Job Description (first 800 chars):
{job_text[:800]}

Return exactly this JSON structure:
{{
  "overall_assessment": "2-3 sentence summary of the candidate's fit",
  "suggestions": [
    "Specific actionable improvement 1",
    "Specific actionable improvement 2",
    "Specific actionable improvement 3"
  ],
  "rewrite_example": "One rewritten bullet point that better matches the job",
  "keyword_gaps": ["keyword1", "keyword2", "keyword3"],
  "phrasing_tips": "One specific phrasing recommendation"
}}"""

    try:
        response = gemini_model.generate_content(prompt)
        raw = response.text.strip()
        # Strip markdown code fences if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)
        data = json.loads(raw)
        data["available"] = True
        return data
    except json.JSONDecodeError as e:
        logger.warning(f"Gemini JSON parse error: {e} — raw: {raw[:200]}")
        return {
            "available": True,
            "overall_assessment": response.text[:500],
            "suggestions": ["Could not parse structured feedback. See overall assessment."],
            "rewrite_example": "",
            "keyword_gaps": [],
            "phrasing_tips": ""
        }
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return {
            "available": False,
            "message": f"AI feedback unavailable: {str(e)}",
            "suggestions": [],
            "rewrite_example": "",
            "keyword_gaps": []
        }

# ── Learning Recommendations ─────────────────────────────────────────────────
def generate_recommendations(missing_skills):
    recs = {}
    for skill in list(missing_skills)[:5]:
        norm = normalize_skill(skill)
        recs[skill] = SKILL_RECOMMENDATIONS.get(norm, {
            'courses': [f'Search "{skill}" on Coursera, Udemy, or Pluralsight'],
            'projects': [f'Build a project using {skill}', f'Contribute to open-source {skill} projects'],
            'certifications': [f'Look for professional {skill} certifications']
        })
    return recs

# ── Weighted Scoring Formula ─────────────────────────────────────────────────
def calculate_weighted_score(semantic_sim, skill_match_rate, experience_years_match,
                              seniority_match, education_match, ats_score):
    """
    Final Score = 0.35×semantic + 0.25×skill_match + 0.20×experience + 0.10×education + 0.10×ats
    """
    exp_score = 1.0 if experience_years_match else 0.5
    sen_score = 1.0 if seniority_match else 0.7
    experience_score = (exp_score * 0.6 + sen_score * 0.4)

    final = (
        0.35 * semantic_sim +
        0.25 * skill_match_rate +
        0.20 * experience_score +
        0.10 * education_match +
        0.10 * (ats_score / 100.0)
    )
    return round(min(1.0, max(0.0, final)) * 100, 2)

def detect_education_match(resume_text, job_text):
    """Simple heuristic for education alignment."""
    job_lower = job_text.lower()
    resume_lower = resume_text.lower()
    if 'phd' in job_lower or 'doctorate' in job_lower:
        return 1.0 if ('phd' in resume_lower or 'doctorate' in resume_lower) else 0.4
    if 'master' in job_lower or "master's" in job_lower:
        if 'master' in resume_lower:
            return 1.0
        if 'bachelor' in resume_lower:
            return 0.7
        return 0.4
    if 'bachelor' in job_lower or "degree" in job_lower:
        if any(w in resume_lower for w in ['bachelor', 'master', 'phd', 'degree', 'b.sc', 'b.e', 'b.tech']):
            return 1.0
        return 0.6
    return 0.8  # job doesn't specify, neutral

# ── Core Match Calculator ────────────────────────────────────────────────────
def calculate_match(resume_text, job_text):
    # 1. Semantic similarity (SBERT or TF-IDF)
    semantic_sim = compute_semantic_similarity(resume_text, job_text)

    # 2. Skills
    resume_skills = extract_skills_spacy(resume_text)
    job_skills = extract_skills_spacy(job_text)
    matched_skills = sorted(resume_skills & job_skills)
    missing_skills_set = job_skills - resume_skills
    skill_match_rate = len(matched_skills) / max(len(job_skills), 1)

    # Classify as mandatory/optional
    missing_mandatory, missing_optional = [], []
    for s in missing_skills_set:
        (missing_mandatory if classify_skill_priority(s, job_text) == 'mandatory'
         else missing_optional).append(s)

    # 3. Experience
    resume_years = extract_experience_years(resume_text)
    required_years = extract_experience_years(job_text)
    resume_level = extract_seniority_level(resume_text)
    required_level = extract_seniority_level(job_text)
    years_match = (resume_years >= required_years) if required_years > 0 else True
    seniority_match = (resume_level == required_level)

    # 4. Education
    edu_match = detect_education_match(resume_text, job_text)

    # 5. ATS compliance
    ats_score, ats_issues = check_ats_compliance(resume_text)

    # 6. Weighted final score
    final_score = calculate_weighted_score(
        semantic_sim, skill_match_rate,
        years_match, seniority_match,
        edu_match, ats_score
    )

    # 7. Section-level similarities
    resume_sections = parse_resume_sections(resume_text)
    section_scores = {}
    for sec_name, sec_text in resume_sections.items():
        if sec_text.strip():
            section_scores[sec_name] = round(
                compute_semantic_similarity(sec_text, job_text) * 100, 2
            )

    # 8. Learning recommendations
    recommendations = generate_recommendations(missing_skills_set)

    return {
        'match_percentage': final_score,
        'score_breakdown': {
            'semantic_similarity': round(semantic_sim * 100, 2),
            'skill_match_rate': round(skill_match_rate * 100, 2),
            'ats_score': ats_score,
        },
        'matched_skills': matched_skills,
        'missing_skills': {
            'mandatory': sorted(missing_mandatory),
            'optional': sorted(missing_optional)
        },
        'experience_match': {
            'resume_years': resume_years,
            'resume_level': resume_level,
            'required_years': required_years,
            'required_level': required_level,
            'match': years_match
        },
        'ats_compliance': {
            'score': ats_score,
            'issues': ats_issues
        },
        'section_scores': section_scores,
        'recommendations': recommendations,
        'skill_counts': {
            'matched': len(matched_skills),
            'missing': len(missing_skills_set),
            'total_required': len(job_skills)
        }
    }

# ── Database helpers ─────────────────────────────────────────────────────────
def save_to_history(resume_text, job_text, result, ai_feedback):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            INSERT INTO analyses
            (timestamp, resume_snippet, job_snippet, match_score, matched_skills, missing_skills, ai_feedback, ats_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.datetime.now().isoformat(),
            resume_text[:300],
            job_text[:300],
            result['match_percentage'],
            json.dumps(result['matched_skills']),
            json.dumps(result['missing_skills']),
            json.dumps(ai_feedback),
            result['ats_compliance']['score']
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"DB save error: {e}")

# ── Flask Routes ─────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        resume_text = ''
        job_text = ''

        ct = request.content_type or ''
        if 'multipart/form-data' in ct:
            f = request.files.get('resume_file')
            if f and f.filename:
                resume_text = extract_text_from_file(f)
            if not resume_text:
                resume_text = request.form.get('resume_text', '')
            job_text = request.form.get('job_description', '')
        elif 'application/json' in ct:
            data = request.get_json(force=True, silent=True) or {}
            resume_text = data.get('resume', '')
            job_text = data.get('job_description', '')
        else:
            # Last resort: try form data
            resume_text = request.form.get('resume_text', '')
            job_text = request.form.get('job_description', '')

        if not resume_text or not resume_text.strip():
            return jsonify({'error': 'Resume is required. Upload a file or paste resume text.'}), 400
        if not job_text or not job_text.strip():
            return jsonify({'error': 'Job description is required.'}), 400

        result = calculate_match(resume_text, job_text)
        logger.info(f"Match calculated: {result['match_percentage']}%")

        # Get AI feedback (non-blocking — won't break if Gemini is unavailable)
        ai_feedback = get_gemini_feedback(resume_text, job_text)
        result['ai_feedback'] = ai_feedback

        # Persist to history
        save_to_history(resume_text, job_text, result, ai_feedback)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}", exc_info=True)
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM analyses ORDER BY id DESC LIMIT 20"
        ).fetchall()
        conn.close()
        records = []
        for r in rows:
            records.append({
                'id': r['id'],
                'timestamp': r['timestamp'],
                'resume_snippet': r['resume_snippet'],
                'job_snippet': r['job_snippet'],
                'match_score': r['match_score'],
                'ats_score': r['ats_score'],
                'matched_skills': json.loads(r['matched_skills'] or '[]'),
                'missing_skills': json.loads(r['missing_skills'] or '{}'),
            })
        return jsonify(records)
    except Exception as e:
        logger.error(f"History error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)