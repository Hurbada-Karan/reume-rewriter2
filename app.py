import streamlit as st
import pypdf
import re
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from groq import Groq

st.set_page_config(
    page_title="AI ATS Resume Evaluator & Auditor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp {
        background-color: #f8fafc;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    .custom-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        margin-bottom: 1rem;
    }
    .metric-container {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    .metric-lbl {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-green {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        margin: 0.25rem;
        border: 1px solid #bbf7d0;
    }
    .badge-red {
        display: inline-block;
        background-color: #fee2e2;
        color: #991b1b;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        margin: 0.25rem;
        border: 1px solid #fecaca;
    }
    .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-weight: 700;
        font-size: 1rem;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(124, 58, 237, 0.3);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def init_db():
    conn = sqlite3.connect("cv_evaluations.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evaluation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            filename TEXT,
            match_score REAL,
            semantic_score REAL,
            matched_skills_count INTEGER,
            missing_skills_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def log_evaluation(filename, match_score, semantic_score, matched_count, missing_count):
    conn = sqlite3.connect("cv_evaluations.db")
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO evaluation_logs 
        (timestamp, filename, match_score, semantic_score, matched_skills_count, missing_skills_count)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        filename,
        round(match_score, 2),
        round(semantic_score, 2),
        matched_count,
        missing_count
    ))
    conn.commit()
    conn.close()

init_db()

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_semantic_model()

COMMON_SKILLS = set([
    "python", "java", "c++", "c#", "sql", "mysql", "postgresql", "mongodb",
    "javascript", "typescript", "react", "node.js", "angular", "html", "css",
    "machine learning", "deep learning", "nlp", "computer vision", "tensorflow",
    "pytorch", "scikit-learn", "pandas", "numpy", "opencv", "docker", "kubernetes",
    "aws", "azure", "gcp", "git", "github", "ci/cd", "rest api", "fastapi", "flask",
    "django", "data analysis", "tableau", "power bi", "agile", "scrum", "devops",
    "communication", "leadership", "problem solving", "project management"
])

def extract_text_from_pdf(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    extracted_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text += text + " "
    return extracted_text

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills(cleaned_text):
    found_skills = set()
    for skill in COMMON_SKILLS:
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, cleaned_text):
            found_skills.add(skill.title())
    return found_skills

def calculate_tfidf_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    return float(sim) * 100

def calculate_semantic_similarity(text1, text2):
    embeddings = model.encode([text1, text2])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(sim) * 100

# Function to dynamically fetch supported models live from Groq API
def get_available_groq_models(api_key):
    if not api_key or not api_key.startswith("gsk_"):
        return []
    try:
        client = Groq(api_key=api_key)
        models_data = client.models.list()
        # Filter out whisper / audio / vision-only models
        valid_models = [
            m.id for m in models_data.data 
            if not any(x in m.id for x in ["whisper", "tts", "safetensors"])
        ]
        return valid_models
    except Exception:
        return []

def generate_automated_audit(cv_text, jd_text, api_key, model_choice):
    clean_key = api_key.strip()
    
    if not clean_key or not clean_key.startswith("gsk_"):
        return "⚠️ **API Key Invalid or Missing:** Please enter a valid Groq API key (starting with `gsk_`) in the sidebar."
    
    prompt = f"""
    You are an expert executive resume reviewer and ATS auditor. 
    Analyze the following resume text against the provided job description.
    
    Job Description:
    {jd_text}
    
    Resume Text:
    {cv_text}
    
    Please perform a detailed audit and output:
    1. **3 Specific Weak Points or Poor Bullet Points** found in the resume.
    2. **Why each point is weak** (e.g., lacks metrics, weak verbs, vague description, missing skills).
    3. **A Recommended High-Impact Rewrite** for each weak point.
    
    Keep your response structured with clear bullet points.
    """
    
    try:
        client = Groq(api_key=clean_key)
        response = client.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": "You are a professional resume auditor and career consultant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ **Groq API Error on model '{model_choice}':** {str(e)}"

st.markdown('<div class="main-header">⚡ AI ATS Resume Evaluator & Auditor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated Candidate Suitability, Skill Gap Analysis & Weakness Audit</div>', unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    
    user_api_key = st.text_input(
        "🔑 Groq API Key (gsk_...):", 
        type="password", 
        help="Get a free key from console.groq.com/keys"
    )
    
    GROQ_API_KEY = user_api_key.strip() if user_api_key else ""
    
    # Dynamically fetch model list live from Groq if API key is present
    fetched_models = get_available_groq_models(GROQ_API_KEY)
    
    if fetched_models:
        selected_model = st.selectbox("🤖 Select Active Groq Model:", fetched_models, index=0)
    else:
        # Emergency static fallbacks in case key is invalid/not entered yet
        selected_model = st.selectbox(
            "🤖 Select Groq Model:", 
            ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "qwen-2.5-32b"],
            index=0
        )
        if GROQ_API_KEY:
            st.caption("⚠️ Could not auto-fetch model list. Using default list.")

    if st.checkbox("Show Analytics Logs"):
        conn = sqlite3.connect("cv_evaluations.db")
        df_logs = pd.read_sql_query("SELECT * FROM evaluation_logs ORDER BY id DESC LIMIT 10", conn)
        conn.close()
        st.subheader("Recent Evaluations")
        st.dataframe(df_logs, use_container_width=True)

col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📋 1. Job Description")
    job_description = st.text_area(
        "Paste Job Requirements:",
        height=260,
        placeholder="Paste full job description including required skills..."
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📂 2. Candidate Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF):",
        type=["pdf"]
    )
    st.markdown('</div>', unsafe_allow_html=True)

if st.button("🚀 Analyze & Evaluate Match", use_container_width=True):
    if not job_description.strip():
        st.warning("Please enter a valid Job Description.")
    elif not uploaded_file:
        st.warning("Please upload a PDF resume.")
    else:
        with st.spinner("Processing NLP embeddings, analyzing skill gap, and auditing weaknesses..."):
            raw_cv_text = extract_text_from_pdf(uploaded_file)
            cleaned_cv = clean_text(raw_cv_text)
            cleaned_jd = clean_text(job_description)

            if len(cleaned_cv) < 50:
                st.error("Could not extract text from the PDF file.")
            else:
                keyword_sim = calculate_tfidf_similarity(cleaned_cv, cleaned_jd)
                semantic_sim = calculate_semantic_similarity(cleaned_cv, cleaned_jd)
                overall_score = (0.60 * semantic_sim) + (0.40 * keyword_sim)

                cv_skills = extract_skills(cleaned_cv)
                jd_skills = extract_skills(cleaned_jd)

                matched_skills = cv_skills.intersection(jd_skills)
                missing_skills = jd_skills.difference(cv_skills)

                log_evaluation(
                    filename=uploaded_file.name,
                    match_score=overall_score,
                    semantic_score=semantic_sim,
                    matched_count=len(matched_skills),
                    missing_count=len(missing_skills)
                )

                st.markdown("<br>", unsafe_allow_html=True)
                
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'''
                    <div class="metric-container">
                        <div class="metric-lbl">Overall Match</div>
                        <div class="metric-val">{overall_score:.1f}%</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'''
                    <div class="metric-container">
                        <div class="metric-lbl">Semantic Similarity</div>
                        <div class="metric-val">{semantic_sim:.1f}%</div>
                    </div>
                    ''', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'''
                    <div class="metric-container">
                        <div class="metric-lbl">Keyword Density</div>
                        <div class="metric-val">{keyword_sim:.1f}%</div>
                    </div>
                    ''', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                v1, v2 = st.columns([1, 1], gap="medium")

                with v1:
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(overall_score, 1),
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100], 'tickwidth': 1},
                            'bar': {'color': "#2563eb"},
                            'bgcolor': "white",
                            'borderwidth': 1,
                            'bordercolor': "#cbd5e1",
                            'steps': [
                                {'range': [0, 50], 'color': "#fee2e2"},
                                {'range': [50, 75], 'color': "#fef3c7"},
                                {'range': [75, 100], 'color': "#dcfce7"}
                            ]
                        }
                    ))
                    fig.update_layout(
                        height=260,
                        margin=dict(l=20, r=20, t=30, b=20),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                with v2:
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    st.subheader("💡 Candidate Status")
                    if overall_score >= 75:
                        st.success("High Fit: Candidate strongly aligns with technical and role requirements.")
                    elif overall_score >= 50:
                        st.warning("Moderate Fit: Partial alignment. Review missing skills before proceeding.")
                    else:
                        st.error("Low Fit: Resume lacks essential keywords and skills required for the role.")
                    st.markdown('</div>', unsafe_allow_html=True)

                s1, s2 = st.columns(2, gap="medium")

                with s1:
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    st.subheader("✅ Matched Skills")
                    if matched_skills:
                        badges = "".join([f'<span class="badge-green">{s}</span>' for s in sorted(matched_skills)])
                        st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)
                    else:
                        st.info("No explicit skill matches identified.")
                    st.markdown('</div>', unsafe_allow_html=True)

                with s2:
                    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                    st.subheader("⚠️ Missing Skills")
                    if missing_skills:
                        badges = "".join([f'<span class="badge-red">{s}</span>' for s in sorted(missing_skills)])
                        st.markdown(f'<div>{badges}</div>', unsafe_allow_html=True)
                    else:
                        st.success("Candidate covers all major skills in the job description.")
                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.subheader("🔍 Automated AI Resume Audit & Rewrite Suggestions")
                st.write("The AI has analyzed the extracted text from your resume to highlight weak points and recommended rewrites:")
                
                audit_report = generate_automated_audit(raw_cv_text, job_description, GROQ_API_KEY, selected_model)
                st.markdown(audit_report)
                st.markdown('</div>', unsafe_allow_html=True)