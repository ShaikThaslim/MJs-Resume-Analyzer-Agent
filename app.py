import os
import streamlit as ui
from crewai import Agent, Task, Crew
from pypdf import PdfReader

# ================================
# 1. PAGE CONFIGURATION
# ================================
ui.set_page_config(
    page_title="AI Resume Analyzer Agent",
    page_icon="📄",
    layout="centered"
)

# Streamlit Secrets నుండి కీ ని ఆటోమేటిక్ గా రీడ్ చేయడానికి
if "GROQ_API_KEY" in ui.secrets:
    os.environ["GROQ_API_KEY"] = ui.secrets["GROQ_API_KEY"]

# ================================
# 2. APPLICATION STYLING
# ================================
ui.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&family=Fira+Code:wght@400;500;600&display=swap');

/* ── Global Reset & Fonts ── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
}

/* ── Background ── */
.stApp {
    background: #060913;
    background-image: radial-gradient(rgba(0, 242, 254, 0.05) 1px, transparent 0), radial-gradient(rgba(16, 185, 129, 0.03) 1px, transparent 0);
    background-size: 24px 24px;
    background-position: 0 0, 12px 12px;
    min-height: 100vh;
}

/* ── Main Title with Gradient ── */
.main-title {
    font-size: 42px;
    font-weight: 700;
    background: linear-gradient(90deg, #00d4ff, #7b2ff7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

/* ── Subtitle ── */
.sub-title {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 28px;
}

/* ── Section Headers ── */
.section-header {
    font-size: 18px;
    font-weight: 700;
    color: #00f2fe;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 32px 0 16px 0;
}

/* ── FIX: Custom IDE Code Block Style (Line Clipping Issues Completely Fixed) ── */
div[data-testid="stCodeBlock"] {
    background-color: #020617 !important;
    border: 1px solid #1e293b !important;
    border-left: 4px solid #00f2fe !important;
    border-radius: 4px !important;
    padding: 4px !important;
}

div[data-testid="stCodeBlock"] pre {
    background-color: transparent !important;
    padding: 12px !important;
    margin: 0 !important;
    line-height: 1.6 !important; /* పర్ఫెక్ట్ లైన్ స్పేసింగ్ కోసం */
}

div[data-testid="stCodeBlock"] code {
    font-family: 'Fira Code', monospace !important;
    color: #e2e8f0 !important;
    background-color: transparent !important;
    padding: 0 !important;
    word-break: normal !important;
}

/* ── Input Fields ── */
.stTextArea textarea, .stFileUploader {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 6px !important;
    color: #f1f5f9 !important;
}
.stTextArea textarea:focus {
    border-color: #00f2fe !important;
}

/* ── High Contrast Action Buttons ── */
.stButton > button {
    background: #00f2fe !important;
    color: #020617 !important;
    border: none;
    border-radius: 6px;
    padding: 10px 24px;
    font-size: 15px;
    font-weight: 700;
    transition: all 0.2s ease;
    box-shadow: 0 0 15px rgba(0, 242, 254, 0.2);
}
.stButton > button:hover {
    background: #10b981 !important;
    color: #ffffff !important;
    box-shadow: 0 0 20px rgba(16, 185, 129, 0.4);
    transform: translateY(-1px);
}

/* ── Download Button ── */
.stDownloadButton > button {
    background: transparent !important;
    color: #10b981 !important;
    border: 2px solid #10b981 !important;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 14px;
    font-weight: 700;
    transition: all 0.2s ease;
}
.stDownloadButton > button:hover {
    background: #10b981 !important;
    color: #020617 !important;
    box-shadow: 0 0 15px rgba(16, 185, 129, 0.3);
}
</style>
""", unsafe_allow_html=True)

# ================================
# 3. UI HEADERS
# ================================
ui.markdown('<div class="main-title">📄 AI RESUME ANALYZER AGENT</div>', unsafe_allow_html=True)
ui.markdown('<div class="sub-title">Upload your resume in PDF format and enter the job description to generate an AI evaluation report.</div>', unsafe_allow_html=True)
ui.divider()

# ================================
# 4. CORE INPUTS
# ================================
uploaded_file = ui.file_uploader("Upload your Resume (PDF format)", type=["pdf"])
sample_jd = ui.text_area(
    "Paste the Job Description (JD) here",
    height=150,
    placeholder="Required Skills: React, Node.js, Python..."
)

# ================================
# 5. EXECUTION & AGENT LOGIC
# ================================
if ui.button("🚀 ANALYZE RESUME"):
    if uploaded_file is not None and sample_jd.strip():
        with ui.spinner("AI Agents are analyzing your profile... Please wait ⏳"):
            try:
                reader = PdfReader(uploaded_file)
                pages_text = [p.extract_text() for p in reader.pages if p.extract_text()]
                extracted_resume_text = "\n".join(pages_text)

                if not extracted_resume_text.strip():
                    ui.error("❌ PDF నుండి text extract కాలేదు. Image scan PDF కాదు కదా?")
                    ui.stop()

                resume_critic = Agent(
                    role='Expert Resume Critic',
                    goal='Compare resume with JD, identify missing skills, give match percentage score, and provide improvement tips.',
                    backstory='You are a senior technical recruitment manager specializing in screening and auditing engineering resumes.',
                    llm="groq/llama-3.1-8b-instant"
                )
                
                interview_coach = Agent(
                    role='Technical Interview Coach',
                    goal='Formulate high-quality technical interview questions with perfect answers based on resume gaps.',
                    backstory='You are an elite engineering interview coach with experience at top global tech firms.',
                    llm="groq/llama-3.1-8b-instant"
                )

                task_review = Task(
                    description=f"""
                    Compare this Resume:
                    ---
                    {extracted_resume_text}
                    ---
                    With this Job Description:
                    ---
                    {sample_jd}
                    ---
                    
                    Your output MUST strictly follow this exact Markdown text structure:
                    
                    Resume Review Report
                    
                    Match Score: X%
                    
                    [Write a paragraph summarizing how the candidate's resume matches the position, explicitly mentioning candidate name if found]
                    
                    Missing Skills:
                    * [Skill 1] (detailed explanation of the gap)
                    * [Skill 2] (detailed explanation of the gap)
                    * [Skill 3] (detailed explanation of the gap)
                    * [Skill 4] (detailed explanation of the gap)
                    
                    Resume Improvement Tips
                    1. Highlight Python Experience: [Specific text matching the format]
                    2. Emphasize SQL Experience: [Specific text matching the format]
                    3. Deepen React.js Knowledge: [Specific text matching the format]
                    
                    Overall, [Provide a closing summary sentence].
                    """,
                    expected_output="Plain Markdown report matching the exact header structure provided in the description.",
                    agent=resume_critic
                )
                
                task_interview = Task(
                    description="""
                    Generate exactly 3 technical interview questions based on the missing gaps reported.
                    
                    Your output MUST strictly follow this exact Markdown text structure:
                    Personalized Technical Interview Questions for [Candidate Name]
                    
                    Based on the resume review report, we've crafted three high-quality technical interview questions to help [Candidate Name] demonstrate their skills and address the identified gaps.
                    
                    Question 1: Python Development
                    Problem Statement: [Write the problem description here]
                    Code Snippet:
                    [Insert clean python code block here]
                    Expected Answer: [Write the expected response breakdown here]
                    
                    Question 2: SQL Query Optimization
                    Problem Statement: [Write the problem description here]
                    Code Snippet:
                    [Insert clean SQL code block showing Before and After optimization]
                    Expected Answer: [Write the expected response breakdown here]
                    
                    Question 3: React.js and Data Structures
                    Problem Statement: [Write the problem description here]
                    Code Snippet:
                    [Insert clean JavaScript/React code block here]
                    Expected Answer: [Write the expected response breakdown here]
                    
                    These technical interview questions are designed to assess [Candidate Name]'s skills in Python development, SQL query optimization, and React.js, while highlighting areas where they need to improve their knowledge and experience.
                    """,
                    expected_output="3 technical interview questions structured exactly with Problem Statement, Code Snippet, and Expected Answer text formats.",
                    agent=interview_coach
                )

                career_crew = Crew(
                    agents=[resume_critic, interview_coach],
                    tasks=[task_review, task_interview]
                )
                career_crew.kickoff()

                review_output    = career_crew.tasks[0].output.raw
                interview_output = career_crew.tasks[1].output.raw
                full_report      = f"# 🎯 AI Resume Analyzer Report\n\n## 📋 Resume Review\n\n{review_output}\n\n---\n\n## 💡 Interview Questions & Answers\n\n{interview_output}"

            except Exception as e:
                ui.error(f"❌ Error during execution: {e}")
                ui.stop()

        # ================================
        # 6. RENDER RESULTS
        # ================================
        ui.success("✅ Report Generated Successfully!")

        ui.markdown('<div class="section-header">📋 RESUME REVIEW & IMPROVEMENT TIPS</div>', unsafe_allow_html=True)
        ui.markdown(review_output)

        ui.divider()

        ui.markdown('<div class="section-header">💡 PERSONALIZED INTERVIEW QUESTIONS & ANSWERS</div>', unsafe_allow_html=True)
        ui.markdown(interview_output)

        ui.divider()

        ui.markdown('<div class="section-header">📥 DOWNLOAD FULL REPORT</div>', unsafe_allow_html=True)
        ui.download_button(
            label="📥 Download Interview Prep Report (.md)",
            data=full_report,
            file_name="Interview_Prep_Report.md",
            mime="text/markdown"
        )

    else:
        ui.warning("⚠️ Please upload your Resume PDF and enter the Job Description!")