import os
from crewai import Agent, Task, Crew, LLM
from pypdf import PdfReader

try:
    import streamlit as st
    if "OPENROUTER_API_KEY" in st.secrets:
        os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]
except Exception:
    pass

openrouter_llm = LLM(
    model="openrouter/meta-llama/llama-3.3-70b-instruct",
    api_key=os.environ.get("OPENROUTER_API_KEY")
)

pdf_path = "NOOR_BABA_Resume (1).pdf"
extracted_resume_text = ""

try:
    reader = PdfReader(pdf_path)
    pages_text = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)
    extracted_resume_text = "\n".join(pages_text)
    if extracted_resume_text.strip():
        print("[SUCCESS] PDF Resume text extracted successfully!")
    else:
        raise ValueError("PDF is empty.")
except Exception as e:
    print(f"[SYSTEM NOTICE] PDF Reader issue: {e}")
    extracted_resume_text = "Name: Shaik Noor Baba. Skills: React, Node.js, JavaScript, Python, Data Structures, Linux."

resume_critic = Agent(
    role='Expert Resume Critic',
    goal='Job Description thoni resume ni compare chesi gaps ni canipettadam',
    backstory='Nuvvu software engineers resumes ni short-list chese senior HR expert vi.',
    llm=openrouter_llm
)

interview_coach = Agent(
    role='Technical Interview Coach',
    goal='Resume and JD batti interview lo adige questions and vatikala perfect answers/code snippets tayaru cheyadam',
    backstory='Nuvvu leading tech companies lo technical interviews tise top coach vi. Detailed answers and code ivvadam nee speciality.',
    llm=openrouter_llm
)

sample_jd = """
Required Skills: React.js, Node.js, JavaScript, Python, Data Structures and Algorithms, Full-Stack Web Development.
"""

task_review = Task(
    description=f"Compare this extracted Resume text: '{extracted_resume_text}' with this Job Description: '{sample_jd}'. Identify missing skills and give 3 resume improvement tips specifically for this candidate.",
    expected_output="A clean Markdown report containing missing skills and 3 actionable resume improvement tips.",
    agent=resume_critic
)

task_interview = Task(
    description="Based on the review report, create 3 high-quality technical interview questions. For EACH question, provide the complete correct Python/JavaScript answer code or detailed explanation so the user can study directly.",
    expected_output="3 personalized technical interview questions with COMPLETE answers, code snippets, and explanations in clean Markdown format.",
    agent=interview_coach
)

career_crew = Crew(
    agents=[resume_critic, interview_coach],
    tasks=[task_review, task_interview]
)

print("\nGenerating Complete Interview Prep Materials... Please wait...\n")
result = career_crew.kickoff()

report_filename = "Interview_Prep_Report.md"
try:
    with open(report_filename, "w", encoding="utf-8") as f:
        f.write(str(result))
    print(f"\n[SUCCESS] Entire report along with Questions & Answers has been saved to '{report_filename}'!")
except Exception as e:
    print(f"[ERROR] Issue saving file: {e}")