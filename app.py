import os
import streamlit as st
from crewai import Agent, Task, Crew, LLM
from pypdf import PdfReader

st.set_page_config(
    page_title="MJs Resume Analyzer Agent",
    page_icon="📄",
    layout="centered"
)

if "OPENROUTER_API_KEY" in st.secrets:
    os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

st.markdown("""
<style>
header[data-testid="stHeader"] {
    visibility: hidden;
    height: 0% !important;
}
footer {
    visibility: hidden;
}
div[data-testid="stDeveloperToolbar"] {
    display: none !important;
}
.stAppToolbar {
    display: none !important;
}
#MainMenu {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.title("📄 MJs RESUME ANALYZER AGENT")
st.write("Upload your resume in PDF format and enter the job description to generate an AI evaluation report.")

uploaded_file = st.file_uploader("Upload your Resume (PDF format)", type=["pdf"])
jd_input = st.text_area("Paste the Job Description (JD) here")

if st.button("ANALYZE RESUME"):
    if uploaded_file and jd_input.strip():
        with st.spinner("Analyzing resume... Please wait..."):
            try:
                openrouter_llm = LLM(
                    model="openrouter/meta-llama/llama-3.3-70b-instruct",
                    api_key=os.environ.get("OPENROUTER_API_KEY")
                )

                reader = PdfReader(uploaded_file)
                pages_text = []
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        pages_text.append(text)
                extracted_resume_text = "\n".join(pages_text)

                if not extracted_resume_text.strip():
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

                task_review = Task(
                    description=f"Compare this extracted Resume text: '{extracted_resume_text}' with this Job Description: '{jd_input}'. Identify missing skills and give 3 resume improvement tips specifically for this candidate.",
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

                result = career_crew.kickoff()
                
                st.success("Analysis Completed!")
                st.markdown(str(result))

            except Exception as e:
                st.error(f"Error during execution: {e}")
    else:
        st.warning("Please upload a resume and paste the job description!")