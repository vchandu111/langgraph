from langchain_community.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any
import os
import streamlit as st
from io import StringIO

# ✅ Set API Key
os.environ["OPENAI_API_KEY"] = ""

# ✅ LangGraph Typing
class GraphState(TypedDict):
    job_description: str
    resumes: List[dict]
    job_keywords: str
    scored_resumes: List[dict]
    results: List[dict]

# ✅ Node Functions
def extract_job_keywords(state):
    llm = ChatOpenAI(temperature=0)
    prompt = f"Extract technical and soft skills from this job description:\n\n{state['job_description']}"
    result = llm.predict(prompt)
    return {**state, "job_keywords": result}

def extract_resume_keywords(resume):
    llm = ChatOpenAI(temperature=0)
    prompt = f"Extract key skills from this resume:\n\n{resume['content']}"
    return llm.predict(prompt)

def score_resume(job_keywords, resume_keywords):
    llm = ChatOpenAI(temperature=0)
    prompt = f"""Compare job keywords:\n{job_keywords}\nwith resume keywords:\n{resume_keywords}
Score the match from 0 to 100 and mention matched keywords."""
    return llm.predict(prompt)

def process_each_resume(resume, job_keywords):
    resume_keywords = extract_resume_keywords(resume)
    fit_score = score_resume(job_keywords, resume_keywords)
    return {
        "name": resume["name"],
        "resume_keywords": resume_keywords,
        "score_details": fit_score
    }

def score_all_resumes_node(state):
    job_keywords = state["job_keywords"]
    scored = [
        process_each_resume(resume, job_keywords)
        for resume in state["resumes"]
    ]
    return {**state, "scored_resumes": scored}

def rank_output_node(state):
    def extract_score(score_text):
        try:
            return int(score_text.split("score:")[1].split()[0])
        except:
            return 0

    sorted_resumes = sorted(
        state["scored_resumes"],
        key=lambda x: extract_score(x["score_details"].lower()),
        reverse=True
    )
    return {"results": sorted_resumes}

# ✅ Build LangGraph
state_schema = {
    "job_description": str,
    "resumes": list,
    "job_keywords": str,
    "scored_resumes": list,
    "results": list,
}

graph = StateGraph(state_schema=state_schema)
graph.add_node("extract_keywords", extract_job_keywords)
graph.add_node("score_resumes", score_all_resumes_node)
graph.add_node("rank_output", rank_output_node)
graph.set_entry_point("extract_keywords")
graph.add_edge("extract_keywords", "score_resumes")
graph.add_edge("score_resumes", "rank_output")
graph.add_edge("rank_output", END)
executable = graph.compile()

# ✅ Streamlit UI
st.title("📊 Resume Ranker for Job Fit")

job_desc = st.text_area("Paste Job Description Here", height=200)

uploaded_files = st.file_uploader(
    "Upload Resume Files (Text Format)", type=["txt"], accept_multiple_files=True
)

if st.button("Rank Resumes"):
    if not job_desc or not uploaded_files:
        st.warning("Please provide both a job description and at least one resume.")
    else:
        # Build resume list
        resumes = []
        for file in uploaded_files:
            content = StringIO(file.getvalue().decode("utf-8")).read()
            resumes.append({"name": file.name, "content": content})

        initial_state = {
            "job_description": job_desc,
            "resumes": resumes,
        }

        with st.spinner("Ranking resumes..."):
            output = executable.invoke(initial_state)

        st.success("Ranking complete!")
        for rank, resume in enumerate(output["results"], start=1):
            st.markdown(f"### 🥇 Rank {rank}: {resume['name']}")
            st.markdown(f"**Score Details:** {resume['score_details']}")
            st.markdown(f"**Resume Keywords:** {resume['resume_keywords']}")
            st.markdown("---")
