#resume_analyzer.py
import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


class ResumeAnalysis(BaseModel):
    candidate_name: Optional[str] = Field(default=None)
    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)
    skills: List[str] = Field(default_factory=list)
    years_of_experience: Optional[float] = Field(default=None)
    matched_requirements: List[str] = Field(default_factory=list)
    missing_requirements: List[str] = Field(default_factory=list)
    score: int = Field(description="0 to 100 match score")
    summary: str


def read_text_file(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8")


def analyze_resume(resume_text: str, job_description: str) -> ResumeAnalysis:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(model=model_name, temperature=0)
    structured_llm = llm.with_structured_output(ResumeAnalysis)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert recruiter and resume screening assistant. "
                "Compare the resume against the job description carefully. "
                "Return an objective structured analysis.",
            ),
            (
                "human",
                "Job Description:\n{job_description}\n\nResume:\n{resume_text}",
            ),
        ]
    )

    chain = prompt | structured_llm
    return chain.invoke(
        {
            "job_description": job_description,
            "resume_text": resume_text,
        }
    )


def main():
    print("=== AI Resume Analyzer ===")
    resume_path = input("Enter resume text file path: ").strip()
    jd_path = input("Enter job description text file path: ").strip()

    resume_text = read_text_file(resume_path)
    job_description = read_text_file(jd_path)

    result = analyze_resume(resume_text, job_description)

    print("\n=== Analysis Result ===")
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()