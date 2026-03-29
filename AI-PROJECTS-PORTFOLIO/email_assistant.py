#email_assistant.py
import json
import os
from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


class EmailResult(BaseModel):
    intent: str = Field(description="Examples: reply, follow-up, apology, summary, meeting")
    tone: str
    subject: str
    draft_email: str
    action_items: List[str] = Field(default_factory=list)


def generate_email(user_request: str, context: str) -> EmailResult:
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.4)
    structured_llm = llm.with_structured_output(EmailResult)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an AI email assistant. Generate professional, concise, usable emails. "
                "Also identify action items if present.",
            ),
            (
                "human",
                "User request:\n{user_request}\n\nContext:\n{context}",
            ),
        ]
    )

    chain = prompt | structured_llm
    return chain.invoke({"user_request": user_request, "context": context})


def main():
    print("=== AI Email Assistant ===")
    user_request = input("What kind of email do you need? ").strip()
    print("Paste any context and then press Enter. Leave blank if none.")
    context = input("Context: ").strip()

    result = generate_email(user_request, context)

    print("\n=== Email Output ===")
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()