#customer_support_bot.py
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def load_support_docs(folder_path: str) -> str:
    texts = []
    for path in Path(folder_path).glob("*.txt"):
        texts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(texts)


def build_support_knowledge_base(raw_text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    docs = splitter.create_documents([raw_text])
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)


def support_response(vectorstore: FAISS, customer_question: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(customer_question)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0.2)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a customer support assistant. "
                "Answer politely using the support knowledge base. "
                "If uncertain, say the case should be escalated to human support.",
            ),
            (
                "human",
                "Support docs:\n{context}\n\nCustomer question:\n{question}",
            ),
        ]
    )

    chain = prompt | llm
    result = chain.invoke({"context": context, "question": customer_question})
    return result.content


def main():
    print("=== Customer Support Bot ===")
    folder_path = input("Enter support docs folder path: ").strip()
    raw_text = load_support_docs(folder_path)
    vectorstore = build_support_knowledge_base(raw_text)

    print("Support bot ready. Type 'exit' to quit.\n")

    while True:
        question = input("Customer: ").strip()
        if question.lower() == "exit":
            break
        answer = support_response(vectorstore, question)
        print(f"Support Bot: {answer}\n")


if __name__ == "__main__":
    main()