#rag_chatbot.py
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


def load_documents_from_folder(folder_path: str) -> str:
    texts = []
    for path in Path(folder_path).glob("*.txt"):
        texts.append(path.read_text(encoding="utf-8"))
    return "\n\n".join(texts)


def build_vectorstore(raw_text: str) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([raw_text])
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(chunks, embeddings)


def answer_question(vectorstore: FAISS, question: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful RAG assistant. Answer only from the provided context. "
                "If the answer is not in the context, say you do not know.",
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}",
            ),
        ]
    )

    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content


def main():
    print("=== RAG Chatbot ===")
    folder_path = input("Enter folder path containing .txt knowledge files: ").strip()
    raw_text = load_documents_from_folder(folder_path)
    vectorstore = build_vectorstore(raw_text)

    print("Knowledge base loaded. Type 'exit' to quit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        answer = answer_question(vectorstore, question)
        print(f"Bot: {answer}\n")


if __name__ == "__main__":
    main()