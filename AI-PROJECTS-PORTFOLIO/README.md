 #README.md
 # AI Apps Starter Pack with LangChain + OpenAI

This project contains five starter Python apps:

1. AI Resume Analyzer
2. RAG Chatbot
3. AI Email Assistant
4. AI Research Agent
5. Customer Support Bot

## Prerequisites

- Python 3.10+
- OpenAI API key
- Internet connection

## Setup

### Step 1: Create project folder

Create a folder, for example:

```bash
mkdir ai_apps_starter
cd ai_apps_starter
```

### Step 2: Add files

Create these files in the folder:

- requirements.txt
- .env
- resume_analyzer.py
- rag_chatbot.py
- email_assistant.py
- research_agent.py
- customer_support_bot.py

### Step 3: Create virtual environment

```bash
python -m venv .venv
```

Activate it:

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### Step 4: Install packages

```bash
pip install -r requirements.txt
```

### Step 5: Configure environment variables

Create `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

---

## App 1: Resume Analyzer

### What it does
Reads a resume and job description from text files and returns:
- extracted profile details
- matched requirements
- missing requirements
- score
- summary

### Run

```bash
python resume_analyzer.py
```

### Input needed
- resume text file
- job description text file

---

## App 2: RAG Chatbot

### What it does
Builds a local knowledge base from `.txt` files and answers questions from that knowledge.

### Run

```bash
python rag_chatbot.py
```

### Input needed
- folder containing `.txt` files

### Example folder
```bash
knowledge/
  company_policy.txt
  product_docs.txt
  faq.txt
```

---

## App 3: Email Assistant

### What it does
Generates structured email drafts from a request and optional context.

### Run

```bash
python email_assistant.py
```

### Output
- intent
- tone
- subject
- draft email
- action items

---

## App 4: Research Agent

### What it does
Uses a search tool plus LLM reasoning to research a topic and summarize findings.

### Run

```bash
python research_agent.py
```

### Note
This is a starter agent. For production use, add:
- citation logic
- source validation
- guardrails
- retries
- evaluation datasets

---

## App 5: Customer Support Bot

### What it does
Answers customer questions from support documentation stored as `.txt` files.

### Run

```bash
python customer_support_bot.py
```

### Input needed
- folder with support docs

---

## Suggested Build Order

Build and test in this order:

1. Resume Analyzer
2. RAG Chatbot
3. Customer Support Bot
4. Email Assistant
5. Research Agent

This order helps because structured extraction is easiest first, RAG is next, and multi-step agent workflows are more complex.

---

## Recommended Production Improvements

After the local VS Code prototype works, move toward production by adding:

- FastAPI backend
- prompt templates in separate files
- logging and tracing
- evaluation datasets
- structured output validation
- authentication
- rate limiting
- human escalation for risky actions
- LangGraph for complex agent state management

---

## Notes

These are starter examples designed for learning and prototyping. They are intentionally simple and readable.

For better performance in real projects:
- replace plain `.txt` ingestion with PDF/HTML/document pipelines
- store embeddings persistently
- add test cases
- separate config from code
- version prompts and schemas