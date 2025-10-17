# support_knowledge_app.py
"""
Support Knowledge Assistant — Multi-page Streamlit app
Features:
- Authentication (streamlit-authenticator)
- Multi-tab UI: Dashboard, Chatbot, Tickets, Documents, Validation, Admin
- RAG (FAISS embeddings) using Google Generative Embeddings (Gemini) and fallback to OpenAI/local JSON
- Google Sheets integration fallback to local JSON files
- Chat history, ticket storage, pilot validation, and helpful utilities

Note: This single file is intentionally verbose and comprehensive for clarity.
"""

import os
import json
import datetime
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv

# Streamlit UI
import streamlit as st

# Authenticator
try:
    import streamlit_authenticator as stauth
    HAS_AUTH = True
except Exception:
    HAS_AUTH = False

# Optional Google Sheets
try:
    import gspread
    from google.oauth2.service_account import Credentials
    USE_SHEETS = True
except Exception:
    USE_SHEETS = False

# LangChain / loaders / vectorstore / embeddings
# Some imports may not be available in all envs; guard them.
try:
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain.chains import create_retrieval_chain
    from google.api_core.exceptions import ResourceExhausted
    LANGCHAIN_AVAILABLE = True
except Exception:
    # Keep flags for graceful degradation
    LANGCHAIN_AVAILABLE = False
    # Provide dummy placeholders (document loaders/fallbacks will rely on local JSON)
    ResourceExhausted = Exception

# Data & plotting
import pandas as pd
import matplotlib.pyplot as plt

# Load environment
load_dotenv()

# ------------------- CONFIG -------------------
DEFAULT_DOCS_PATH = Path.cwd() / "support_knowledge.pdf"
INDEX_PATH = Path("./faiss_index")
QUERIES_FILE = Path("queries.json")
PILOT_FILE = Path("pilot_results.json")
TICKETS_FILE = Path("ticket_raised.json")
USERS_FILE = Path("users.json")  # persisted user list for auth (optional)

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/embedding-001")

# ✅ Primary: Gemini Flash | Fallback: Gemini Pro
PRIMARY_CHAT_MODEL = os.getenv("PRIMARY_CHAT_MODEL", "gemini-1.5-flash")
FALLBACK_CHAT_MODEL = os.getenv("FALLBACK_CHAT_MODEL", "gemini-1.0-pro")

TOP_K = int(os.getenv("TOP_K", 4))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # optional fallback
hf_ = os.getenv("hf_")

SHEET_NAME = os.getenv("SHEET_NAME", "PDF_AI_Internship_Infosys")
WORKSHEET_NAME = os.getenv("WORKSHEET_NAME", "Sheet1")
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE", "credentials.json")

HEADERS = [
    "Ticket ID",
    "Ticket Content",
    "Ticket Timestamp",
    "Ticket By",
    "Ticket Raised By",
    "Ticket Category",
    "Ticket Problem",
    "Ticket Solution",
    "Ticket Status"
]

# ------------------- INITIAL WARNINGS -------------------
st.set_page_config(page_title="Support Knowledge Assistant", layout="wide")

# Check important keys
if not GOOGLE_API_KEY:
    st.sidebar.warning("⚠️ GOOGLE_API_KEY not found in environment. Gemini endpoints may not work.")
if not HAS_AUTH:
    st.sidebar.warning("⚠️ streamlit-authenticator not installed. Install for auth: pip install streamlit-authenticator")
if not LANGCHAIN_AVAILABLE:
    st.sidebar.warning("⚠️ langchain or related packages not available. RAG features disabled and local JSON fallback will be used.")

# ------------------- GOOGLE SHEETS SETUP -------------------
sheet = None
if USE_SHEETS:
    try:
        creds_path = Path(CREDENTIALS_FILE)
        if not creds_path.exists():
            raise FileNotFoundError(f"Credentials file not found: {CREDENTIALS_FILE}")

        scope = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        creds = Credentials.from_service_account_file(str(creds_path), scopes=scope)
        client = gspread.authorize(creds)

        spreadsheet = client.open(SHEET_NAME)
        sheet = spreadsheet.worksheet(WORKSHEET_NAME)

        existing_headers = sheet.row_values(1) or []
        if existing_headers != HEADERS:
            try:
                if existing_headers:
                    sheet.delete_row(1)
            except Exception:
                pass
            sheet.insert_row(HEADERS, 1)

        st.sidebar.success(f"✅ Connected to Google Sheets → {WORKSHEET_NAME}")

    except Exception as e:
        USE_SHEETS = False
        sheet = None
        # Provide readable message in sidebar
        st.sidebar.error(f"Google Sheets setup failed: {e}")
else:
    st.sidebar.info("Using local JSON fallback for tickets/queries unless gspread present.")

# ------------------- AUTHENTICATION -------------------
# We'll offer two modes: streamlit-authenticator based auth or a lightweight internal auth
# that saves basic user info to users.json. If stauth is installed, prefer it.
# ----------------------------
# Load Users from Google Sheets
# ----------------------------
def load_users():
    data = USER_SHEET.get_all_records()
    # Convert into dict {username: {name, password}}
    users = {}
    for row in data:
        users[row["username"]] = {
            "name": row["name"],
            "password": row["password"]
        }
    return users


USERS_FILE = Path("users.json")

# ----------------- User Storage Helpers -----------------
def load_users_from_file(users_file: Path = USERS_FILE) -> Dict[str, Dict[str, str]]:
    """Load users dictionary from JSON file. If not found, return empty dict."""
    if users_file.exists():
        try:
            return json.loads(users_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_users_to_file(users: Dict[str, Dict[str, str]], users_file: Path = USERS_FILE):
    """Save users dictionary to JSON file."""
    users_file.write_text(json.dumps(users, indent=2), encoding="utf-8")

# ----------------- Default Users -----------------
_default_users = {
    "admin": {"name": "Administrator", "password": "adminpass"},
    "agent": {"name": "Support Agent", "password": "agentpass"}
}

# If users.json doesn’t exist → create with default users
if not USERS_FILE.exists():
    save_users_to_file(_default_users)

# Load current users into memory
USERS = load_users_from_file()

# ----------------- Setup Authenticator -----------------
authenticator = None
if HAS_AUTH:
    try:
        # Build credentials dict for streamlit-authenticator
        credentials = {"usernames": {}}
        for u, info in USERS.items():
            # Fix: Ensure 'name' key exists in user info dictionary
            user_name = info.get("name", u)
            credentials["usernames"][u] = {
                "name": user_name,
                "password": info["password"]  # ⚠️ In real apps, use hashed passwords!
            }

        authenticator = stauth.Authenticate(
            credentials,
            "support_knowledge_cookie",
            "support_knowledge_signature",
            cookie_expiry_days=1
        )
    except Exception as e:
        st.warning(f"⚠️ Authenticator setup failed: {e}")
        authenticator = None

# ------------------- MODEL LOADER & FALLBACK -------------------
from langchain import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
def load_llm(model_name=None):
    """Load a chat model: Hugging Face -> OpenAI -> local JSON fallback"""
    # Try Hugging Face first
    if os.getenv("HF_API_KEY"):
        try:
            return HuggingFaceHub(
                repo_id="bigscience/bloomz-7b1-mt",
                model_kwargs={"temperature":0, "max_new_tokens":512},
                huggingfacehub_api_token=os.getenv("HF_API_KEY")
            )
        except Exception as e:
            st.warning(f"⚠️ Hugging Face model load failed: {e}")

    # Optional OpenAI fallback
    if OPENAI_API_KEY:
        try:
            return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
        except Exception as e:
            st.warning(f"⚠️ OpenAI fallback failed: {e}")

    # Local JSON fallback
    if QUERIES_FILE.exists():
        st.warning("📂 Using local JSON fallback for responses.")
        class LocalJSONModel:
            def __init__(self, file):
                with open(file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)

            def invoke(self, input_text):
                ans = self.data.get(input_text)
                if isinstance(ans, dict) and "answer" in ans:
                    return {"answer": ans["answer"]}
                return {"answer": ans or "⚠️ No saved response found."}

        return LocalJSONModel(QUERIES_FILE)

    st.error("🚨 No LLM available. Set HF_API_KEY or OPENAI_API_KEY.")
    return None


def get_chat_model():
    """
    Try to return a chat LLM instance:
    - Gemini primary
    - Gemini fallback
    - OpenAI fallback
    - Local JSON fallback (LocalJSONModel) — mimics interface.
    """
    # Gemini
    if GOOGLE_API_KEY and LANGCHAIN_AVAILABLE:
        try:
            return ChatGoogleGenerativeAI(model=PRIMARY_CHAT_MODEL, google_api_key=GOOGLE_API_KEY)
        except ResourceExhausted:
            st.warning(f"⚠️ Quota exhausted for {PRIMARY_CHAT_MODEL}, trying fallback...")
        except Exception as e:
            st.warning(f"⚠️ Gemini primary failed: {e}")

        try:
            return ChatGoogleGenerativeAI(model=FALLBACK_CHAT_MODEL, google_api_key=GOOGLE_API_KEY)
        except Exception as e:
            st.warning(f"⚠️ Gemini fallback failed: {e}")

    # OpenAI fallback
    if OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
        except Exception as e:
            st.warning(f"⚠️ OpenAI fallback failed: {e}")

    # Local JSON fallback
    if QUERIES_FILE.exists():
        class LocalJSONModel:
            def __init__(self, file):
                with open(file, "r", encoding="utf-8") as f:
                    try:
                        self.data = json.load(f)
                    except Exception:
                        self.data = {}

            def invoke(self, input_text: str):
                ans = self.data.get(input_text, None)
                if isinstance(ans, dict) and "answer" in ans:
                    return {"answer": ans["answer"]}
                return {"answer": ans or "⚠️ No saved response found."}
        return LocalJSONModel(QUERIES_FILE)

    # If nothing
    raise RuntimeError("No LLM available. Provide GOOGLE_API_KEY or OPENAI_API_KEY, or queries.json fallback.")

# ------------------- HELPERS -------------------
def find_files(path: Path):
    if path.is_file():
        return [path]
    exts = {".txt", ".md", ".pdf"}
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def load_documents(paths: List[Path]) -> List[Any]:
    """
    Load files into Document-like objects (langchain Documents if available).
    """
    docs = []
    if not LANGCHAIN_AVAILABLE:
        st.warning("LangChain document loaders not available; returning basic file texts as dicts.")
        for p in paths:
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            docs.append({"page_content": text, "metadata": {"source": str(p)}})
        return docs

    for p in paths:
        try:
            if p.suffix.lower() in {".txt", ".md"}:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif p.suffix.lower() == ".pdf":
                docs.extend(PyPDFLoader(str(p)).load())
        except Exception as e:
            st.error(f"❌ Failed to load {p}: {e}")
    return docs


def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Split documents using RecursiveCharacterTextSplitter if available.
    """
    if not LANGCHAIN_AVAILABLE:
        # naive split fallback
        split_docs = []
        for d in docs:
            text = d.get("page_content", "") if isinstance(d, dict) else getattr(d, "page_content", "")
            i = 0
            while i < len(text):
                chunk_text = text[i:i + chunk_size]
                split_docs.append({"page_content": chunk_text, "metadata": getattr(d, "metadata", {})})
                i += chunk_size - chunk_overlap
        return split_docs

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
from pathlib import Path
import os

INDEX_PATH = Path("./faiss_index")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import streamlit as st
from pathlib import Path

INDEX_PATH = Path("./faiss_index")
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import streamlit as st
from pathlib import Path

# Example constants
INDEX_PATH = Path("faiss_index")

def build_or_load_faiss(chunks, rebuild=True):
    """
    Build or load a FAISS vectorstore using Hugging Face sentence-transformers.
    Only uses Hugging Face; no OpenAI or Google fallback.
    """
    import pathlib
    from pathlib import Path

    if not chunks:
        raise ValueError("No document chunks found. Upload documents first.")

    INDEX_PATH = Path("faiss_index")

    # Try importing sentence-transformers
    try:
        from langchain.embeddings import HuggingFaceEmbeddings
    except ImportError:
        st.error("⚠️ `sentence-transformers` not installed. Install it with `pip install sentence-transformers`.")
        return None

    # Initialize embeddings
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"❌ Failed to initialize Hugging Face embeddings: {e}")
        return None

    # Build or load FAISS
    try:
        from langchain.vectorstores import FAISS

        if rebuild:
            st.info("📂 Building new FAISS index with Hugging Face embeddings...")
            vs = FAISS.from_documents(chunks, embeddings)
            INDEX_PATH.mkdir(parents=True, exist_ok=True)
            vs.save_local(str(INDEX_PATH))
            st.success("✅ FAISS index built successfully.")
        else:
            st.info("📂 Loading existing FAISS index from local cache...")
            vs = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
            st.success("✅ FAISS index loaded successfully.")

        return vs

    except Exception as e:
        st.error(f"❌ Failed to build or load FAISS index: {e}")
        return None



def load_llm(model_name=None):
    """
    Load a chat model with fallback support:
    1. Gemini primary → Gemini fallback
    2. OpenAI
    3. Local JSON fallback
    """
    # Try Gemini first
    if GOOGLE_API_KEY:
        try:
            return ChatGoogleGenerativeAI(model=PRIMARY_CHAT_MODEL, google_api_key=GOOGLE_API_KEY)
        except Exception as e:
            msg = str(e).lower()
            if "quota" in msg or "429" in msg:
                st.warning("⚠️ Gemini primary quota exceeded. Trying fallback model...")
            else:
                st.warning(f"⚠️ Gemini primary failed: {e}")

        try:
            return ChatGoogleGenerativeAI(model=FALLBACK_CHAT_MODEL, google_api_key=GOOGLE_API_KEY)
        except Exception as e:
            st.warning(f"⚠️ Gemini fallback failed: {e}")

    # Fallback to OpenAI
    if OPENAI_API_KEY:
        try:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
        except Exception as e:
            st.warning(f"⚠️ OpenAI fallback failed: {e}")

    # Final local JSON fallback
    if QUERIES_FILE.exists():
        st.warning("📂 Using local JSON fallback for responses.")
        class LocalJSONModel:
            def __init__(self, file):
                with open(file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)

            def invoke(self, input_text):
                ans = self.data.get(input_text)
                if isinstance(ans, dict) and "answer" in ans:
                    return {"answer": ans["answer"]}
                return {"answer": ans or "⚠️ No saved response found."}

        return LocalJSONModel(QUERIES_FILE)

    st.error("🚨 No LLM available. Set GOOGLE_API_KEY or OPENAI_API_KEY.")
    return None

def save_query_answer(query, answer, status):
    data = []
    if QUERIES_FILE.exists():
        try:
            with open(QUERIES_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    data.append({"query": query, "answer": answer, "status": status, "timestamp": datetime.datetime.now().isoformat()})
    with open(QUERIES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def append_ticket_to_sheets(ticket_data):
    global sheet

    if sheet is None:
        st.error("⚠️ Google Sheet not connected. Ticket not saved.")
        return

    try:
        row = [
            ticket_data.get("ticket_id", ""),
            ticket_data.get("ticket_timestamp", ""),
            ticket_data.get("ticket_by", ""),
            ticket_data.get("ticket_raised_by", ""),
            ticket_data.get("ticket_content", ""),
            ticket_data.get("ticket_problem", ""),
            ticket_data.get("ticket_category", ""),
            ticket_data.get("ticket_solution", ""),
            ticket_data.get("ticket_status", ""),
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
        st.success("✅ Ticket saved to Google Sheets.")
    except Exception as e:
        st.error(f"❌ Failed to save ticket to Google Sheets: {e}")




# ------------------- RAG / CLASSIFICATION -------------------
CATEGORY_PROMPT = ChatPromptTemplate.from_template(
    """You are a support ticket classifier. Based only on the knowledge base provided:
{context}

Classify this ticket into the most relevant category:
{input}

Reply with only the category name."""
) if LANGCHAIN_AVAILABLE else None

ASSISTANT_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful support assistant. Use only the following documents to answer: 
{context}
User Question: {input}"""
) if LANGCHAIN_AVAILABLE else None


from transformers import pipeline

# ✅ Create a Hugging Face pipeline (only once, globally)
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

import re
from difflib import SequenceMatcher

def parse_document_categories(docs):
    """
    Parse the document text to extract category and problem mappings.
    Returns a list of tuples: (category, problem_text)
    """
    category_problems = []
    category = None
    problem_accum = []

    # Combine all docs text
    full_text = "\n".join([doc.page_content for doc in docs])

    # Split by lines
    lines = full_text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Detect category lines
        cat_match = re.match(r"Category:\s*(.+)", line, re.IGNORECASE)
        if cat_match:
            # Save previous category problems
            if category and problem_accum:
                problem_text = " ".join(problem_accum).strip()
                if problem_text:
                    category_problems.append((category, problem_text))
                problem_accum = []
            category = cat_match.group(1).strip()
            continue
        # Detect problem lines
        prob_match = re.match(r"Problem:\s*(.+)", line, re.IGNORECASE)
        if prob_match:
            # Save previous problem if any
            if problem_accum:
                problem_text = " ".join(problem_accum).strip()
                if category and problem_text:
                    category_problems.append((category, problem_text))
                problem_accum = []
            problem_accum.append(prob_match.group(1).strip())
            continue
        # Detect solution lines or other lines, ignore or stop accumulating problem
        sol_match = re.match(r"Solution:", line, re.IGNORECASE)
        if sol_match:
            # End of problem description
            if category and problem_accum:
                problem_text = " ".join(problem_accum).strip()
                category_problems.append((category, problem_text))
            problem_accum = []
            continue
        # Accumulate problem lines if no other match
        if problem_accum is not None:
            problem_accum.append(line)

    # Add last problem if any
    if category and problem_accum:
        problem_text = " ".join(problem_accum).strip()
        category_problems.append((category, problem_text))

    return category_problems

def parse_document_problems_solutions(docs):
    """
    Parse the document text to extract category, problem, and solution mappings.
    Returns a list of tuples: (category, problem, solution)
    """
    problems_solutions = []
    category = None
    problem = None
    solution_accum = []

    # Combine all docs text
    full_text = "\n".join([doc.page_content for doc in docs])

    # Split by lines
    lines = full_text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Detect category lines
        cat_match = re.match(r"Category:\s*(.+)", line, re.IGNORECASE)
        if cat_match:
            category = cat_match.group(1).strip()
            continue
        # Detect problem lines
        prob_match = re.match(r"Problem:\s*(.+)", line, re.IGNORECASE)
        if prob_match:
            # Save previous problem solution if any
            if problem and solution_accum:
                solution_text = " ".join(solution_accum).strip()
                if category and solution_text:
                    problems_solutions.append((category, problem, solution_text))
                solution_accum = []
            problem = prob_match.group(1).strip()
            continue
        # Detect solution lines
        sol_match = re.match(r"Solution:\s*(.+)", line, re.IGNORECASE)
        if sol_match:
            solution_accum.append(sol_match.group(1).strip())
            continue
        # Accumulate solution lines if after Solution:
        if solution_accum:
            solution_accum.append(line)

    # Add last problem solution if any
    if problem and solution_accum:
        solution_text = " ".join(solution_accum).strip()
        if category and solution_text:
            problems_solutions.append((category, problem, solution_text))

    return problems_solutions

def best_match_category(ticket_content, category_problems):
    """
    Find the best matching category for the ticket content by comparing similarity
    with the extracted problem texts from the document.
    Returns the category with the highest similarity score above a threshold.
    """
    best_category = None
    best_score = 0.0
    threshold = 0.4  # similarity threshold to accept

    for category, problem_text in category_problems:
        score = SequenceMatcher(None, ticket_content.lower(), problem_text.lower()).ratio()
        if score > best_score:
            best_score = score
            best_category = category

    if best_score >= threshold:
        return best_category
    return None

def best_match_problem_solution(ticket_content, problems_solutions):
    """
    Find the best matching problem and solution for the ticket content by comparing similarity
    with the extracted problem texts from the document.
    Returns the (category, problem, solution) with the highest similarity score above a threshold.
    """
    best_match = None
    best_score = 0.0
    threshold = 0.4  # similarity threshold to accept

    for category, problem, solution in problems_solutions:
        score = SequenceMatcher(None, ticket_content.lower(), problem.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = (category, problem, solution)

    if best_score >= threshold:
        return best_match
    return None

def categorize_ticket(content, retriever, docs=None):
    if docs:
        category_problems = parse_document_categories(docs)
        categories = list(set(cat for cat, prob in category_problems))
        if categories:
            result = classifier(content, candidate_labels=categories)
            return result['labels'][0]

    # Fallback to LLM if no categories or no docs
    category_prompt = ChatPromptTemplate.from_template(
        """You are a support ticket classifier. Based only on the knowledge base provided:
        {context}

        Classify this ticket into the most relevant category:
        {input}

        Reply with only the category name. If no relevant category is found, reply with 'Uncategorized'."""
    )
    try:
        llm = load_llm(PRIMARY_CHAT_MODEL)
        doc_chain = create_stuff_documents_chain(llm, category_prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        result = rag_chain.invoke({"input": content})
        return result["answer"].strip()
    except ResourceExhausted:
        print("⚠️ Quota exceeded. Falling back to flash...")
        llm = load_llm(FALLBACK_CHAT_MODEL)
        doc_chain = create_stuff_documents_chain(llm, category_prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        result = rag_chain.invoke({"input": content})
        return result["answer"].strip()
    except Exception as e:
        return "Uncategorized"

def update_ticket_status_in_sheets(ticket_id, new_status):
    global sheet

    if sheet is None:
        st.error("⚠️ Google Sheet not connected. Cannot update ticket.")
        return

    try:
        all_values = sheet.get_all_values()
        for idx, row in enumerate(all_values[1:], start=2):  # skip header row
            if str(row[0]) == str(ticket_id):
                sheet.update_cell(idx, 9, new_status)  # 9th column = ticket_status
                st.success("✅ Ticket status updated.")
                return
        st.warning("⚠️ Ticket ID not found.")
    except Exception as e:
        st.error(f"❌ Failed to update ticket status: {e}")


from transformers import pipeline

# ✅ Lightweight model for short responses
generator = pipeline("text-generation", model="distilgpt2")

def resolve_ticket(content: str, retriever=None, prompt=None):
    """
    Suggest a resolution using Hugging Face text generation instead of LLM.
    """
    try:
        prompt_text = f"Provide steps to resolve the following support ticket:\n\n{content}\n\nResolution:"
        response = generator(prompt_text, max_length=150, num_return_sequences=1)
        return response[0]["generated_text"].split("Resolution:")[-1].strip()
    except Exception as e:
        st.warning(f"Resolution generation failed: {e}")
        return (
            "Suggested steps to troubleshoot:\n"
            "1. Reproduce the issue and collect logs.\n"
            "2. Check configuration and recent changes.\n"
            "3. Restart service and verify.\n"
            "4. If persists, escalate to engineering."
        )

# ------------------- PILOT VALIDATION -------------------
def run_pilot_validation(labeled_samples: List[Dict[str, str]], retriever) -> Dict[str, Any]:
    correct = 0
    total = len(labeled_samples)
    per_class = {}
    for s in labeled_samples:
        predicted = categorize_ticket(s["content"], retriever)
        label = s.get("label", "").strip()
        per_class.setdefault(label or "Unknown", {"total": 0, "correct": 0})
        per_class[label or "Unknown"]["total"] += 1
        if predicted.lower() == label.lower():
            correct += 1
            per_class[label or "Unknown"]["correct"] += 1
    accuracy = correct / total if total else 0.0
    metrics = {"accuracy": accuracy, "total": total, "per_class": per_class}
    with open(PILOT_FILE, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.datetime.now().isoformat(), "metrics": metrics}, f, indent=4)
    return metrics

# ------------------- TICKET STORAGE -------------------
def save_ticket(ticket: Dict):
    """
    Save a ticket to local TICKETS_FILE with consistent headers.
    """
    row = {
        "Ticket ID": ticket.get("ticket_id", ""),
        "Ticket Content": ticket.get("ticket_content", ""),
        "Ticket Timestamp": ticket.get("ticket_timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "Ticket By": ticket.get("ticket_by", ""),
        "Ticket Raised By": ticket.get("ticket_raised_by", ""),
        "Ticket Category": ticket.get("ticket_category", ""),
        "Ticket Problem": ticket.get("ticket_problem", ""),
        "Ticket Solution": ticket.get("ticket_solution", ""),
        "Ticket Status": ticket.get("ticket_status", "Open"),
    }

    data = []
    if TICKETS_FILE.exists():
        try:
            with open(TICKETS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            data = []
    data.append(row)
    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ------------------- UI HELPERS -------------------
def human_datetime(ts: Optional[str]) -> str:
    if not ts:
        return ""
    try:
        dt = datetime.datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(ts)


# ------------------- APP UI SECTIONS -------------------
def init_state():
    """Initialize session state keys used by the app."""
    if "docs" not in st.session_state:
        st.session_state.docs = []
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "queries" not in st.session_state:
        st.session_state.queries = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "top_k" not in st.session_state:
        st.session_state.top_k = TOP_K
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = CHUNK_SIZE
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = CHUNK_OVERLAP
    if "primary_model" not in st.session_state:
        st.session_state.primary_model = PRIMARY_CHAT_MODEL
    if "fallback_model" not in st.session_state:
        st.session_state.fallback_model = FALLBACK_CHAT_MODEL
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "admin_mode" not in st.session_state:
        st.session_state.admin_mode = False


def sidebar_settings():
    st.sidebar.title("⚙️ Settings")
    st.session_state.top_k = st.sidebar.number_input("Top K", value=st.session_state.top_k, min_value=1, max_value=20)
    st.session_state.chunk_size = st.sidebar.number_input("Chunk size", value=st.session_state.chunk_size, min_value=100, max_value=2000)
    st.session_state.chunk_overlap = st.sidebar.number_input("Chunk overlap", value=st.session_state.chunk_overlap, min_value=0, max_value=1000)
    st.sidebar.markdown("**Models**")
    st.session_state.primary_model = st.sidebar.text_input("Primary chat model", value=st.session_state.primary_model)
    st.session_state.fallback_model = st.sidebar.text_input("Fallback chat model", value=st.session_state.fallback_model)

    if st.sidebar.button("Rebuild index"):
        if st.session_state.chunks:
            with st.spinner("Building index..."):
                try:
                    st.session_state.vectorstore = build_or_load_faiss(st.session_state.chunks, rebuild=True)
                    st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr", search_kwargs={"k": st.session_state.top_k}
                    )
                    st.sidebar.success("✅ Index rebuilt.")
                except Exception as e:
                    st.sidebar.error(f"Failed to rebuild index: {e}")
        else:
            st.sidebar.warning("Upload & split documents first.")


def sidebar_documents_uploader():
    st.sidebar.title("📄 Upload Documents")
    uploaded = st.sidebar.file_uploader("Upload docs (pdf, txt, md)", accept_multiple_files=True, type=["pdf", "txt", "md"])
    if uploaded:
        dest = Path("./uploaded_docs")
        dest.mkdir(exist_ok=True)
        for up in uploaded:
            fp = dest / up.name
            with open(fp, "wb") as f:
                f.write(up.getbuffer())
        st.sidebar.success("✅ Uploaded files saved to ./uploaded_docs")


# ------------------- Chatbot UI -------------------
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline

def build_or_load_faiss(chunks, index_path="faiss_index", rebuild=True):
    """
    Build or load a FAISS vectorstore using Hugging Face embeddings only.
    """
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if rebuild:
        vs = FAISS.from_documents(chunks, embeddings)
        vs.save_local(index_path)
        st.success("✅ FAISS index built using Hugging Face embeddings.")
        return vs
    elif not rebuild and os.path.exists(index_path):
        try:
            vs = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            st.success("✅ FAISS index loaded from local cache.")
            return vs
        except Exception as e:
            st.error(f"❌ Failed to load FAISS index: {e}")
            return None
    else:
        st.error("🚨 No FAISS index found. Upload documents first.")
        return None

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline

CHAT_HISTORY_FILE = "chat_history.json"

def chatbot_ui():
    st.header("🤖 Chatbot (Hugging Face)")

    # Load chat history
    if os.path.exists(CHAT_HISTORY_FILE):
        try:
            with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                chat_history_data = json.load(f)
                current_user = st.session_state.get("current_user", "yam")
                chat_history = chat_history_data.get(current_user, [])
        except:
            chat_history = []
    else:
        chat_history = []

    # Display chat history
    for msg in chat_history:
        st.markdown(f"**You:** {msg.get('user', '')}")
        st.markdown(f"**Bot:** {msg.get('bot', '')}")
        st.markdown("---")

    user_query = st.text_input("Ask a question about your documents:", key="chat_input")

    if st.button("Send") and user_query:
        try:
            # Load FAISS index
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            docs = retriever.get_relevant_documents(user_query)

            if not docs:
                response = "No relevant documents found in index."
            else:
                # Combine context from retrieved documents
                context = " ".join([doc.page_content for doc in docs])

                # Generate answer using Hugging Face model
                generator = pipeline("text2text-generation", model="google/flan-t5-small")
                response = generator(
                    f"Answer the question based on the following context:\n{context}\n\nQuestion: {user_query}",
                    max_length=300
                )[0]['generated_text']

            # Append to history
            chat_history.append({"user": user_query, "bot": response})

            # Save history
            if os.path.exists(CHAT_HISTORY_FILE):
                try:
                    with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
                        full_history = json.load(f)
                except:
                    full_history = {}
            else:
                full_history = {}
            full_history[current_user] = chat_history
            with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(full_history, f, indent=2, ensure_ascii=False)

            # Display the new response
            st.markdown(f"**You:** {user_query}")
            st.markdown(f"**Bot:** {response}")
            st.markdown("---")

            # Rerun to update history display
            st.rerun()

        except Exception as e:
            st.error(f"❌ Error generating answer locally: {e}")

# ✅ Simple login system (for demo)
def login_ui():
    st.sidebar.header("🔐 Authentication")

    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    choice = st.sidebar.radio("Choose action:", ["Login", "Signup"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username and password:  # Replace with real check
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.sidebar.success(f"Welcome, {username}!")
        else:
            st.sidebar.error("Invalid credentials")

    if st.session_state["authenticated"]:
        st.sidebar.info(f"Logged in as {st.session_state['username']}")
        if st.sidebar.button("Logout"):
            st.session_state["authenticated"] = False
            st.session_state["username"] = None
            st.rerun()

# ------------------- Documents & Indexing UI -------------------
def docs_ui():
    st.header("📚 Documents & Indexing")
    st.write("Load documents from a local path or upload files in the sidebar, then split and build a FAISS index for RAG.")

    docs_path = st.text_input("Local docs path", value=str(DEFAULT_DOCS_PATH))
    col1, col2 = st.columns([3, 1])

    with col1:
        if st.button("Load & Split Documents"):
            try:
                paths = find_files(Path(docs_path))
                if not paths:
                    st.warning("No supported files found at the path.")
                else:
                    st.session_state.docs = load_documents(paths)
                    st.session_state.chunks = split_documents(
                        st.session_state.docs,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap
                    )
                    st.success(f"Loaded {len(st.session_state.docs)} docs → {len(st.session_state.chunks)} chunks.")
            except Exception as e:
                st.error(f"Failed to load & split documents: {e}")

        if st.session_state.chunks:
            st.subheader("Sample chunk")
            try:
                # chunks may be langchain Documents or dicts
                chunk = st.session_state.chunks[0]
                if hasattr(chunk, "page_content"):
                    preview = chunk.page_content[:1000]
                elif isinstance(chunk, dict):
                    preview = chunk.get("page_content", "")[:1000]
                else:
                    preview = str(chunk)[:1000]
                st.code(preview)
            except Exception:
                st.write("Unable to preview chunk.")

            if st.button("Build FAISS index"):
                with st.spinner("Building index..."):
                    try:
                        st.session_state.vectorstore = build_or_load_faiss(st.session_state.chunks, rebuild=True)
                        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                            search_type="mmr", search_kwargs={"k": st.session_state.top_k}
                        )
                        st.success("✅ Index built.")
                    except Exception as e:
                        st.error(f"Failed to build index: {e}")

    with col2:
        st.markdown("### Quick tools")
        if st.button("Save chunks as JSON (debug)"):
            try:
                outp = "chunks_dump.json"
                simple_chunks = []
                for c in st.session_state.chunks:
                    try:
                        text = c.page_content
                    except Exception:
                        text = c.get("page_content", "")
                    simple_chunks.append({"text": text[:200], "meta": getattr(c, "metadata", {})})
                with open(outp, "w", encoding="utf-8") as f:
                    json.dump(simple_chunks, f, indent=2)
                st.success(f"Saved chunk samples to {outp}")
            except Exception as e:
                st.error(f"Failed to save chunks: {e}")

    st.markdown("---")
    st.caption("Tip: If you have a large PDF, increase chunk size to preserve context or decrease overlap to save tokens.")


# ------------------- Tickets UI -------------------
def tickets_ui():
    st.header("🎫 Tickets")
    st.write("Create new tickets, view recent tickets, and update status.")

    # Ensure queries list is initialized
    if "queries" not in st.session_state:
        st.session_state.queries = []

    # ===================== Ticket Creation Form =====================
    with st.form("ticket_form", clear_on_submit=True):
        ticket_content = st.text_area("Ticket content", key="tf_content")
        ticket_by = st.text_input("Ticket submitted by", value=st.session_state.get("current_user", ""), key="tf_by")
        ticket_raised_by = st.text_input("Who raised the ticket", key="tf_raised_by")
        ticket_problem = st.text_input("Describe the problem", key="tf_problem")

        submitted = st.form_submit_button("Create & Process Ticket")

        if submitted:
            if not ticket_content.strip():
                st.error("⚠️ Please enter ticket content before submitting.")
                st.stop()

            # ===================== Create Ticket =====================
            ticket = {
                "ticket_id": str(uuid.uuid4())[:8],
                "ticket_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticket_content": ticket_content.strip(),
                "ticket_problem": ticket_problem.strip() or ticket_content.strip(),
                "ticket_by": ticket_by or st.session_state.get("current_user", "User"),
                "ticket_raised_by": ticket_raised_by or "Form",
                "ticket_category": "Uncategorized",
                "ticket_solution": "",
                "ticket_status": "Open",
            }

            # Parse documents for problems and solutions
            problems_solutions = []
            if st.session_state.get("docs"):
                problems_solutions = parse_document_problems_solutions(st.session_state.docs)

            # Categorization (optional)
            if LANGCHAIN_AVAILABLE and st.session_state.get("retriever"):
                try:
                    ticket["ticket_category"] = categorize_ticket(ticket_content, st.session_state.retriever, st.session_state.get("docs"))
                except Exception as e:
                    st.warning(f"Categorization failed: {e}")

            # Match best problem and solution
            if problems_solutions:
                matched = best_match_problem_solution(ticket_content, problems_solutions)
                if matched:
                    ticket["ticket_category"], ticket["ticket_problem"], ticket["ticket_solution"] = matched

            # Auto-resolution (optional) if no match found
            if not ticket["ticket_solution"] and LANGCHAIN_AVAILABLE and st.session_state.get("retriever"):
                try:
                    ticket["ticket_solution"] = resolve_ticket(ticket_content, st.session_state.retriever, ASSISTANT_PROMPT)
                except Exception as e:
                    st.warning(f"Resolution failed: {e}")

            # Save locally
            st.session_state.queries.append(ticket)
            save_ticket(ticket)

            # ✅ Save to Google Sheets (using helper)
            if USE_SHEETS:
                try:
                    append_ticket_to_sheets(ticket)
                except Exception as e:
                    st.error(f"❌ Failed to save ticket to Google Sheets: {e}")

            st.success(f"✅ Ticket `{ticket['ticket_id']}` created → Category: {ticket['ticket_category']}")
            st.write("**Proposed Solution:**")
            st.write(ticket["ticket_solution"] or "No solution generated.")

    # ===================== Ticket History Section =====================
    st.markdown("---")
    st.subheader("📜 Recent / Past Tickets")

    tickets_data = []
    if TICKETS_FILE.exists():
        try:
            tickets_data = json.loads(TICKETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            tickets_data = []

    combined = tickets_data + st.session_state.get("queries", [])

    if combined:
        for t in list(reversed(combined[-30:])):  # Show last 30
            ticket_id = t.get("ticket_id", "?")
            category = t.get("ticket_category", "Uncategorized")
            status = t.get("ticket_status", "Open")

            with st.expander(f"🎫 {ticket_id} — {category} ({status})"):
                tc = t.get("ticket_content", "")
                tp = t.get("ticket_problem", "")
                ts = t.get("ticket_solution", "")

                new_content = st.text_area("Ticket content", tc, key=f"{ticket_id}_content")
                new_problem = st.text_input("Ticket problem description", tp, key=f"{ticket_id}_problem")

                status_options = ["Open", "In Progress", "Resolved", "Closed"]
                current_status_index = status_options.index(status) if status in status_options else 0
                new_status = st.selectbox("Update status", status_options, index=current_status_index, key=f"{ticket_id}_status")

                if st.button("💾 Save Changes", key=f"{ticket_id}_save"):
                    t["ticket_content"] = new_content
                    t["ticket_problem"] = new_problem
                    t["ticket_status"] = new_status

                    # Auto re-resolve if retriever exists
                    if LANGCHAIN_AVAILABLE and st.session_state.get("retriever"):
                        try:
                            t["ticket_solution"] = resolve_ticket(new_content, st.session_state.retriever, ASSISTANT_PROMPT)
                        except Exception as e:
                            st.error(f"Resolution failed on save: {e}")

                    # Update JSON
                    persisted = []
                    if TICKETS_FILE.exists():
                        try:
                            persisted = json.loads(TICKETS_FILE.read_text(encoding="utf-8"))
                        except Exception:
                            persisted = []

                    found = False
                    for idx, row in enumerate(persisted):
                        if row.get("ticket_id") == ticket_id:
                            persisted[idx] = t
                            found = True
                            break
                    if not found:
                        persisted.append(t)

                    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
                        json.dump(persisted, f, indent=4)

                    # ✅ Update Google Sheet row
                    if USE_SHEETS:
                        try:
                            append_ticket_to_sheets(t)
                        except Exception as e:
                            st.error(f"❌ Failed to update Google Sheets: {e}")

                    st.success(f"✅ Ticket `{ticket_id}` updated → Status: {new_status}")
    else:
        st.info("⚠️ No tickets found yet.")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import json
from pathlib import Path

def dashboard_ui():
    global TICKET_SHEET
    st.subheader("📊 My Ticket Dashboard")

    # ================== 1. Check current user ==================
    current_user = st.session_state.get("current_user", None)
    if not current_user:
        st.error("⚠️ You must be logged in to view your dashboard.")
        return

    # ================== 2. Load ticket data ==================
    if TICKET_SHEET is not None:
        try:
            data = TICKET_SHEET.get_all_records()
        except Exception as e:
            st.error(f"❌ Failed to load ticket data from Google Sheets: {e}")
            return
    else:
        if Path("tickets.json").exists():
            try:
                data = json.loads(Path("tickets.json").read_text(encoding="utf-8"))
            except Exception as e:
                st.error(f"❌ Failed to load ticket data from local file: {e}")
                return
        else:
            data = []

    if not data:
        st.info("⚠️ No tickets found yet.")
        return

    df = pd.DataFrame(data)

    # ------------------ Normalize column names ------------------
    def normalize(name: str) -> str:
        return "".join(ch for ch in str(name).lower() if ch.isalnum())

    col_map = {normalize(col): col for col in df.columns}

    def get_col(expected_key: str):
        return col_map.get(normalize(expected_key))

    desired = {
        "ticket_timestamp": get_col("ticket_timestamp"),
        "ticket_by": get_col("ticket_by"),
        "ticket_raised_by": get_col("ticket_raised_by"),
        "ticket_category": get_col("ticket_category"),
        "ticket_status": get_col("ticket_status"),
    }

    # Fallbacks and normalization
    if desired["ticket_by"] is None and desired["ticket_raised_by"] is not None:
        df["ticket_by"] = df[desired["ticket_raised_by"]].astype(str)
    elif desired["ticket_by"] is not None and desired["ticket_by"] != "ticket_by":
        df["ticket_by"] = df[desired["ticket_by"]].astype(str)

    if "ticket_by" not in df.columns:
        st.error("❌ 'ticket_by' column not found.")
        return

    if desired["ticket_status"] is None:
        df["ticket_status"] = "Unknown"
    elif desired["ticket_status"] != "ticket_status":
        df["ticket_status"] = df[desired["ticket_status"]].astype(str)

    # ================== 3. Filter by logged-in user ==================
    df["ticket_by"] = df["ticket_by"].fillna("").astype(str)
    user_df = df[df["ticket_by"].str.lower() == current_user.lower()]

    if user_df.empty:
        st.info("📭 No tickets found for your account yet.")
        return

    st.success(f"✅ Showing dashboard for **{current_user}** — Total Tickets: {len(user_df)}")

    # ================== Analytical Dashboard ==================
    st.markdown("### 📈 Ticket Analytics Overview")

    total_tickets = len(user_df)
    resolved_tickets = len(user_df[user_df["ticket_status"].str.lower() == "resolved"])
    unresolved_tickets = total_tickets - resolved_tickets

    col1, col2, col3 = st.columns(3)
    col1.metric("📬 Total Tickets", total_tickets)
    col2.metric("✅ Resolved Tickets", resolved_tickets)
    col3.metric("⏳ Unresolved Tickets", unresolved_tickets)

 

    # ================== 5. Pie Chart for Ticket Categories ==================
    st.subheader("📂 Ticket Distribution by Category")
    if desired["ticket_category"] is not None and desired["ticket_category"] != "ticket_category":
        user_df["ticket_category"] = user_df[desired["ticket_category"]].astype(str)
        desired["ticket_category"] = "ticket_category"

    user_df["ticket_category"] = user_df["ticket_category"].fillna("Uncategorized")

    if "ticket_category" in user_df.columns and user_df["ticket_category"].notna().any():
        fig_category = px.pie(
            user_df,
            names="ticket_category",
            title="Tickets by Category",
            hole=0.4
        )
        st.plotly_chart(fig_category, use_container_width=True)
    else:
        st.info("ℹ️ No category data available.")


    # ================== Line Graph: Daily Tickets ==================
    st.subheader("📅 Ticket Creation by Day & Time")
    if desired["ticket_timestamp"] is not None:
        if desired["ticket_timestamp"] != "ticket_timestamp":
            user_df["ticket_timestamp"] = user_df[desired["ticket_timestamp"]]

        user_df["ticket_timestamp"] = pd.to_datetime(user_df["ticket_timestamp"], errors="coerce")
        user_df = user_df.dropna(subset=["ticket_timestamp"])

        if not user_df.empty:
            user_df["day"] = user_df["ticket_timestamp"].dt.date
            user_df["hour"] = user_df["ticket_timestamp"].dt.hour

            # ✅ FIX: Use .size() instead of referencing non-existent columns
            daily_counts = user_df.groupby("day").size().reset_index(name="Tickets")

            fig_line = px.line(
                daily_counts,
                x="day",
                y="Tickets",
                markers=True,
                title="Tickets Created Over Time (Daily)"
            )
            fig_line.update_traces(mode="lines+markers")
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("ℹ️ Timestamp could not be parsed. Check date format.")
    else:
        st.info("ℹ️ No 'ticket_timestamp' column found.")

    # ================== 📊 Tickets per Category per Month ==================
    st.subheader("📅 Tickets per Category per Month")
    if "ticket_timestamp" in user_df.columns and "ticket_category" in user_df.columns:
        user_df["month"] = user_df["ticket_timestamp"].dt.to_period("M").astype(str)

        monthly_category_counts = (
            user_df.groupby(["month", "ticket_category"])
            .size()
            .reset_index(name="Tickets")
        )

        if not monthly_category_counts.empty:
            fig_bar = px.bar(
                monthly_category_counts,
                x="month",
                y="Tickets",
                color="ticket_category",
                title="Tickets per Category per Month",
                barmode="group"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("ℹ️ Not enough data for monthly category analytics.")
    else:
        st.info("ℹ️ Category or timestamp column not found — skipping monthly analytics.")

    # ================== 📋 Table: All Tickets ==================
    st.markdown("### 📋 All Tickets")
    if "ticket_timestamp" in user_df.columns:
        user_df = user_df.sort_values(by="ticket_timestamp", ascending=False)

    st.dataframe(user_df, use_container_width=True)


# ------------------- Pilot Validation UI -------------------
def validation_ui():
    st.header("📊 Validate Tagging & Category Accuracy")
    st.write("Run pilot validation using a dataset, the last ticket text, or ad-hoc user text.")

    validation_mode = st.radio(
        "Validation Mode:",
        ["By File Upload (Pilot Dataset)", "By Last Ticket Text", "By User Input Text"]
    )

    if validation_mode == "By File Upload (Pilot Dataset)":
        file_uploader = st.file_uploader("Upload Pilot Dataset (CSV or JSON)", type=["csv", "json"])
        if st.button("Run Pilot Validation (File)"):
            if not st.session_state.retriever and LANGCHAIN_AVAILABLE:
                st.error("⚠️ Build index first.")
            else:
                if not file_uploader:
                    st.error("⚠️ Upload a pilot dataset file first.")
                else:
                    try:
                        if file_uploader.type == "application/json" or file_uploader.name.endswith('.json'):
                            samples = json.load(file_uploader)
                            if isinstance(samples, dict):
                                for k in ("samples", "data", "items"):
                                    if k in samples and isinstance(samples[k], list):
                                        samples = samples[k]
                                        break
                        else:
                            df = pd.read_csv(file_uploader)
                            samples = df.to_dict(orient="records")
                        # Ensure sample records have 'content' and 'label' keys
                        normalized = []
                        for s in samples:
                            if isinstance(s, dict):
                                if "content" in s and "label" in s:
                                    normalized.append({"content": s["content"], "label": s["label"]})
                                elif "text" in s and "category" in s:
                                    normalized.append({"content": s["text"], "label": s["category"]})
                                else:
                                    # attempt to join available fields as content
                                    normalized.append({"content": json.dumps(s), "label": s.get("label", "")})
                        metrics = run_pilot_validation(normalized, st.session_state.retriever)
                        outfile = "pilot_validation_results.json"
                        with open(outfile, "w", encoding="utf-8") as f:
                            json.dump({"timestamp": datetime.datetime.now().isoformat(),
                                       "metrics": metrics}, f, indent=4)
                        st.json(metrics)
                        st.success(f"✅ Pilot dataset validation complete — results saved to {outfile}")
                    except Exception as e:
                        st.error(f"Pilot validation failed: {e}")

    elif validation_mode == "By Last Ticket Text":
        # Use last ticket from session or file
        last_ticket = None
        if st.session_state.queries:
            last_ticket = st.session_state.queries[-1]
        else:
            if TICKETS_FILE.exists():
                try:
                    file_tickets = json.loads(TICKETS_FILE.read_text(encoding="utf-8"))
                    if file_tickets:
                        last_ticket = file_tickets[-1]
                except Exception:
                    last_ticket = None
        if last_ticket:
            content = last_ticket.get("ticket_content") or last_ticket.get("Ticket Content")
            st.info(f"Validating last ticket content: {content[:200]}...")
            if st.button("Run Pilot Validation (Ticket Text)"):
                try:
                    samples = [{"content": content, "label": last_ticket.get("ticket_category", last_ticket.get("Ticket Category", ""))}]
                    metrics = run_pilot_validation(samples, st.session_state.retriever)
                    st.json(metrics)
                    st.success("✅ Validation complete using last ticket")
                except Exception as e:
                    st.error(f"Ticket text validation failed: {e}")
        else:
            st.warning("⚠️ No tickets found to validate.")

    elif validation_mode == "By User Input Text":
        user_text = st.text_area("Enter text to validate tagging & category accuracy", height=150)
        if st.button("Run Validation (User Input)"):
            if not st.session_state.retriever and LANGCHAIN_AVAILABLE:
                st.error("⚠️ Build index first.")
            elif not user_text.strip():
                st.error("⚠️ Enter some text first.")
            else:
                try:
                    predicted_label = categorize_ticket(user_text, st.session_state.retriever)
                    samples = [{"content": user_text, "label": predicted_label}]
                    metrics = run_pilot_validation(samples, st.session_state.retriever)
                    st.json(metrics)
                    st.success("✅ Validation complete for user input text")
                except Exception as e:
                    st.error(f"User input validation failed: {e}")


# ------------------- Admin UI -------------------
def admin_ui():
    st.header("🔒 Admin")
    st.write("User management and debug controls.")

    # User list management
    st.subheader("Users")
    users = load_users_from_file()
    df_users = pd.DataFrame([{"username": u, "name": users[u]["name"]} for u in users])
    st.dataframe(df_users)

    with st.expander("Add new user"):
        new_username = st.text_input("Username")
        new_name = st.text_input("Name")
        new_password = st.text_input("Password", type="password")
        if st.button("Create user"):
            if not new_username or not new_password:
                st.error("Provide username and password.")
            else:
                users = load_users_from_file()
                if new_username in users:
                    st.error("User exists.")
                else:
                    users[new_username] = {"name": new_name or new_username, "password": new_password}
                    save_users_to_file(users)
                    st.success(f"Created user {new_username}. Re-login may be required.")
                    st.experimental_rerun()

    st.markdown("---")
    st.subheader("Debug & Index")
    if st.button("Delete local FAISS index (danger)"):
        try:
            import shutil
            if INDEX_PATH.exists():
                shutil.rmtree(INDEX_PATH)
            st.success("Deleted FAISS index.")
        except Exception as e:
            st.error(f"Delete failed: {e}")

    if st.button("Show session_state keys"):
        st.write(dict(st.session_state))

def signup_ui():
    st.subheader("📝 Create a new account")
    name = st.text_input("Full Name")
    username = st.text_input("Choose Username")
    password = st.text_input("Choose Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Signup"):
        if password != confirm:
            st.error("Passwords do not match ❌")
        else:
            # Check if user exists
            users = load_users()
            if username in users:
                st.error("Username already exists 🚫")
            else:
                USER_SHEET.append_row([name, username, password])
                st.success("Account created ✅ Please login now.")

# ----------------------------
# Signin
# ----------------------------
def signin_ui():
    st.subheader("🔑 Login to your account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        users = load_users()
        if username in users and users[username]["password"] == password:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["name"] = users[username]["name"]
            st.success(f"Welcome {users[username]['name']} 👋")
        else:
            st.error("Invalid username or password ❌")

# ----------------------------
# Logout
# ----------------------------
def logout_ui():
    if st.button("Logout"):
        st.session_state.clear()
        st.success("Logged out successfully ✅")


# ---------------- Local Fallback ----------------
TICKETS_FILE = Path("tickets.json")

def load_tickets():
    global sheet

    if sheet is None:
        st.error("⚠️ Google Sheet not connected. Cannot load tickets.")
        return []

    try:
        records = sheet.get_all_records()
        return records
    except Exception as e:
        st.error(f"❌ Failed to load tickets from Google Sheets: {e}")
        return []


import os
from pathlib import Path
import gspread

TICKET_SHEET = None  # define globally once

def init_google_sheet():
    global TICKET_SHEET
    try:
        creds_file = os.getenv("GOOGLE_SHEET_CREDS", "service_account.json")
        sheet_name = os.getenv("TICKET_SHEET_NAME", "SupportTickets")

        if Path(creds_file).exists():
            gc = gspread.service_account(filename=creds_file)
            sh = gc.open(sheet_name)
            TICKET_SHEET = sh.sheet1
            print(f"✅ Connected to Google Sheets → {TICKET_SHEET.title}")
        else:
            print("⚠️ No creds file found → using local tickets.json")
            TICKET_SHEET = None

    except Exception as e:
        print(f"⚠️ Google Sheets setup failed: {e}")
        TICKET_SHEET = None



def run_async(func, *args, **kwargs):
    import asyncio
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Already running (Streamlit server), use nest_asyncio
        import nest_asyncio
        nest_asyncio.apply()
        return asyncio.get_event_loop().run_until_complete(func(*args, **kwargs))
    else:
        return asyncio.run(func(*args, **kwargs))



# ------------------- Layout & Routing -------------------
def main_app():
    # ---------------- Init ----------------
    init_state()
    sidebar_settings()
    sidebar_documents_uploader()
    init_google_sheet()  # 🔹 ensures TICKET_SHEET or local fallback is ready

    # ---------------- Authentication ----------------
    st.sidebar.title("🔐 Authentication")

    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    login_success = False
    name = None

    # Case 1: Using streamlit-authenticator (if configured)
    if authenticator:
        try:
            name, authentication_status, username = authenticator.login("Login", "sidebar")

            if authentication_status:
                login_success = True
                st.session_state.current_user = username
                st.sidebar.success(f"Welcome, {name}! 🎉")
                authenticator.logout("Logout", "sidebar")

            elif authentication_status is False:
                st.sidebar.error("❌ Username/password is incorrect")

            elif authentication_status is None:
                st.sidebar.info("Please enter your username and password")

        except Exception as e:
            st.sidebar.error(f"⚠️ Authentication error: {e}")
            login_success = False

    # Case 2: Local auth (Signup + Signin stored in JSON)
    else:
        choice = st.sidebar.radio("Choose action:", ["Login", "Signup"], horizontal=True)

        if choice == "Login":
            with st.sidebar.form("local_login"):
                lu = st.text_input("Username")
                lp = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Login")

                if submitted:
                    users = load_users_from_file()
                    if lu in users and users[lu]["password"] == lp:
                        st.session_state.current_user = lu
                        st.sidebar.success(f"✅ Welcome, {users[lu]['name']}!")
                        login_success = True
                    else:
                        st.sidebar.error("❌ Invalid credentials")

        elif choice == "Signup":
            with st.sidebar.form("local_signup"):
                name = st.text_input("Full Name")
                su = st.text_input("Choose Username")
                sp = st.text_input("Choose Password", type="password")
                cp = st.text_input("Confirm Password", type="password")
                submitted = st.form_submit_button("Create Account")

                if submitted:
                    users = load_users_from_file()
                    if su in users:
                        st.sidebar.error("🚫 Username already exists. Try another.")
                    elif sp != cp:
                        st.sidebar.error("❌ Passwords do not match")
                    else:
                        users[su] = {"name": name, "password": sp}
                        save_users_to_file(users)
                        st.sidebar.success("🎉 Account created. Please log in now.")

    # ---------------- Post-authentication ----------------
    if not st.session_state.current_user:
        st.sidebar.info("ℹ️ You are not logged in. Use the authentication controls above.")
        st.stop()

    user = st.session_state.current_user
    st.sidebar.success(f"👤 Logged in as: {user}")

    # ---------------- Tabs Layout ----------------
    tab_titles = ["Dashboard", "Chatbot", "Tickets", "Documents", "Validation", "Admin"]
    tabs = st.tabs(tab_titles)

    # Dashboard
    with tabs[0]:
        dashboard_ui()

    # Chatbot
    with tabs[1]:
        chatbot_ui()

    # Tickets
    with tabs[2]:
        try:
            tickets_ui()
        except Exception as e:
            st.error(f"⚠️ Ticket system error: {e}")

    # Documents
    with tabs[3]:
        docs_ui()

    # Validation
    with tabs[4]:
        validation_ui()

    # Admin
    with tabs[5]:
        if user and (user == "admin" or st.session_state.get("admin_mode", False)):
            admin_ui()
        else:
            st.warning("⚠️ Admin access restricted. Log in as admin or enable `admin_mode`.")

    # ---------------- Footer ----------------
    st.markdown("---")
    st.caption("📘 Support Knowledge Assistant — Multi-page Streamlit application. Configure APIs in `.env` and ensure dependencies are installed.")


if __name__ == "__main__":
    main_app()
