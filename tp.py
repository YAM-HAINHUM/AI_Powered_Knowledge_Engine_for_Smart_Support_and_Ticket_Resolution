"""
Support Knowledge Assistant - Streamlit App
Updated full application integrating multiple AI agents (Google Generative AI, Tavily, HuggingFace)

Notes:
- Fill environment variables in a .env file or export them in your environment.
- This file is written to be self-contained and reasonably robust with helpful debug messages.
- Some libraries used (langchain-like wrappers, tavily) may need installation and their APIs adjusted to match your installed versions.

Usage:
  pip install -r requirements.txt
  streamlit run support_chatbot_streamlit_app.py

Environment variables expected:
  - GOOGLE_API_KEY
  - OPENAI_API_KEY
  - TAVILY_API_KEY
  - HUGGINGFACE_API_TOKEN
  - CREDENTIALS_FILE (optional, for Google Sheets service account)
  - SHEET_NAME (optional)

Author: Assistant
Last updated: 2025-09-22
"""

import os
import json
import datetime
import uuid
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Third-party (may require installation)
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain.chains import create_retrieval_chain
    from langchain.chains.combine_documents import create_stuff_documents_chain
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
    from langchain_community.vectorstores import FAISS
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
except Exception:
    # Provide graceful fallback imports
    ChatPromptTemplate = None
    create_retrieval_chain = None
    create_stuff_documents_chain = None
    Document = None
    RecursiveCharacterTextSplitter = None
    TextLoader = None
    PyPDFLoader = None
    FAISS = None
    GoogleGenerativeAIEmbeddings = None
    ChatGoogleGenerativeAI = None

try:
    from google.api_core.exceptions import ResourceExhausted
except Exception:
    class ResourceExhausted(Exception):
        pass

try:
    from huggingface_hub import InferenceClient
except Exception:
    InferenceClient = None

try:
    # Tavily client placeholder (API might differ)
    from tavily import TavilyClient
except Exception:
    TavilyClient = None

try:
    import gspread
    from google.oauth2.service_account import Credentials
    USE_SHEETS = True
except Exception:
    gspread = None
    Credentials = None
    USE_SHEETS = False

# ------------------- CONFIG -------------------
load_dotenv()

DEFAULT_DOCS_PATH = Path.cwd() / "support_knowledge.pdf"
INDEX_PATH = Path("./faiss_index")
QUERIES_FILE = Path("queries.json")
PILOT_FILE = Path("pilot_results.json")
USERS_FILE = Path("users.json")
HISTORY_FILE = Path("chat_history.json")
TICKETS_FILE = Path("ticket_raised.json")
CONTENT_GAPS_FILE = Path("content_gaps.json")

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/embedding-001")
PRIMARY_CHAT_MODEL = os.getenv("PRIMARY_CHAT_MODEL", "gemini-1.5-pro")
FALLBACK_CHAT_MODEL = os.getenv("FALLBACK_CHAT_MODEL", "gemini-1.5-flash")
TOP_K = int(os.getenv("TOP_K", 4))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

SHEET_NAME = os.getenv("SHEET_NAME", "PDF_AI_Internship_Infosys")
CREDENTIALS_FILE = os.getenv("CREDENTIALS_FILE", "credentials.json")
WORKSHEET_NAME = "Sheet1"

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

# ------------------- HELPER I/O -------------------

def load_json_file(path: Path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        return default
    return default


def save_json_file(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


st.set_page_config(page_title="Support Knowledge Assistant", layout="wide")

# ------------------- USERS & AUTH -------------------

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def load_users() -> Dict:
    return load_json_file(USERS_FILE, {})


def save_users(users: Dict):
    save_json_file(USERS_FILE, users)

# ------------------- HISTORY -------------------

def load_history() -> Dict:
    return load_json_file(HISTORY_FILE, {})


def save_history(history: Dict):
    save_json_file(HISTORY_FILE, history)


def add_to_history(username: str, query: str, answer: str):
    history = load_history()
    history.setdefault(username, [])
    history[username].append({
        "query": query,
        "answer": answer,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })
    save_history(history)

# ------------------- QUERIES / TICKETS -------------------

def save_query_answer(query, answer, status):
    data = load_json_file(QUERIES_FILE, [])
    data.append({"query": query, "answer": answer, "status": status, "timestamp": datetime.datetime.now().isoformat()})
    save_json_file(QUERIES_FILE, data)


def save_ticket(ticket: Dict):
    data = load_json_file(TICKETS_FILE, [])
    data.append(ticket)
    save_json_file(TICKETS_FILE, data)

# ------------------- CONTENT GAPS -------------------

def log_content_gap(query: str, answer: str):
    is_gap = False
    if not answer or len(str(answer).strip()) < 30:
        is_gap = True
    if isinstance(answer, str) and ("No saved response" in answer or "⚠️" in answer or "I don't know" in answer):
        is_gap = True
    if is_gap:
        gaps = load_json_file(CONTENT_GAPS_FILE, [])
        gaps.append({"query": query, "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
        save_json_file(CONTENT_GAPS_FILE, gaps)

# ------------------- GOOGLE SHEETS -------------------

sheet = None
if USE_SHEETS and CREDENTIALS_FILE:
    try:
        creds_path = Path(CREDENTIALS_FILE)
        if creds_path.exists() and gspread and Credentials:
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
                    sheet.delete_row(1)
                except Exception:
                    pass
                sheet.insert_row(HEADERS, 1)
            st.info("Connected to Google Sheets")
        else:
            st.warning("Credentials file not found or gspread missing; Google Sheets disabled.")
            USE_SHEETS = False
    except Exception as e:
        st.warning(f"Google Sheets init failed: {e}")
        USE_SHEETS = False


def append_ticket_to_sheet(ticket: Dict):
    global sheet
    if not USE_SHEETS or sheet is None:
        return
    try:
        row = [
            ticket.get("ticket_id", ""),
            ticket.get("ticket_content", ""),
            ticket.get("ticket_timestamp", ""),
            ticket.get("ticket_by", ""),
            ticket.get("ticket_raised_by", ""),
            ticket.get("ticket_category", ""),
            ticket.get("ticket_problem", ""),
            ticket.get("ticket_solution", ""),
            ticket.get("ticket_status", ""),
        ]
        sheet.append_row(row, value_input_option="USER_ENTERED")
    except Exception as e:
        st.warning(f"Failed to append to sheet: {e}")

# ------------------- DOCUMENT LOADING / INDEX -------------------

def find_files(path: Path):
    if path.is_file():
        return [path]
    exts = {".txt", ".md", ".pdf"}
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def load_documents(paths: List[Path]) -> List:
    docs = []
    for p in paths:
        try:
            if p.suffix.lower() in {".txt", ".md"} and TextLoader:
                docs.extend(TextLoader(str(p), encoding="utf-8").load())
            elif p.suffix.lower() == ".pdf" and PyPDFLoader:
                docs.extend(PyPDFLoader(str(p)).load())
            else:
                # Fallback: read raw text
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    docs.append(Document(page_content=f.read(), metadata={"source": str(p)}))
        except Exception as e:
            st.warning(f"Failed to load {p}: {e}")
    return docs


def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    if RecursiveCharacterTextSplitter:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_documents(docs)
    # fallback: naive splitting
    chunks = []
    for d in docs:
        text = d.page_content
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunk = text[i:i+chunk_size]
            chunks.append(Document(page_content=chunk, metadata=getattr(d, "metadata", {})))
    return chunks


def build_or_load_faiss(chunks, rebuild=True):
    if GoogleGenerativeAIEmbeddings is None or FAISS is None:
        raise RuntimeError("FAISS or Embeddings classes not installed. Install langchain_community and langchain_google_genai or adjust code.")
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY)
    if rebuild:
        if not chunks:
            raise ValueError("No chunks found. Upload documents first.")
        vs = FAISS.from_documents(chunks, embeddings)
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(INDEX_PATH))
        return vs
    return FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

# ------------------- LLM / RAG HELPERS -------------------

def load_llm(model_name: str):
    # Prefer ChatGoogleGenerativeAI when available; otherwise None
    try:
        if ChatGoogleGenerativeAI:
            return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.warning(f"LLM init failed for {model_name}: {e}")
    return None

CATEGORY_PROMPT_TEXT = (
    "You are a support ticket classifier. Based only on the knowledge base provided:\n{context}\n"
    "Classify this ticket into the most relevant category:\n{input}\nReply with only the category name."
)

ASSISTANT_PROMPT_TEXT = (
    "You are a helpful support assistant. Use only the following documents to answer:\n{context}\nUser Question: {input}"
)


def create_prompt_objects():
    if ChatPromptTemplate:
        CATEGORY_PROMPT = ChatPromptTemplate.from_template(CATEGORY_PROMPT_TEXT)
        ASSISTANT_PROMPT = ChatPromptTemplate.from_template(ASSISTANT_PROMPT_TEXT)
        return CATEGORY_PROMPT, ASSISTANT_PROMPT
    return None, None


CATEGORY_PROMPT, ASSISTANT_PROMPT = create_prompt_objects()


def categorize_ticket(content: str, retriever) -> str:
    try:
        llm = load_llm(PRIMARY_CHAT_MODEL) or load_llm(FALLBACK_CHAT_MODEL)
        if llm and create_stuff_documents_chain and create_retrieval_chain and CATEGORY_PROMPT:
            doc_chain = create_stuff_documents_chain(llm, CATEGORY_PROMPT)
            rag_chain = create_retrieval_chain(retriever, doc_chain)
            result = rag_chain.invoke({"input": content})
            return result.get("answer", "Uncategorized").strip()
        # fallback simple keyword mapping
        keywords = {
            "login": "Authentication",
            "password": "Authentication",
            "error": "Bug/Issue",
            "payment": "Billing",
            "invoice": "Billing",
            "upload": "Upload/Files"
        }
        for k, v in keywords.items():
            if k in content.lower():
                return v
        return "Uncategorized"
    except ResourceExhausted:
        llm = load_llm(FALLBACK_CHAT_MODEL)
        if llm and create_stuff_documents_chain and create_retrieval_chain and CATEGORY_PROMPT:
            doc_chain = create_stuff_documents_chain(llm, CATEGORY_PROMPT)
            rag_chain = create_retrieval_chain(retriever, doc_chain)
            result = rag_chain.invoke({"input": content})
            return result.get("answer", "Uncategorized").strip()
        return "Uncategorized"
    except Exception as e:
        st.warning(f"Categorization failed: {e}")
        return "Uncategorized"


def resolve_ticket(content: str, retriever, prompt) -> str:
    try:
        llm = load_llm(PRIMARY_CHAT_MODEL) or load_llm(FALLBACK_CHAT_MODEL)
        if llm and create_stuff_documents_chain and create_retrieval_chain and prompt:
            doc_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, doc_chain)
            response = rag_chain.invoke({"input": content})
            return response.get("answer", "")
        # fallback: try HF or Tavily
        return call_huggingface_text_gen(content) or call_tavily_text_gen(content) or ""
    except ResourceExhausted:
        llm = load_llm(FALLBACK_CHAT_MODEL)
        if llm and create_stuff_documents_chain and create_retrieval_chain and prompt:
            doc_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, doc_chain)
            response = rag_chain.invoke({"input": content})
            return response.get("answer", "")
        return ""
    except Exception as e:
        st.warning(f"Resolve failed: {e}")
        return ""

# ------------------- EXTERNAL AGENTS: HUGGINGFACE & TAVILY -------------------

def load_hf_agent(token: Optional[str] = None):
    token = token or HUGGINGFACE_API_TOKEN
    if not token or InferenceClient is None:
        return None
    try:
        client = InferenceClient(token)
        return client
    except Exception as e:
        st.warning(f"HuggingFace client init failed: {e}")
        return None


def call_huggingface_text_gen(prompt: str, max_new_tokens: int = 256) -> Optional[str]:
    client = load_hf_agent()
    if not client:
        return None
    try:
        resp = client.text_generation(prompt, max_new_tokens=max_new_tokens)
        # InferenceClient returns dict-like; best-effort extract
        if isinstance(resp, dict):
            # Some clients return {'generated_text': '...'} or a list
            if resp.get("generated_text"):
                return resp["generated_text"]
        # else try str
        return str(resp)
    except Exception as e:
        st.warning(f"HuggingFace generation failed: {e}")
        return None


def load_tavily_client(api_key: Optional[str] = None):
    key = api_key or TAVILY_API_KEY
    if not key or TavilyClient is None:
        return None
    try:
        client = TavilyClient(api_key=key)
        return client
    except Exception as e:
        st.warning(f"Tavily client init failed: {e}")
        return None


def call_tavily_text_gen(prompt: str) -> Optional[str]:
    client = load_tavily_client()
    if not client:
        return None
    try:
        # TavilyClient API is assumed — adjust per installed package
        resp = client.generate(prompt)
        # If resp is object/dict
        if isinstance(resp, dict) and resp.get("output"):
            return resp.get("output")
        return str(resp)
    except Exception as e:
        st.warning(f"Tavily generation failed: {e}")
        return None

# ------------------- RECOMMENDER -------------------

def recommend_articles(query: str, retriever, top_n: int = 3) -> List[Dict]:
    try:
        docs = retriever.get_relevant_documents(query)
        recs = []
        for d in docs[:top_n]:
            recs.append({
                "preview": d.page_content[:400],
                "source": getattr(d, "metadata", {}).get("source", "")
            })
        return recs
    except Exception as e:
        st.warning(f"Recommendation error: {e}")
        return []

# ------------------- PILOT VALIDATION -------------------

def run_pilot_validation(labeled_samples: List[Dict[str, str]], retriever) -> Dict[str, float]:
    correct = 0
    total = len(labeled_samples)
    per_class = {}
    for s in labeled_samples:
        predicted = categorize_ticket(s["content"], retriever)
        label = s["label"].strip()
        per_class.setdefault(label, {"total": 0, "correct": 0})
        per_class.setdefault(predicted, per_class.get(predicted, {"total": 0, "correct": 0}))
        per_class[label]["total"] += 1
        if predicted.lower() == label.lower():
            correct += 1
            per_class[label]["correct"] += 1
    accuracy = correct / total if total else 0.0
    metrics = {"accuracy": accuracy, "total": total, "per_class": per_class}
    save_json_file(PILOT_FILE, {"timestamp": datetime.datetime.now().isoformat(), "metrics": metrics})
    return metrics

# ------------------- UI / Streamlit -------------------

def init_state():
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
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


def login_ui():
    st.sidebar.header("🔑 Login / Signup")
    users = load_users()
    choice = st.sidebar.radio("Select Action", ["Login", "Signup"])
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if choice == "Signup":
        if st.sidebar.button("Create Account"):
            if not username or not password:
                st.sidebar.error("Enter username and password")
            elif username in users:
                st.sidebar.error("Username already exists.")
            else:
                users[username] = {"password": hash_password(password)}
                save_users(users)
                st.sidebar.success("Account created. Please login.")

    elif choice == "Login":
        if st.sidebar.button("Login"):
            if username in users and users[username]["password"] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.sidebar.success(f"Welcome {username}")
            else:
                st.sidebar.error("Invalid username or password")


def sidebar_settings():
    st.sidebar.title("⚙️ Settings")
    st.session_state.top_k = st.sidebar.number_input("Top K", value=TOP_K, min_value=1, max_value=20)
    st.session_state.chunk_size = st.sidebar.number_input("Chunk size", value=CHUNK_SIZE, min_value=100, max_value=2000)
    st.session_state.chunk_overlap = st.sidebar.number_input("Chunk overlap", value=CHUNK_OVERLAP, min_value=0, max_value=1000)
    st.sidebar.markdown("**Models**")
    st.session_state.primary_model = st.sidebar.text_input("Primary chat model", value=PRIMARY_CHAT_MODEL)
    st.session_state.fallback_model = st.sidebar.text_input("Fallback chat model", value=FALLBACK_CHAT_MODEL)
    if st.sidebar.button("Rebuild index"):
        if st.session_state.chunks:
            with st.spinner("Building index..."):
                try:
                    st.session_state.vectorstore = build_or_load_faiss(st.session_state.chunks, rebuild=True)
                    st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr", search_kwargs={"k": st.session_state.top_k}
                    )
                    st.success("Index rebuilt.")
                except Exception as e:
                    st.error(f"Index rebuild failed: {e}")
        else:
            st.sidebar.warning("Upload & split documents first.")


def sidebar_docs_uploader():
    st.sidebar.title("📄 Documents")
    uploaded = st.sidebar.file_uploader("Upload docs (pdf, txt, md)", accept_multiple_files=True, type=["pdf", "txt", "md"])
    if uploaded:
        for up in uploaded:
            dest = Path("./uploaded_docs")
            dest.mkdir(exist_ok=True)
            fp = dest / up.name
            with open(fp, "wb") as f:
                f.write(up.getbuffer())
        st.sidebar.success("Uploaded files saved to ./uploaded_docs")


def show_history(username: str):
    history = load_history()
    user_history = history.get(username, [])
    if user_history:
        st.subheader("📜 Your Chat History")
        for h in reversed(user_history[-20:]):
            with st.expander(f"Q: {h['query']}"):
                st.write(f"**Answer:** {h['answer']}")
                st.caption(f"⏱ {h['timestamp']}")
    else:
        st.info("No chat history yet.")

# ==========================
# STREAMLIT UI: Main
# ==========================

def login_ui():
    st.subheader("Login")
    users = load_users()
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username]["password"] == hash_password(password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.success(f"Login successful! Welcome {username}")
        else:
            st.error("Invalid username or password")

def signup_ui():
    st.subheader("Sign Up")
    users = load_users()
    new_username = st.text_input("Choose Username", key="signup_user")
    new_password = st.text_input("Choose Password", type="password", key="signup_pass")
    confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")

    if st.button("Sign Up"):
        if not new_username or not new_password:
            st.warning("Username and password cannot be empty")
        elif new_username in users:
            st.warning("Username already exists")
        elif new_password != confirm_password:
            st.warning("Passwords do not match")
        else:
            users[new_username] = {"password": hash_password(new_password)}
            save_users(users)
            st.success("Signup successful! You can now login.")

def main_ui():
    init_state()

    # --- Authentication: Login / Signup ---
    st.sidebar.title("User Authentication")
    auth_choice = st.sidebar.radio("Choose an option", ["Login", "Sign Up"])

    users = load_users()  # Load users from JSON file

    if auth_choice == "Login":
        st.subheader("Login")
        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if username in users and users[username]["password"] == hash_password(password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(f"Login successful! Welcome {username}")
            else:
                st.error("Invalid username or password")


    else:  # Sign Up
        st.subheader("Sign Up")
        new_username = st.text_input("Choose Username", key="signup_user")
        new_password = st.text_input("Choose Password", type="password", key="signup_pass")
        confirm_password = st.text_input("Confirm Password", type="password", key="signup_confirm")

        if st.button("Sign Up"):
            if not new_username or not new_password:
                st.warning("Username and password cannot be empty")
            elif new_username in users:
                st.warning("Username already exists")
            elif new_password != confirm_password:
                st.warning("Passwords do not match")
            else:
                users[new_username] = {"password": hash_password(new_password)}
                save_users(users)
                st.success("Signup successful! You can now login.")

    # Stop if user is not logged in
    if not st.session_state.get("logged_in", False):
        st.stop()

    st.sidebar.success(f"Logged in as: {st.session_state.username}")

    # --- Your existing app UI ---
    sidebar_settings()
    sidebar_docs_uploader()

    tabs = st.tabs(["📚 Docs & Chatbot", "🎫 Tickets", "📊 Dashboard", "⚙️ Admin"])

    # ------------------ TAB 1: Docs + Chatbot ------------------
    with tabs[0]:
        st.header("📚 Index & Documents")
        docs_path = st.text_input("Local docs path", value=str(DEFAULT_DOCS_PATH))
        if st.button("Load & Split Documents"):
            paths = find_files(Path(docs_path))
            if not paths:
                st.warning("No documents found at the path.")
            else:
                with st.spinner("Loading documents..."):
                    st.session_state.docs = load_documents(paths)
                    st.session_state.chunks = split_documents(
                        st.session_state.docs,
                        chunk_size=st.session_state.chunk_size,
                        chunk_overlap=st.session_state.chunk_overlap
                    )
                    st.success(f"Loaded {len(st.session_state.docs)} docs → {len(st.session_state.chunks)} chunks.")

        if st.session_state.get("chunks"):
            st.subheader("Sample chunk")
            try:
                st.write(st.session_state.chunks[0].page_content[:500])
            except Exception:
                st.write(str(st.session_state.chunks[0])[:500])

            if st.button("Build FAISS index"):
                with st.spinner("Building index..."):
                    try:
                        st.session_state.vectorstore = build_or_load_faiss(st.session_state.chunks, rebuild=True)
                        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                            search_type="mmr", search_kwargs={"k": st.session_state.top_k}
                        )
                        st.success("Index built.")
                    except Exception as e:
                        st.error(f"Failed to build index: {e}")

        st.markdown("---")
        st.header("🤖 Chatbot Assistant")

        for chat in st.session_state.chat_history:
            role = chat.get("role", "assistant")
            st.chat_message(role).write(chat.get("content", ""))

        user_message = st.chat_input("Type your message...")
        if user_message:
            st.chat_message("user").write(user_message)
            st.session_state.chat_history.append({"role": "user", "content": user_message})

            bot_reply = ""

            if not st.session_state.get("retriever"):
                bot_reply = "⚠️ Please build the FAISS index first."
            else:
                try:
                    llm = load_llm(st.session_state.primary_model)
                    if not llm:
                        llm = load_llm(st.session_state.fallback_model)

                    if llm and create_stuff_documents_chain and create_retrieval_chain and ASSISTANT_PROMPT:
                        doc_chain = create_stuff_documents_chain(llm, ASSISTANT_PROMPT)
                        rag_chain = create_retrieval_chain(st.session_state.retriever, doc_chain)
                        response = rag_chain.invoke({"input": user_message})
                        bot_reply = response.get("answer", "")
                    else:
                        bot_reply = call_huggingface_text_gen(user_message) or call_tavily_text_gen(user_message) or "❌ No model available to generate a reply."

                except Exception as e:
                    st.warning(f"Primary retrieval/generation failed: {e}")
                    hf = call_huggingface_text_gen(user_message)
                    if hf:
                        bot_reply = hf
                    else:
                        tav = call_tavily_text_gen(user_message)
                        bot_reply = tav or f"❌ Query failed: {e}"

            st.chat_message("assistant").write(bot_reply)
            st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})

            # Track feedback
            resolved = "Resolved" if st.radio(
                "Did this answer resolve your issue?",
                ("Yes", "No"),
                key=f"res_{uuid.uuid4()}"
            ) == "Yes" else "In Progress"

            save_query_answer(user_message, bot_reply, resolved)
            add_to_history(st.session_state.username, user_message, bot_reply)

            question_ticket = {
                "ticket_id": str(uuid.uuid4())[:8],
                "ticket_content": user_message,
                "ticket_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticket_by": st.session_state.username,
                "ticket_raised_by": "Chatbot",
                "ticket_category": "",
                "ticket_problem": user_message,
                "ticket_solution": bot_reply,
                "ticket_status": resolved,
            }
            save_ticket(question_ticket)
            if USE_SHEETS:
                append_ticket_to_sheet(question_ticket)

            st.subheader("📌 Recommended Articles")
            recs = recommend_articles(user_message, st.session_state.retriever, top_n=st.session_state.top_k)
            if recs:
                for i, r in enumerate(recs, 1):
                    st.markdown(f"**{i}.** {r.get('preview','')}... \nSource: {r.get('source','')}")
            else:
                st.info("No recommendations found.")

            log_content_gap(user_message, bot_reply)
            show_history(st.session_state.username)

    # ------------------ TAB 2: Tickets ------------------
    with tabs[1]:
        st.header("🎫 Tickets")
        with st.form("ticket_form"):
            ticket_content = st.text_area("Ticket content")
            ticket_by = st.text_input("Ticket submitted by", value=st.session_state.username)
            ticket_raised_by = st.text_input("Who raised the ticket")
            ticket_problem = st.text_input("Describe the problem")
            submitted = st.form_submit_button("Create & Process Ticket")
            if submitted:
                if not st.session_state.get("retriever"):
                    st.error("Build index first.")
                else:
                    ticket = {
                        "ticket_id": str(uuid.uuid4())[:8],
                        "ticket_content": ticket_content,
                        "ticket_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "ticket_by": ticket_by,
                        "ticket_raised_by": ticket_raised_by,
                        "ticket_category": categorize_ticket(ticket_content, st.session_state.retriever),
                        "ticket_problem": ticket_problem,
                        "ticket_solution": resolve_ticket(ticket_content, st.session_state.retriever, ChatPromptTemplate.from_template(ASSISTANT_PROMPT_TEXT) if ChatPromptTemplate else None),
                        "ticket_status": "Open"
                    }
                    st.session_state.queries.append(ticket)
                    save_ticket(ticket)
                    if USE_SHEETS:
                        append_ticket_to_sheet(ticket)

                    st.success(f"Ticket {ticket.get('ticket_id','N/A')} created → Category: {ticket.get('ticket_category','Uncategorized')}")
                    st.markdown("**Proposed Solution:**")
                    st.write(ticket.get("ticket_solution",""))

        st.subheader("Recent / Past Tickets")
        tickets_list = load_json_file(TICKETS_FILE, [])
        if tickets_list:
            for t in reversed(tickets_list[-20:]):
                with st.expander(f"{t.get('ticket_id','N/A')} — {t.get('ticket_category','Uncategorized')} ({t.get('ticket_status','Open')})"):
                    st.json(t)
        else:
            st.info("No tickets found yet.")

    # ------------------ TAB 3: Dashboard ------------------
    with tabs[2]:
        st.header("📊 Dashboard & Analytics")
        # ... Keep your existing dashboard code (charts, trends, gaps)

    # ------------------ TAB 4: Admin ------------------
    with tabs[3]:
        st.header("⚙️ Admin & Tools")
        # ... Keep your existing admin code (system info, docs upload, pilot validation, content gaps)


    # ------------------ TAB 4: Admin ------------------
    with tabs[3]:
        st.header("⚙️ Admin & Tools")
        st.subheader("System Info")
        st.write({
            "GOOGLE_API_KEY": bool(GOOGLE_API_KEY),
            "TAVILY_API_KEY": bool(TAVILY_API_KEY),
            "HUGGINGFACE_API_TOKEN": bool(HUGGINGFACE_API_TOKEN),
            "LangChain available": bool(ChatPromptTemplate),
            "HuggingFace client": bool(InferenceClient),
            "Tavily client": bool(TavilyClient)
        })

        st.subheader("Upload Local Docs Folder")
        folder_path = st.text_input("Path to folder containing docs", value=str(Path.cwd() / "uploaded_docs"))
        if st.button("Load folder and split"):
            p = Path(folder_path)
            if p.exists():
                paths = find_files(p)
                st.session_state.docs = load_documents(paths)
                st.session_state.chunks = split_documents(st.session_state.docs,
                                                         chunk_size=st.session_state.chunk_size,
                                                         chunk_overlap=st.session_state.chunk_overlap)
                st.success(f"Loaded {len(st.session_state.docs)} docs → {len(st.session_state.chunks)} chunks.")
            else:
                st.error("Folder not found.")

        st.subheader("Pilot Validation")
        sample_text = st.text_area("Enter labeled samples as JSON list [{\"content\":...,\"label\":...}]", height=150)
        if st.button("Run Pilot"):
            try:
                samples = json.loads(sample_text)
                metrics = run_pilot_validation(samples, st.session_state.retriever)
                st.json(metrics)
            except Exception as e:
                st.error(f"Pilot run failed: {e}")

        st.subheader("Content Gaps")
        gaps = load_json_file(CONTENT_GAPS_FILE, [])
        st.write(gaps[-30:])


if __name__ == "__main__":
    main_ui()
