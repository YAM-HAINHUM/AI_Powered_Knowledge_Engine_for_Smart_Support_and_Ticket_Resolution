import os
import json
import datetime
import uuid
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Dict
from oauth2client.service_account import ServiceAccountCredentials

import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from google.api_core.exceptions import ResourceExhausted

# Optional: Google Sheets
try:
    import gspread 
    from oauth2client.service_account import ServiceAccountCredentials
    USE_SHEETS = True
except ImportError:
    USE_SHEETS = False

# ------------------- CONFIG -------------------
from pathlib import Path
import os
import streamlit as st
from dotenv import load_dotenv

from pathlib import Path
import os
import streamlit as st
from dotenv import load_dotenv

# ------------------- CONFIG -------------------
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
from google.api_core.exceptions import ResourceExhausted

# Load environment variables
load_dotenv()

DEFAULT_DOCS_PATH = Path.cwd() / "support_knowledge.pdf"
INDEX_PATH = Path("./faiss_index")
QUERIES_FILE = Path("queries.json")
PILOT_FILE = Path("pilot_results.json")

EMBED_MODEL = os.getenv("EMBED_MODEL", "models/embedding-001")

# ✅ Primary: Gemini Pro | Fallback: Gemini Flash
PRIMARY_CHAT_MODEL = os.getenv("PRIMARY_CHAT_MODEL", "gemini-1.5-pro")
FALLBACK_CHAT_MODEL = os.getenv("FALLBACK_CHAT_MODEL", "gemini-1.5-flash")

TOP_K = int(os.getenv("TOP_K", 4))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 120))

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.warning("⚠️ GOOGLE_API_KEY not found in environment. Set it in `.env`.")

# ------------------- MODEL LOADER -------------------
from langchain_google_genai import ChatGoogleGenerativeAI


def get_chat_model():
    """Try Gemini Pro, fall back to Gemini Flash, then JSON."""
    if GOOGLE_API_KEY:
        try:
            return ChatGoogleGenerativeAI(
                model=PRIMARY_CHAT_MODEL,
                google_api_key=GOOGLE_API_KEY
            )
        except ResourceExhausted:
            st.warning(f"⚠️ Quota exceeded for {PRIMARY_CHAT_MODEL}, switching to {FALLBACK_CHAT_MODEL}...")
        except Exception as e1:
            st.warning(f"⚠️ Gemini model {PRIMARY_CHAT_MODEL} failed: {e1}")

        try:
            return ChatGoogleGenerativeAI(
                model=FALLBACK_CHAT_MODEL,
                google_api_key=GOOGLE_API_KEY
            )
        except Exception as e2:
            st.error(f"🚨 Gemini fallback model {FALLBACK_CHAT_MODEL} failed: {e2}")

    # Fallback to local JSON if no Gemini works
    if QUERIES_FILE.exists():
        st.warning("📂 Using local JSON fallback.")
        class LocalJSONModel:
            def __init__(self, file):
                with open(file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)

            def invoke(self, input_text):
                return {"content": self.data.get(input_text, "⚠️ No saved response found.")}

        return LocalJSONModel(QUERIES_FILE)

    st.error("🚨 No fallback available. Please check Google API key or queries.json.")
    return None


# ------------------- SHEETS CONFIG -------------------
SHEET_NAME = os.getenv("SHEET_NAME", "PDF_AI_Internship_Infosys")  # full spreadsheet name
WORKSHEET_NAME = "Sheet1"  # ✅ fixed
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


# ------------------- GOOGLE SHEETS SETUP -------------------
# ------------------- GOOGLE SHEETS SETUP -------------------
try:
    import gspread
    from google.oauth2.service_account import Credentials
    USE_SHEETS = True
except ImportError:
    USE_SHEETS = False
    st.warning("⚠️ gspread not installed. Using JSON fallback.")

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

        # ✅ Open spreadsheet by name and worksheet "Sheet3"
        spreadsheet = client.open(SHEET_NAME)
        sheet = spreadsheet.worksheet("Sheet1")

        # ✅ Ensure headers exist
        existing_headers = sheet.row_values(1) or []
        if existing_headers != HEADERS:
            if existing_headers:
                try:
                    sheet.delete_row(1)
                except Exception:
                    pass
            sheet.insert_row(HEADERS, 1)

        st.success("✅ Connected to Google Sheets → Sheet1")

    except gspread.WorksheetNotFound:
        st.error("🚨 Worksheet 'Sheet3' not found in your spreadsheet. Please create it manually.")
        USE_SHEETS = False
        sheet = None
    except gspread.SpreadsheetNotFound:
        st.error(f"🚨 Spreadsheet '{SHEET_NAME}' not found. Make sure it exists and is shared with the service account.")
        USE_SHEETS = False
        sheet = None
    except Exception as e:
        msg = str(e)
        if "storageQuotaExceeded" in msg:
            st.error("🚨 Service account’s Google Drive storage quota exceeded. Falling back to JSON.")
        else:
            st.error(f"⚠️ Google Sheets setup failed: {msg}")
        USE_SHEETS = False
        sheet = None

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import streamlit as st

def get_chat_model():
    """Use Gemini (Google API key), fallback to OpenAI (OpenAI API key)."""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-1.0-pro",   # ✅ Gemini model
            google_api_key=GOOGLE_API_KEY
        )
    except Exception as e1:
        st.warning(f"⚠️ Gemini failed: {e1}")
        if OPENAI_API_KEY:
            try:
                return ChatOpenAI(
                    model="gpt-4o-mini",  # ✅ OpenAI model
                    api_key=OPENAI_API_KEY
                )
            except Exception as e2:
                st.error(f"🚨 OpenAI fallback failed: {e2}")
                return None
        else:
            st.error("🚨 No OpenAI API key available for fallback.")
            return None


# ------------------- HELPERS -------------------

def find_files(path: Path):
    if path.is_file():
        return [path]
    exts = {".txt", ".md", ".pdf"}
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def load_documents(paths: List[Path]) -> List[Document]:
    docs: List[Document] = []
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def build_or_load_faiss(chunks, rebuild=True):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY)
    if rebuild:
        if not chunks:
            raise ValueError("No chunks found. Upload documents first.")
        vs = FAISS.from_documents(chunks, embeddings)
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(INDEX_PATH))
        return vs
    return FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

def save_query_answer(query, answer, status):
    data = []
    if QUERIES_FILE.exists():
        with open(QUERIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.append({"query": query, "answer": answer, "status": status, "timestamp": datetime.datetime.now().isoformat()})
    with open(QUERIES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_llm(model_name):
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=GOOGLE_API_KEY)

def append_ticket_to_sheet(ticket: Dict):
    """Append a ticket dictionary as a new row in Google Sheets."""
    if not USE_SHEETS:
        return
    global sheet
    if 'sheet' not in globals() or sheet is None:
        st.error("Google Sheets client not initialized.")
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
        st.success("✅ Ticket saved to Google Sheets.")
    except Exception as e:
        msg = e.args[0] if hasattr(e, "args") and e.args else str(e)
        st.error(f"Failed to append ticket to Google Sheets: {msg}")

# ------------------- RAG / CLASSIFICATION -------------------

CATEGORY_PROMPT = ChatPromptTemplate.from_template(
    """You are a support ticket classifier. Based only on the knowledge base provided:
    {context}

    Classify this ticket into the most relevant category:
    {input}

    Reply with only the category name."""
)

ASSISTANT_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful support assistant. Use only the following documents to answer: 
    {context}  
    User Question: {input}"""
)

def categorize_ticket(content, retriever):
    try:
        llm = load_llm(PRIMARY_CHAT_MODEL)
        doc_chain = create_stuff_documents_chain(llm, CATEGORY_PROMPT)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        result = rag_chain.invoke({"input": content})
        return result["answer"].strip()
    except ResourceExhausted:
        llm = load_llm(FALLBACK_CHAT_MODEL)
        doc_chain = create_stuff_documents_chain(llm, CATEGORY_PROMPT)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        result = rag_chain.invoke({"input": content})
        return result["answer"].strip()

def resolve_ticket(content, retriever, prompt):
    try:
        llm = load_llm(PRIMARY_CHAT_MODEL)
        doc_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        response = rag_chain.invoke({"input": content})
        return response["answer"]
    except ResourceExhausted:
        llm = load_llm(FALLBACK_CHAT_MODEL)
        doc_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        response = rag_chain.invoke({"input": content})
        return response["answer"]

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
    with open(PILOT_FILE, "w", encoding="utf-8") as f:
        json.dump({"timestamp": datetime.datetime.now().isoformat(), "metrics": metrics}, f, indent=4)
    return metrics


TICKETS_FILE = Path("ticket_raised.json")

def save_ticket(ticket: Dict):
    """Save a ticket or question to ticket_raised.json with all headers."""
    # Ensure consistent structure with headers
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
        with open(TICKETS_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    data.append(row)
    with open(TICKETS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# ------------------- STREAMLIT UI -------------------

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
                st.session_state.vectorstore = build_or_load_faiss(st.session_state.chunks, rebuild=True)
                st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr", search_kwargs={"k": st.session_state.top_k}
                )
                st.success("✅ Index rebuilt.")
        else:
            st.warning("Upload & split documents first.")

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
        st.sidebar.success("✅ Uploaded files saved to ./uploaded_docs")

def main_ui():
    st.set_page_config(page_title="Support Knowledge Assistant", layout="wide")
    st.title("🚀 Support Knowledge Assistant — Streamlit")

    init_state()
    sidebar_settings()
    sidebar_docs_uploader()

    col1, col2 = st.columns([2, 1])

    # -------- Left Panel --------
    with col1:
        st.header("📚 Index & Documents")
        docs_path = st.text_input("Local docs path", value=str(DEFAULT_DOCS_PATH))
        if st.button("Load & Split Documents"):
            paths = find_files(Path(docs_path))
            st.session_state.docs = load_documents(paths)
            st.session_state.chunks = split_documents(
                st.session_state.docs,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap
            )
            st.success(f"Loaded {len(st.session_state.docs)} docs → {len(st.session_state.chunks)} chunks.")

        if st.session_state.chunks:
            st.subheader("Sample chunk")
            st.write(st.session_state.chunks[0].page_content[:500])
            if st.button("Build FAISS index"):
                with st.spinner("Building index..."):
                    st.session_state.vectorstore = build_or_load_faiss(st.session_state.chunks, rebuild=True)
                    st.session_state.retriever = st.session_state.vectorstore.as_retriever(
                        search_type="mmr", search_kwargs={"k": st.session_state.top_k}
                    )
                    st.success("✅ Index built.")

        st.markdown("---")
        st.header("💬 Ask a Question")
        query = st.text_area("Enter your question", height=100)
        if st.button("Get Answer"):
            if not st.session_state.retriever:
                st.error("⚠️ Build index first.")
            else:
                with st.spinner("Querying LLM..."):
                    try:
                        llm = load_llm(st.session_state.primary_model)
                        doc_chain = create_stuff_documents_chain(llm, ASSISTANT_PROMPT)
                        rag_chain = create_retrieval_chain(st.session_state.retriever, doc_chain)
                        response = rag_chain.invoke({"input": query})
                        answer = response["answer"]
                    except ResourceExhausted:
                        llm = load_llm(st.session_state.fallback_model)
                        doc_chain = create_stuff_documents_chain(llm, ASSISTANT_PROMPT)
                        rag_chain = create_retrieval_chain(st.session_state.retriever, doc_chain)
                        response = rag_chain.invoke({"input": query})
                        answer = response["answer"]

                    st.markdown("**Answer:**")
                    st.write(answer)

                    resolved = st.radio("Did this answer resolve your issue?", ("Yes", "No"))
                    status = "Resolved" if resolved == "Yes" else "In Progress"
                    save_query_answer(query, answer, status)
                    st.success(f"Saved query (status: {status})")

                    question_ticket = {
                        "ticket_id": str(uuid.uuid4())[:8],
                        "ticket_content": query,
                        "ticket_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "ticket_by": "User",
                        "ticket_raised_by": "Question",
                        "ticket_category": "",
                        "ticket_problem": query,
                        "ticket_solution": answer,
                        "ticket_status": status,
                    }
                    save_ticket(question_ticket)
                    st.info("Question also saved to ticket_raised.json.")

    # -------- Right Panel --------
    with col2:
        st.header("🎫 Tickets")
        with st.form("ticket_form"):
            ticket_content = st.text_area("Ticket content")
            ticket_by = st.text_input("Ticket submitted by")
            ticket_raised_by = st.text_input("Who raised the ticket")
            ticket_problem = st.text_input("Describe the problem")
            submitted = st.form_submit_button("Create & Process Ticket")
            if submitted:
                if not st.session_state.retriever:
                    st.error("⚠️ Build index first.")
                else:
                    ticket = {
                        "ticket_id": str(uuid.uuid4())[:8],
                        "ticket_content": ticket_content,
                        "ticket_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "ticket_by": ticket_by,
                        "ticket_raised_by": ticket_raised_by,
                        "ticket_category": "",
                        "ticket_problem": ticket_problem,
                        "ticket_solution": "",
                        "ticket_status": "Open"
                    }
                    ticket["ticket_category"] = categorize_ticket(ticket_content, st.session_state.retriever)
                    ticket["ticket_solution"] = resolve_ticket(ticket_content, st.session_state.retriever, ASSISTANT_PROMPT)

                    st.session_state.queries.append(ticket)

                    st.success(f"✅ Ticket {ticket['ticket_id']} created → Category: {ticket['ticket_category']}")
                    st.markdown("**Proposed Solution:**")
                    st.write(ticket["ticket_solution"])

                    resolved = st.radio(
                        f"Did this solution resolve Ticket {ticket['ticket_id']}?",
                        ("Yes", "No"),
                        key=f"res_{ticket['ticket_id']}"
                    )
                    ticket["ticket_status"] = "Resolved" if resolved == "Yes" else "In Progress"

                    save_ticket(ticket)
                    if USE_SHEETS:
                        append_ticket_to_sheet(ticket)
                        st.info("Ticket also saved to Google Sheets.")

        # -------- Past / Recent Tickets Section --------
        st.markdown("---")
        st.subheader("Recent / Past Tickets")
        if st.session_state.queries:
            for t in reversed(st.session_state.queries[-20:]):
                with st.expander(f"{t['ticket_id']} — {t['ticket_category']} ({t['ticket_status']})"):
                    new_content = st.text_area("Ticket content", t["ticket_content"], key=f"{t['ticket_id']}_content")
                    new_problem = st.text_input("Ticket problem description", t["ticket_problem"], key=f"{t['ticket_id']}_problem")
                    status_options = ["Open", "In Progress", "Resolved", "Closed"]
                    new_status = st.selectbox(
                        "Update status",
                        status_options,
                        index=status_options.index(t["ticket_status"]),
                        key=f"{t['ticket_id']}_status"
                    )

                    if st.button("Save Changes", key=f"{t['ticket_id']}_save"):
                        t["ticket_content"] = new_content
                        t["ticket_problem"] = new_problem
                        t["ticket_status"] = new_status

                        if st.session_state.retriever:
                            t["ticket_solution"] = resolve_ticket(
                                t["ticket_content"], st.session_state.retriever, ASSISTANT_PROMPT
                            )

                        save_ticket(t)
                        if USE_SHEETS:
                            append_ticket_to_sheet(t)
                        st.success(f"✅ Ticket {t['ticket_id']} updated → Status: {t['ticket_status']}")

        else:
            st.info("⚠️ No tickets found yet.")

        # -------- Dataset / Tagging Validation Section --------
        st.markdown("---")
        st.header("📊 Validate Tagging & Category Accuracy")

        validation_mode = st.radio(
            "Validation Mode:",
            ["By File Upload (Pilot Dataset)", "By Last Ticket Text", "By User Input Text"]
        )

        if validation_mode == "By File Upload (Pilot Dataset)":
            file_uploader = st.file_uploader("Upload Pilot Dataset (CSV or JSON)", type=["csv", "json"])
            if st.button("Run Pilot Validation (File)"):
                if not st.session_state.retriever:
                    st.error("⚠️ Build index first.")
                else:
                    if not file_uploader:
                        st.error("⚠️ Upload a pilot dataset file first.")
                        return
                    import pandas as pd
                    try:
                        if file_uploader.type == "application/json" or file_uploader.name.endswith('.json'):
                            samples = json.load(file_uploader)
                        else:
                            df = pd.read_csv(file_uploader)
                            samples = df.to_dict(orient="records")

                        metrics = run_pilot_validation(samples, st.session_state.retriever)
                        outfile = "pilot_validation_results.json"
                        with open(outfile, "w", encoding="utf-8") as f:
                            json.dump({"timestamp": datetime.datetime.now().isoformat(),
                                       "metrics": metrics}, f, indent=4)

                        st.json(metrics)
                        st.success(f"✅ Pilot dataset validation complete — results saved to {outfile}")
                    except Exception as e:
                        st.error(f"Pilot validation failed: {e}")

        elif validation_mode == "By Last Ticket Text":
            if st.session_state.queries:
                last_ticket = st.session_state.queries[-1]
                st.info(f"Validating last ticket: {last_ticket['ticket_content']}")
                if st.button("Run Pilot Validation (Ticket Text)"):
                    try:
                        samples = [{"content": last_ticket["ticket_content"], "label": last_ticket["ticket_category"]}]
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
                if not st.session_state.retriever:
                    st.error("⚠️ Build index first.")
                elif not user_text.strip():
                    st.error("⚠️ Enter some text first.")
                else:
                    try:
                        # Create a single-sample dataset
                        samples = [{"content": user_text, "label": categorize_ticket(user_text, st.session_state.retriever)}]
                        metrics = run_pilot_validation(samples, st.session_state.retriever)
                        st.json(metrics)
                        st.success("✅ Validation complete for user input text")
                    except Exception as e:
                        st.error(f"User input validation failed: {e}")

    st.markdown("---")
    st.caption("⚡ Validate ticket tagging quality and category accuracy using pilot datasets, last ticket, or custom user input.")

if __name__ == "__main__":
    main_ui()
