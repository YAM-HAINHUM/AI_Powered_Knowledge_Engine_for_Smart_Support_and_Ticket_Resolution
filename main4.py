import os
import json
from pathlib import Path
from dotenv import load_dotenv
import datetime
import uuid

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
load_dotenv()

DOCS_PATH = Path(r"C:\Users\Admin\Desktop\AI Internship Infosys\support_knowledge.pdf")
INDEX_PATH = Path("./faiss_index")
REBUILD_INDEX = True

EMBED_MODEL = "models/embedding-001"
PRIMARY_CHAT_MODEL = "gemini-1.5-pro"
FALLBACK_CHAT_MODEL = "gemini-1.5-flash"

TOP_K = 4
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
QUERIES_FILE = Path("queries.json")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found! Please add it to your .env file.")

SHEET_NAME = "Sheet1"
CREDENTIALS_FILE = "credentials.json"

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

if USE_SHEETS and Path(CREDENTIALS_FILE).exists():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key("13IfoSOZtEdqQ1mDhQrPHtEO06OFMOSs0M4TWy5PrG6U").worksheet(SHEET_NAME)
    print(f"✅ Connected to Google Sheet: {SHEET_NAME}")

    first_row = sheet.row_values(1)
    if first_row != HEADERS:
        sheet.clear()
        sheet.insert_row(HEADERS, 1)

# ------------------- HELPERS -------------------
def find_files(path: Path):
    if path.is_file():
        return [path]
    exts = {".txt", ".md", ".pdf"}
    return [p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts]

def load_documents(paths: list[Path]) -> list[Document]:
    docs = []
    for p in paths:
        if p.suffix.lower() in {".txt", ".md"}:
            docs.extend(TextLoader(str(p), encoding="utf-8").load())
        elif p.suffix.lower() == ".pdf":
            docs.extend(PyPDFLoader(str(p)).load())
    return docs

def split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def build_or_load_faiss(chunks, rebuild=True):
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=GOOGLE_API_KEY)
    if rebuild:
        if not chunks:
            raise ValueError("⚠️ No chunks found. Please check your document path.")
        vs = FAISS.from_documents(chunks, embeddings)
        INDEX_PATH.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(INDEX_PATH))
        print("✅ FAISS index built and saved.")
        return vs
    vs = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)
    print("✅ FAISS index loaded.")
    return vs

def save_query_answer(query, answer, status):
    data = []
    if QUERIES_FILE.exists():
        with open(QUERIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.append({"query": query, "answer": answer, "status": status})
    with open(QUERIES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("📝 Query saved to queries.json")

    if USE_SHEETS and 'sheet' in globals():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row(["", query, timestamp, "", "", "", "", answer, status])
        print("✅ Query saved to Google Sheets!")

def load_llm(model_name):
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=GOOGLE_API_KEY)

# ------------------- TICKET HANDLING -------------------
def categorize_ticket(content, retriever):
    category_prompt = ChatPromptTemplate.from_template(
        """You are a support ticket classifier. Based only on the knowledge base provided:
        {context}

        Classify this ticket into the most relevant category:
        {input}

        Reply with only the category name."""
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

def resolve_ticket(content, retriever, prompt):
    try:
        llm = load_llm(PRIMARY_CHAT_MODEL)
        doc_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        response = rag_chain.invoke({"input": content})
        return response["answer"]
    except ResourceExhausted:
        print("⚠️ Quota exceeded. Falling back to flash...")
        llm = load_llm(FALLBACK_CHAT_MODEL)
        doc_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)
        response = rag_chain.invoke({"input": content})
        return response["answer"]

def process_ticket(ticket, retriever, prompt):
    ticket['ticket_category'] = categorize_ticket(ticket['ticket_content'], retriever)
    ticket['ticket_solution'] = resolve_ticket(ticket['ticket_content'], retriever, prompt)

    print("\n🤖 Suggested Solution:", ticket['ticket_solution'])
    resolved = input("✅ Was your issue resolved? (yes/no): ")
    ticket['ticket_status'] = "Resolved" if resolved.lower() == "yes" else "In Progress"

    if USE_SHEETS and 'sheet' in globals():
        row_data = [
            ticket.get("ticket_id"),
            ticket.get("ticket_content"),
            ticket.get("ticket_timestamp"),
            ticket.get("ticket_by"),
            ticket.get("ticket_raised_by"),
            ticket.get("ticket_category"),
            ticket.get("ticket_problem"),
            ticket.get("ticket_solution"),
            ticket.get("ticket_status")
        ]
        sheet.append_row(row_data)
        print(f"✅ Ticket {ticket['ticket_id']} saved with status {ticket['ticket_status']}.")

    return ticket

def update_ticket_status(ticket_id, new_status):
    if USE_SHEETS and 'sheet' in globals():
        cell = sheet.find(ticket_id)
        if cell:
            row = cell.row
            sheet.update_cell(row, HEADERS.index("Ticket Status") + 1, new_status)
            print(f"🔄 Ticket {ticket_id} status updated to {new_status}")
        else:
            print(f"⚠️ Ticket ID {ticket_id} not found.")

# ------------------- MAIN -------------------
def main():
    paths = find_files(DOCS_PATH)
    docs = load_documents(paths)
    print(f"📄 Loaded {len(docs)} documents")
    if not docs:
        print("⚠️ No documents found! Exiting.")
        return

    chunks = split_documents(docs)
    print(f"✂️ Split into {len(chunks)} chunks")

    vectorstore = build_or_load_faiss(chunks, REBUILD_INDEX)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K})

    prompt = ChatPromptTemplate.from_template(
        """You are a helpful support assistant. Use only the following documents to answer: 
        {context}  
        User Question: {input}"""
    )

    while True:
        user_input = input("\n💬 Type 'ticket' to create, 'update' to update status, or ask a question (or type 'exit'): ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        if user_input.lower() == "ticket":
            ticket = {
                "ticket_id": str(uuid.uuid4())[:8],
                "ticket_content": input("Enter Ticket Content: "),
                "ticket_timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticket_by": input("Ticket submitted by (email/web/etc): "),
                "ticket_raised_by": input("Who raised the ticket: "),
                "ticket_category": "",
                "ticket_problem": input("Describe the problem: "),
                "ticket_solution": "",
                "ticket_status": "Open"
            }
            process_ticket(ticket, retriever, prompt)

        elif user_input.lower() == "update":
            ticket_id = input("Enter Ticket ID to update: ")
            new_status = input("Enter new status (Open/In Progress/Closed/Resolved): ")
            update_ticket_status(ticket_id, new_status)

        else:
            query = user_input
            try:
                llm = load_llm(PRIMARY_CHAT_MODEL)
                doc_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, doc_chain)
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
            except ResourceExhausted:
                print("⚠️ Quota exceeded, switching model...")
                llm = load_llm(FALLBACK_CHAT_MODEL)
                doc_chain = create_stuff_documents_chain(llm, prompt)
                rag_chain = create_retrieval_chain(retriever, doc_chain)
                response = rag_chain.invoke({"input": query})
                answer = response["answer"]
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

            print("\n🤖 Answer:", answer)
            resolved = input("✅ Did this answer resolve your issue? (yes/no): ")
            status = "Resolved" if resolved.lower() == "yes" else "In Progress"
            save_query_answer(query, answer, status)
            print(f"📌 Status set to {status}")

if __name__ == "__main__":
    main()
