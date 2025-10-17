import os
import json
from pathlib import Path
from dotenv import load_dotenv
import datetime

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from google.api_core.exceptions import ResourceExhausted

try:
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    USE_SHEETS = True
except ImportError:
    USE_SHEETS = False

load_dotenv()  # load .env file

DOCS_PATH = Path(r"C:\Users\Admin\Desktop\AI Internship Infosys\support_knowledge.pdf")
INDEX_PATH = Path("./faiss_index")
REBUILD_INDEX = True

# Google Generative AI Models
EMBED_MODEL = "models/embedding-001"
PRIMARY_CHAT_MODEL = "gemini-1.5-pro"
FALLBACK_CHAT_MODEL = "gemini-1.5-flash"

TOP_K = 4
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
QUERIES_FILE = Path("queries.json")  # JSON file for local storage

# Load API key from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not found! Please add it to your .env file.")

# Google Sheets config
SHEET_NAME = "Sheet1"               # <-- Update this with your sheet tab name
CREDENTIALS_FILE = "credentials.json"

if USE_SHEETS and Path(CREDENTIALS_FILE).exists():
    scope = ["https://spreadsheets.google.com/feeds",
             "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, scope)
    client = gspread.authorize(creds)
    sheet = client.open_by_key("13IfoSOZtEdqQ1mDhQrPHtEO06OFMOSs0M4TWy5PrG6U").worksheet(SHEET_NAME)
    print(f"✅ Connected to Google Sheet: {SHEET_NAME}")

# ------------------- FUNCTIONS -------------------

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

def save_query_answer(query, answer):
    # Save locally
    data = []
    if QUERIES_FILE.exists():
        with open(QUERIES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    data.append({"query": query, "answer": answer})
    with open(QUERIES_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("📝 Query saved to queries.json")

    # Save to Google Sheets
    if USE_SHEETS and 'sheet' in globals():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([timestamp, query, answer])
        print("✅ Query saved to Google Sheets!")

def load_llm(model_name):
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, google_api_key=GOOGLE_API_KEY)

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
        """
You are a helpful support assistant. Use only the following documents to answer:

{context}

User Question:
{input}
"""
    )

    llm = load_llm(PRIMARY_CHAT_MODEL)
    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    while True:
        query = input("\n💬 Enter your question (or type 'exit' to quit): ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye!")
            break

        try:
            response = rag_chain.invoke({"input": query})
            answer = response["answer"]
        except ResourceExhausted:
            print("⚠️ Quota exceeded on gemini-1.5-pro. Switching to gemini-1.5-flash...")
            llm = load_llm(FALLBACK_CHAT_MODEL)
            doc_chain = create_stuff_documents_chain(llm, prompt)
            rag_chain = create_retrieval_chain(retriever, doc_chain)
            response = rag_chain.invoke({"input": query})
            answer = response["answer"]
        except Exception as e:
            print(f"❌ Error during query: {e}")
            continue

        print("\n🤖 Answer:", answer)
        save_query_answer(query, answer)

if __name__ == "__main__":
    main()
