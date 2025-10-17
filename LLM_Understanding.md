
LLM Understanding

AI -> ML -> DL

ML -> SUPERVISED LEARNING
      UNSUPERVISED LEARNING
      REINFORCEMENT LEARNING


[ 1 2 3 4 5 
  5 6 7 8 9
  2 3 4 5 6
  2 3 4 5 6
            ]

[ what is an apple ] -> [0.45,0.43,4504,4354 ] -> [ cosine similarity (A.B/|A||B|)] -> [0.443,45,35,4543,556 ] -> DECODER -> [ APPLE IS A FRUIT ]

Transformer ( Probability )

      .99/0.76/0.65....
Apple is/are/ave/had/has........a......... fruit/vegetable/coconut (0.10)


32 -> 8 bit
LORA : 8 bit
QLORA : 4 bit
1 Bit-net : 1 Bit ( Theorotical )

[ 1 2 3 4 5 
  5 6 7 8 9
  2 3 4 5 6
  2 3 4 5 6
            ]

Language Modeling -> Transformer

Language Modeling : Probability distribution over a sequence of token/words p(x.....Xn)

p(the, mouse, ate, the, cheese) = 0.02
p(the, the, mouse, ate, cheese) = 0.0001 syntactical knowledge 
p(the,cheese, ate, the, cheese) = 0.001 semantic knowledge
p(the, mouse, ate, the, cheese) = 0.989 (syntactical knowledge + semantic knowledge)


* LMs are generative models : X1:n ~ p(x1....xn)

Autoregressive (AR language models) :

P(x1.....xn) = P(x1)p(x1/x2)p(x3/x2x1)..... = pi (i)p(xi/x1:i-1)
No approx : chain rule of probability

=> You only need a model that can predict the next token given past token

=> AR language Models
* Task: Predict the next word/token
Steps: 1) Tokenize 2) Forwarding 3) Predict probablity of next token
       4) Sample 5) Detokenize
       4 and 5 needed during infrerencing only

Loss:
* Classify next token's index
  => Cross Entropy loss
Training example: I saw a cat on a mat <eos>
Loss = -log(p (cat)) -> min

Model Prediction : p( * / I saw a )
       ------
       0
       ------
       cat   1
       ------
       0
       ------
       0

=> maximize text's log-likelihood

max ∏(p(xi/xi-1)) = min ( - Σ logp(xi/X (i:/i-1) ) = minL(xi:1)

Tokenizer:
- More general than words ( typo )
- Shorter sequences that with character

LLM Evaluation :
Perplexity

* Validation loss []
* To be more interpretable : use perplexity

Data:
- Use all of the clean internet
- Download all the internet: 250 Billion pages, 1PB -> 1e6GB
- Text extration from HTML ( Challenges: Math, boiler plates )
- Filter Undesirable content ( NSFW, harmful content, PIL, hate speeches )
- Deduplication (URL/ document/ line ). eg -> (header/footer/menu)
- Heuristic filtering : RM low quality documents.
- Wikipedia : Model based filter is required
- Data Mix : (Classify data categories )

* Collection Data is a huge part of paraction LLM ( ~the key )

Coomon academic datasets:
* C4 ( a50 B tokens/ 800GB )
* The pile ( 280 tokens )
* Dolma ( 3 T tokens )
* Fineweb ( 15T tokens )

* SCaling Laws :
* Empirically : more data and large model ( Better Performance )
                  * (Large Models : Overfitting )
                  
------------------------------------------- Advance RAG -------------------------------------------

RAG: (Accuracy looses with basic RAG)
- Hybrid Search

( semantic search + syntatic search )

- Not just Vector Dense search

- Syntatic Searhc -> Exact Search -> Keyword Search or Sparse Vector

Sparse Matrix : DHE, BOIN, TF-IDF -> Text -> Sparse Matrix
0 0 0 0 1 0 0 0

         -> Sparse Matrix
Document                   [ vector store, keyword search ] -> ( resut from vector and result from keyword ) -> ( combined weight )
         -> Dense Vector

         -> ( Response from vector  from retriever ) -> ( LLM + RESP )


( User ask something : what is the discount my competitor is providing on product x ) ->(
      LLM layer ( Hey this my data set, this user's query expand it make it better for vector search )
)

for milestone 2 
Apply LLMs (e.g., GPT, LLaMA) for content categorization based on semantic meaning.
• Validate tagging quality and category accuracy through pilot datasets.  
 python -m streamlit run LLM_milestone2.py
myenv\Scripts\activate


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