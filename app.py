import streamlit as st
import glob
import os
from backend import preprocess_apple_data, build_faiss_index, retrieve_documents_faiss, retrieve_bm25, \
                   guardrail_input, guardrail_output, generate_response

def get_all_pdf_files(folder_path):
    return glob.glob(os.path.join(folder_path, "*.pdf"))

# ------------------------ APP START ------------------------

def initialize_app():
    if "initialized" not in st.session_state:
        folder_path = r"Data"
        file_paths = get_all_pdf_files(folder_path)
        documents = preprocess_apple_data(file_paths)
        faiss_index, doc_texts = build_faiss_index(documents)
        st.session_state.documents = documents
        st.session_state.faiss_index = faiss_index
        st.session_state.doc_texts = doc_texts
        st.session_state.conversation_history = []
        st.session_state.initialized = True
        st.success("Preprocessing and FAISS indexing complete!")
    else:
        st.info("Using cached FAISS index and documents")

def handle_query(user_query):
    is_valid, msg = guardrail_input(user_query)
    if not is_valid:
        st.error(msg)
        return
    retrieved_docs_with_scores = retrieve_documents_faiss(
        user_query, 
        st.session_state.faiss_index, 
        st.session_state.doc_texts, 
        top_k=3
    )
    if not retrieved_docs_with_scores:
        st.warning("No documents found with FAISS. Trying BM25 fallback...")
        retrieved_docs_with_scores = retrieve_bm25(user_query, st.session_state.documents, top_k=3)
    retrieved_texts = [doc for doc, _ in retrieved_docs_with_scores]
    response_text = generate_response(user_query, retrieved_texts, st.session_state.conversation_history)
    response_text = guardrail_output(response_text)

    st.session_state.conversation_history.append({
        "user": user_query,
        "retrieved_docs": retrieved_docs_with_scores,
        "assistant": response_text
    })

def display_history():
    st.subheader("Conversation History")
    for turn in reversed(st.session_state.conversation_history):
        st.write(f"**User:** {turn['user']}")        
        for i, (doc, score) in enumerate(turn['retrieved_docs']):
            st.write(f"**Doc {i+1} Confidence Score:** {score:.4f}")
        st.write(f"**Assistant:** {turn['assistant']}")


def main():
    st.title("Apple Inc. Financial RAG Chatbot with FAISS")

    initialize_app()

    user_query = st.text_input("Enter your question about Apple's financials")
    if st.button("Submit Query"):
        handle_query(user_query)

    display_history()

    st.subheader("Testing Suggestions")
    st.write("- 'What is Apple’s revenue for fiscal year 2022?'")
    st.write("- 'What are the major risk factors mentioned?'")
    st.write("- 'Who are Apple’s executive officers?'")

if __name__ == "__main__":
    main()
