import os
import streamlit as st
from rag_utility import process_document_to_chroma_db, answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("🦙 Llama-3.3-70B - Multi-PDF RAG")

# ZMIANA: accept_multiple_files=True pozwala wrzucić kilka PDFów naraz
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

# Sprawdzamy czy lista nie jest pusta
if uploaded_files:
    file_names = []
    
    # ZMIANA: Pętla zapisująca wszystkie pliki na dysk
    for uploaded_file in uploaded_files:
        save_path = os.path.join(working_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_names.append(uploaded_file.name)

    # Przekazujemy listę plików do funkcji w rag_utility
    if st.button("Process Documents"): # Dodałem przycisk, żeby nie mieliło przy każdym odświeżeniu
        with st.spinner("Processing documents..."):
            process_document_to_chroma_db(file_names)
        st.info("Documents Processed Successfully!")

user_question = st.text_area("Ask your question about the documents")

if st.button("Answer"):
    # Pobieramy pełną odpowiedź (tekst + źródła)
    response = answer_question(user_question)
    
    st.markdown("### Llama-3.3-70B Response")
    st.markdown(response["result"]) # Wyświetlamy samą odpowiedź
    
    # ZMIANA: Logika wyciągania i wyświetlania źródeł
    st.markdown("### Source Documents")
    
    # Używamy set(), żeby uniknąć duplikatów (np. jeśli 3 fragmenty są z tego samego pliku)
    sources = set()
    for doc in response["source_documents"]:
        # Wyciągamy samą nazwę pliku ze ścieżki (metadata['source'])
        file_name = os.path.basename(doc.metadata['source'])
        sources.add(file_name)
    
    # Wyświetlamy ładną listę
    for source in sources:
        st.caption(f"📄 {source}")