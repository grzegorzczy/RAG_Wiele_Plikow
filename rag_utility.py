import os
from dotenv import load_dotenv

# ZMIANA: Używamy PyPDFLoader (jest stabilniejszy dla wielu plików)
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

# Modele
embedding = HuggingFaceEmbeddings()
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ZMIANA: Funkcja przyjmuje teraz LISTĘ plików (file_names), a nie jeden string
def process_document_to_chroma_db(file_names):
    all_documents = [] # Lista na treść ze wszystkich PDFów
    
    # Iterujemy przez każdy przesłany plik
    for file_name in file_names:
        file_path = f"{working_dir}/{file_name}"
        if os.path.exists(file_path):
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load()) # Dodajemy treść do wspólnej listy

    # Dzielimy tekst (chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(all_documents)
    
    # Tworzymy bazę wektorową dla wszystkich dokumentów naraz
    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=f"{working_dir}/doc_vectorstore"
    )
    return 0

def answer_question(user_question):
    vectordb = Chroma(
        persist_directory=f"{working_dir}/doc_vectorstore",
        embedding_function=embedding
    )
    
    retriever = vectordb.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True # ZMIANA: Ważne! Prosimy o zwrot źródeł
    )
    
    response = qa_chain.invoke({"query": user_question})
    
    # ZMIANA: Zwracamy cały obiekt response, żeby wyciągnąć źródła w app.py
    return response