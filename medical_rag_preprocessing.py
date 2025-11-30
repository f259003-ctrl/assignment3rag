# medical_rag_preprocessing.py
import pandas as pd
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.schema import Document
import os
from dotenv import load_dotenv

load_dotenv()

class MedicalRAGPreprocessor:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
    def load_dataset(self, file_path):
        """Load and preprocess medical transcriptions dataset"""
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(df)} records")
        return df
    
    def create_documents(self, df):
        """Create LangChain documents from dataframe"""
        documents = []
        for idx, row in df.iterrows():
            text = f"""
            Medical Specialty: {row['medical_specialty']}
            Transcription: {row['transcription']}
            Description: {row['description']}
            """
            metadata = {
                "medical_specialty": row['medical_specialty'],
                "description": row['description'],
                "source": "medical_transcriptions"
            }
            documents.append(Document(page_content=text, metadata=metadata))
        return documents
    
    def chunk_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into chunks"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)
    
    def create_vector_store(self, chunks, save_path="medical_faiss_index"):
        """Create and save FAISS vector store"""
        vector_store = FAISS.from_documents(chunks, self.embeddings)
        vector_store.save_local(save_path)
        return vector_store

# Usage
if __name__ == "__main__":
    preprocessor = MedicalRAGPreprocessor()
    df = preprocessor.load_dataset("mtsamples.csv")
    documents = preprocessor.create_documents(df)
    chunks = preprocessor.chunk_documents(documents)
    vector_store = preprocessor.create_vector_store(chunks)
