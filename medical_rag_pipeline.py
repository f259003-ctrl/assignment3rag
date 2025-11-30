# medical_rag_pipeline.py
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import os

class MedicalRAGSystem:
    def __init__(self, vector_store_path="medical_faiss_index"):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        self.vector_store = FAISS.load_local(
            vector_store_path, 
            self.embeddings, 
            allow_dangerous_deserialization=True
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1
        )
        self.setup_qa_chain()
    
    def setup_qa_chain(self):
        """Setup the retrieval QA chain with medical-specific prompt"""
        prompt_template = """
        You are a medical assistant providing evidence-based information. 
        Use the following medical context to answer the question accurately and safely.
        
        Context: {context}
        
        Question: {question}
        
        Guidelines:
        - Provide accurate, evidence-based information
        - Include citations from the context when possible
        - Clearly state if information is limited
        - Emphasize that this is for informational purposes only
        - Recommend consulting healthcare professionals for medical advice
        
        Answer:
        """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def query(self, question):
        """Query the medical RAG system"""
        result = self.qa_chain({"query": question})
        return {
            "answer": result["result"],
            "source_documents": result["source_documents"]
        }
