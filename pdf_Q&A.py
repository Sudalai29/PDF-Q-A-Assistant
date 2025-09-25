import json
from typing import List, TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
import os, certifi, ssl
import gradio as gr


os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from openai import OpenAI

#--- Load and split PDF ---
print("[INFO] Loading and splitting PDF...")
loader = PyPDFLoader(r"Path to Your PDF file")
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = splitter.split_documents(pages)

# --- Embedding and vector store ---
print("[INFO] Creating embeddings and FAISS store...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embedding_model)

# --- LLM setup ---
llm = ChatOpenAI(
    model= "mistralai/mistral-7b-instruct:free", 
    temperature=0,
    openai_api_key="****",  # Replace with your key
    openai_api_base="https://openrouter.ai/api/v1"
)

summary_llm = ChatOpenAI(
    model="mistralai/mistral-7b-instruct:free",
    temperature=0.4,
    openai_api_key="****", # Replace with your key
    openai_api_base="https://openrouter.ai/api/v1"
)

# --- Prompt templates ---
qa_prompt = PromptTemplate.from_template(
    """You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {question}
Answer:"""
)

summary_prompt = PromptTemplate.from_template(
    """Summarize the following answer within 50 words, focusing on key points.:

Answer:
{answer}

Summary:"""
)

# --- Chains ---
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

def summarize_answer(answer_text: str) -> str:
    return summary_llm.invoke(summary_prompt.format(answer=answer_text))

# -----------------------------
# LangGraph Implementation
# -----------------------------

# --- State definition ---
class GraphState(TypedDict):
    question: str
    docs: List[Document]
    answer: str
    summary: str

# --- Step 1: Retrieve relevant chunks ---
def retrieve_docs(state: GraphState) -> GraphState:
    print("[LangGraph] Retrieving documents...")
    docs = vectorstore.similarity_search(state["question"], k=4)
    return {**state, "docs": docs}

# --- Step 2: Answer generation ---
def generate_answer(state: GraphState) -> GraphState:
    print("[LangGraph] Generating answer...")
    answer = qa_chain.invoke({"question": state["question"], "context": state["docs"]})
    return {**state, "answer": answer}

# --- Step 3: Summarization ---
def summarize_output(state: GraphState) -> GraphState:
    print("[LangGraph] Summarizing answer...")

    response = summary_llm.invoke(summary_prompt.format(answer=state["answer"]))

    # Force extract string from response
    summary_text = getattr(response, "content", str(response))

    # Optionally trim whitespace
    summary_text = summary_text.strip()

    return {**state, "summary": summary_text}


# --- Build the LangGraph ---
graph = StateGraph(GraphState)
graph.add_node("retrieve", retrieve_docs)
graph.add_node("qa", generate_answer)
graph.add_node("summarize", summarize_output)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "qa")
graph.add_edge("qa", "summarize")
graph.set_finish_point("summarize")

app = graph.compile()

# # --- Interactive CLI ---
# while True:
#     query = input("\nAsk a question about the PDF (or type 'exit'): ")
#     if query.lower() in ["exit", "quit"]:
#         print("[INFO] Exiting...")
#         break

#     result = app.invoke({"question": query})
#     print("\n[Agent 1 Answer]:", result["answer"])
#     print("\n[Agent 2 Summary]:", result["summary"])


# --- Keep everything in your script as is (up to app = graph.compile()) ---

# --- Gradio UI wrapper ---
def qa_pipeline(question: str):
    if not question.strip():
        return "⚠️ Please enter a question.", ""
    result = app.invoke({"question": question})
    return result["answer"], result["summary"]

demo = gr.Interface(
    fn=qa_pipeline,
    inputs=gr.Textbox(label="Ask a Question about the PDF", placeholder="Type your question here..."),
    outputs=[
        gr.Textbox(label="Detailed Answer", lines=10),
        gr.Textbox(label="Summary", lines=5)
    ],
    title="📘 PDF Q&A Assistant",
    description="Ask questions from the uploaded PDF and get AI-generated answers + summaries."
)

if __name__ == "__main__":
    demo.launch()
