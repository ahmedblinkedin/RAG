import gradio as gr
import ollama
from bs4 import BeautifulSoup as bs
from gradio_pdf import PDF
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from pathlib import Path

# Load the data from the pdf

loader = PyPDFLoader("test/pdf/Master Data Management (MDM).pdf")



# Split the loaded documents into chunks
pages = loader.load_and_split()

# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=pages, embedding=embeddings)

# Define the function to call the Ollama Llama3 model
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model='llama3', messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG setup
retriever = vectorstore.as_retriever()

def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return ollama_llm(question, formatted_context)

# Define the Gradio interface
def get_important_facts(question,doc):
    return rag_chain(question)


dir_ = Path(__file__)
print(dir_)
# Create a Gradio app interface
demo = gr.Interface(
    get_important_facts,
    [gr.Textbox(label="Question"), PDF(label="Document")],
    gr.Textbox(),
    examples=[["Un des article que j'ai publié sur Linkedin mis sous format pdf", str("test/pdf/Master Data Management (MDM).pdf")]]
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch(share=True)