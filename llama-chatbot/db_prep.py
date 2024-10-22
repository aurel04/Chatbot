import os
from PyPDF2 import PdfReader
import docx2txt
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Path to your documents folder
current_dir = os.path.dirname(os.path.abspath(__file__))
documents_folder = os.path.join(current_dir, "documents")

# Collect all the documents you want to process
documents = []
for filename in os.listdir(documents_folder):
    if filename.endswith(".pdf"):
        with open(os.path.join(documents_folder, filename), "rb") as file:
            pdf_reader = PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            documents.append(text)
    elif filename.endswith(".docx"):
        text = docx2txt.process(os.path.join(documents_folder, filename))
        documents.append(text)
    elif filename.endswith(".txt"):
        with open(os.path.join(documents_folder, filename), "r", encoding="utf-8") as file:
            text = file.read()
            documents.append(text)

# Combine all text into a single string
all_text = "\n".join(documents)

# Split the text into smaller chunks
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
text_chunks = text_splitter.split_text(all_text)

# Create embeddings for each chunk
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store from the text chunks
vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

# Save the FAISS index and metadata to disk
save_directory = os.path.join(current_dir, "faiss_db")
os.makedirs(save_directory, exist_ok=True)
vector_store.save_local(save_directory)

print("FAISS vector store saved successfully!")
