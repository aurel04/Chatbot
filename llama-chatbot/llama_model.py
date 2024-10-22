import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from PyPDF2 import PdfReader
import docx2txt
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import tempfile

load_dotenv()  
groq_api_key = os.environ['GROQ_API_KEY']

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello, Ask me anything!"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        user_input = st.chat_input("Ask me something....")

        if user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="avataaars", seed="Aneka")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts", seed="Aneka")

def create_conversational_chain(vector_store):
    # Create a prompt template
    prompt_template = ChatPromptTemplate.from_template("""
        Jawab pertanyaan dalam bahasa Indonesia.
        Perkenalkan dirimu sebagai bot pembantu dengan nama 'bobot'
        Pertanyaan: {question}
        Konteks: {context}
        Jawaban:
    """)

    # Create the LLM with the template
    llm = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name='llama3-70b-8192'
    )

    # Create the memory object
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    # Create the conversational retrieval chain with the prompt template
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type='stuff',
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": prompt_template,
            "document_variable_name": "context"  # Ensure the documents are passed as context
        }
    )
    return chain

def update_vector_store(new_texts, vector_store, embeddings):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    new_chunks = text_splitter.split_text(new_texts)
    vector_store.add_texts(new_chunks, embedding=embeddings)

def save_uploaded_file(uploaded_file):
    save_dir = "D:/Aurel/MAGANG/SPIL/CHATBOT/llama-chatbot/documents"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_saved_files():
    save_dir = "D:/Aurel/MAGANG/SPIL/CHATBOT/llama-chatbot/documents"
    if not os.path.exists(save_dir):
        return []
    return [os.path.join(save_dir, f) for f in os.listdir(save_dir)]

def main():
    load_dotenv()  
    groq_api_key = os.environ['GROQ_API_KEY']
    # Initialize session state
    initialize_session_state()
    st.set_page_config(page_title="Ask your Document")
    st.header("Document Chatbot")
    
    # Load pre-built FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                       model_kwargs={'device': 'cpu'})
    vector_store_path = "D:/Aurel/MAGANG/SPIL/CHATBOT/llama-chatbot/faiss_db"
    
    with st.spinner('Loading knowledge base...'):
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)

    # Initialize Streamlit
    st.sidebar.title("Upload Your Document")
    uploaded_files = st.sidebar.file_uploader("Accepted file format: .pdf, .docx, or .txt", type=["pdf", "txt", "docx"], accept_multiple_files=True)
    st.sidebar.write("Existing Files:")
    saved_files = load_saved_files()
    for save in saved_files:
        st.sidebar.write(save)

    if uploaded_files:
        # Extract text from uploaded files
        all_text = ""
        for uploaded_file in uploaded_files:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                save_uploaded_file(uploaded_file)
            elif uploaded_file.type == "text/plain":
                text = uploaded_file.read().decode("utf-8")
                save_uploaded_file(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                text = docx2txt.process(uploaded_file)
                save_uploaded_file(uploaded_file)
            else:
                st.write(f"Unsupported file type: {uploaded_file.name}")
                continue
            all_text += text
        if all_text:
            with st.spinner('Updating database...'):
                update_vector_store(all_text, vector_store, embeddings)
                vector_store.save_local(vector_store_path)
        
    if vector_store:
        # Create the chain object 
        chain = create_conversational_chain(vector_store)
        display_chat_history(chain)


if __name__ == "__main__":
    main()