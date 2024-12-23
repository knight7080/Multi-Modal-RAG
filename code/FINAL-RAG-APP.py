import streamlit as st  # UI Appliction
import os
import ollama # Llama
import chromadb
from extractous import Extractor, TesseractOcrConfig
from langchain.text_splitter import CharacterTextSplitter
from streamlit_pdf_viewer import pdf_viewer
upload_dir = "./my-docs"

st.set_page_config(layout="wide")
st.title("MultiModal RAG App")
# Embedding storage
chromadb.api.client.SharedSystemClient.clear_system_cache()
client = chromadb.Client()
collection = client.get_or_create_collection(name="new")

#UPLOAD FILE, SAVE FILE AND TEXT FILE
with st.sidebar:
    st.header("Configuration Options")

    # Dropdown for selecting Multi-Model LLM
    multi_model_llm = st.selectbox(
        "Select Multi-Model LLM",
        options=["tinyllama","llama3.2","llava:latest"]
    )
    
    # File upload button
    uploaded_file = st.file_uploader("Choose a Document", type=["pdf","ppt"])
    

# GET TEXT -> SPLIT -> ADD TO CHROMA DB
if uploaded_file is not None:
    col1, col2 = st.columns([1, 2])

    with col1:
        st.write("### Uploaded Document")
        save_path = os.path.join(upload_dir, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        pdf_viewer(input=uploaded_file.getvalue(),width=700)
        # Extract text from OCR for multimodality
        extractor = Extractor()
        extractor.set_extract_string_max_length(1000)
        extractor = Extractor().set_ocr_config(TesseractOcrConfig().set_language("eng"))
        result, metadata = extractor.extract_file_to_string(save_path)
        with open(os.path.join(upload_dir,uploaded_file.name+".txt"),"w",encoding='utf-8') as f:
            f.write(result)
        document_content = str(metadata) + "\n" + result
                # CREATE EMBEDDINGS OF THE DOCUMENT
        text_splitter = CharacterTextSplitter(
            chunk_size=1800,
            chunk_overlap=500,
            is_separator_regex=False,
        )
        texts = text_splitter.create_documents([result])
        # store each document in a vector embedding database
        for i, d in enumerate(texts):
            response = ollama.embeddings(model="nomic-embed-text", prompt=d.page_content)
            embedding = response["embedding"]
            # add collection to chroma db
            collection.add(
                ids=[str(i)],
                embeddings=[embedding],
                documents=[d.page_content]
            )
    with col2:
        st.text("PROMPT YOUR DOCUMENT")
        prompt = st.text_area("Enter your text query")

        if st.button("Search and Extract Text"):
            if prompt:
                response = ollama.embeddings(
                prompt=prompt,
                model="nomic-embed-text"
                )
                results = collection.query(
                query_embeddings=[response["embedding"]],
                n_results=10
                )
                data = results['documents'][0][0]
                
                # GENERATION
                # generate a response combining the prompt and data we retrieved in step 2
                output = ollama.generate(
                model="tinyllama",
                system=f"You are a dedicated and highly knowledgeable Study Tutor committed to providing accurate, reliable, and data-driven assistance to students. Your primary objective is to respond to questions and prompts using only the information explicitly contained in the dataset provided. You must avoid any form of speculation, assumption, or fabrication. If the dataset does not include the necessary details to address a prompt, you must inform the user clearly and politely, requesting them to provide additional context, specificity, or clarification to refine their request. When answering, ensure your responses are thorough, well-structured, and focused on the needs of the user while strictly adhering to the scope of the provided data. Here is the dataset you will be working with: {data}.",
                prompt=f"{data}. Based on this dataset, Answer for prompt: {prompt}."
                )

                st.text(output["response"])

                st.text("Source:")
                st.text(data)


# GET PROMPT AND GENERATE RESPONSE

# Compare hardware specification of TESLA  M1060 and Tesla C2050
# GPU specification of Tesla GPU's
