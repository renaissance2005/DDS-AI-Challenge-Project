
import os
import redis
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from chromadb import ChromaDB

# Function to load or create vector store
def load_or_create_vector_store():
    vector_store_path = './chroma_db_store'
    if os.path.exists(vector_store_path):
        vector_store = ChromaDB.load(vector_store_path)
        print("Loaded existing ChromaDB vector store.")
    else:
        vector_store = ChromaDB()
        vector_store.save(vector_store_path)
        print("Created and persisted new ChromaDB vector store.")
    return vector_store

# Function to connect to Redis
def connect_to_redis():
    r = redis.Redis(host='localhost', port=6379, db=0)
    if r.dbsize() > 0:
        print("Connected to Redis and found existing document store.")
    else:
        print("Connected to Redis but no existing documents found, initializing new document store.")
    return r

# Function to store PDF content in ChromaDB
def store_pdf_in_vector_store(data, vector_store):
    for chunk in data:
        vector_store.add({'content': chunk.page_content})
    print("Stored PDF content in the vector store.")

# Function to store PDF content in Redis
def store_pdf_in_redis(data, redis_store):
    for i, chunk in enumerate(data):
        redis_store.set(f'pdf_chunk_{i}', chunk.page_content)
    print("Stored PDF content in the Redis document store.")

# Streamlit app to handle file upload and processing
def main():
    st.title("PDF Processing and Storage with ChromaDB and Redis")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Display the name of the file uploaded
        st.write(f"File uploaded: {uploaded_file.name}")

        # Initialize the PDF loader with specific parameters
        loader = UnstructuredPDFLoader(
            file_path=uploaded_file,
            strategy='hi_res',
            extract_images_in_pdf=True,
            infer_table_structure=True,
            chunking_strategy="by_title",
            max_characters=4000,
            new_after_n_chars=4000,
            combine_text_under_n_chars=2000,
            mode='elements',
            image_output_dir_path='./figures'
        )

        # Load the content from the PDF
        st.write("Extracting content from the PDF...")
        data = loader.load()
        st.write(f"Extracted {len(data)} chunks from the PDF.")

        # Load or create vector store and Redis
        vector_store = load_or_create_vector_store()
        redis_store = connect_to_redis()

        # Process and store the data
        if st.button("Process and Store PDF Content"):
            store_pdf_in_vector_store(data, vector_store)
            store_pdf_in_redis(data, redis_store)
            st.write("PDF content processed and stored in ChromaDB and Redis.")

if __name__ == "__main__":
    main()
