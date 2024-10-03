
# 📄 Technical Specification Fulfillment Application

This repository contains a Python-based application that processes both Excel and PDF files. It extracts content from the PDF file, stores it in a vector store (ChromaDB) and document store (Redis), and provides a Streamlit user interface for uploading files and processing them.

## 🛠 Features
- **PDF Content Extraction**: Extracts structured content from PDFs using the `UnstructuredPDFLoader`.
- **Excel Integration**: Processes Excel files containing a column named 'Minimum Specification' to extract queries for further processing.
- **Vector Store (ChromaDB)**: Stores the content into a vector store for efficient retrieval and embedding.
- **Document Store (Redis)**: Stores the content into a Redis document store for flexible and fast retrieval.
- **Streamlit UI**: Provides a user-friendly interface to upload both Excel and PDF files, and process the content.
- **Persistence**: Utilizes persisted storage for both ChromaDB and Redis, allowing reuse across multiple sessions without needing to reprocess the data.

## 📂 Project Structure

The project consists of four main Python files:

1. **`main.py`**:
   - The entry point for the application, providing the Streamlit interface.
   - Allows users to upload both an Excel file (with 'Minimum Specification') and a PDF file.
   - Extracts PDF content, processes the data, and stores it in ChromaDB and Redis.
   - Reuses persisted ChromaDB and Redis stores if they exist.

2. **`image_processing.py`**:
   - Handles encoding and summarizing of images (if present in the PDF).
   - Generates image summaries for document retrieval.

3. **`embeddings.py`**:
   - Handles embedding generation using the `OllamaEmbeddings` model for any required text.

4. **`evaluation.py`**:
   - Provides evaluation logic to compare generated responses to expected responses.
   - Rates responses for accuracy and completeness.

## 🚀 How to Run the Application

1. **Clone the repository**:
   ```bash
   git clone https://github.com/renaissance2005/DDS-AI-Challenge-Project.git
   cd DDS-AI-Challenge-Project
   ```

2. **Install dependencies**:
   Ensure you have Python 3.10+, Ollama and Redis Server and install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application**:
   Start the Streamlit app using:
   ```bash
   streamlit run main.py
   ```

4. **Upload Excel and PDF Files**:
   - Open the app in your browser.
   - Upload an Excel file containing a 'Minimum Specification' column.
   - Upload a PDF file for processing.
   - Click the button to process and store the content in ChromaDB and Redis.

5. **Download the processed results**:
   After processing, the content is stored in ChromaDB and Redis, ready for retrieval in future sessions.

## 🧑‍💻 Tech Stack

- **Python**: Core programming language.
- **ChromaDB**: Used as a vector store for efficient document embedding and retrieval.
- **Redis**: Used as a document store to store the PDF content.
- **Streamlit**: Provides an interactive UI for file uploads and processing.
- **LangChain Community Tools**: Used for PDF extraction and unstructured data processing.

## 🛡️ Error Handling

- The program checks that both an Excel and a PDF file are uploaded before processing.
- The Excel file must contain a 'Minimum Specification' column.
- Error messages are displayed if any issues arise during file upload or processing.

## 🌟 Future Improvements

- Add agents to check and refine the results.
- Incorporate options to use more powerful multimodal language models.
- Extend the use of embeddings for more advanced document search capabilities.

## 📬 Contact

Feel free to reach out if you have any questions or suggestions:
- **Email**: davidkeat@graduate.utm.my
- **GitHub**: [yourusername](https://github.com/renaissance2005)

