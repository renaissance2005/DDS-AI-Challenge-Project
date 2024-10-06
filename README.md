
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

The project consists of the following files. For user to use the program, only the files in item 3 are required:

1. **`populatedata.ipynb`**:
   - This is run on Jupyter Notebook for testing and debugging during development.
   - Loaded the brochure HP-dataset.pdf as example document which should be in the same folder of this file. 
   - Extracts PDF content, processes the data, and stores it in ChromaDB and Redis.
   - Data extracted include text, table and image (multimodal).
   - This file can be run in Jupyter notebook.

2. **`tsfa.py`**:
   - The evaluation of the program is done using this code by comparing the ‘Expected Response’ and ‘Generated Response’ columns from an excel file with relevance given a score from 1-3.
   - The input file Sample-TenderDoc.xlsx should be in the same folder of this file. 
   - Take note that the processing and storage of brochure/datasheet (company documents) are not done here as it was implemented by the by populatedata.ipynb.
   - This file can be run using python command. 

3. **`chktechspec.py & process.py`**:
   - This is the actual program where the user can run by giving an excel file with at least a column named ‘Minimum Specification’.
   - User can upload any arbitrary excel file as long as it contains a column named 'Minimum Specification'. The file will generate the same file with a generated column called 'Generated Specification'. 
   - A new file will be generated with the column ‘Generated Response’ which the user can use as reference for specification fulfillment of the existing product brochure/datasheet.
   - A streamlit command is used to call the file chktechspec.py (streamlit run chktechspec.py).
   - User can download the processed file in the standard name of 'Updated-Sample-TenderDoc.xlsx'.

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
   streamlit run chktechspec.py
   ```

4. **Upload an Excel File**:
   - Open the app in your browser.
   - Upload an Excel file containing a 'Minimum Specification' column.
   - Click the download button after the file is processed. 

## 🧑‍💻 Tech Stack

- **Python**: Core programming language.
- **ChromaDB**: Used as a vector store for efficient document embedding and retrieval.
- **Redis**: Used as a document store to store the PDF content.
- **Ollama**: Installed and run 2 models: llama3.1:8b (inference) and llava-llama3 (vision).
- **Streamlit**: Provides an interactive UI for file uploads and processing.
- **LangChain Community Tools**: Used for PDF extraction and unstructured data processing.

## 🛡️ Error Handling

- The program checks that both an Excel is uploaded before processing.
- The Excel file must contain a 'Minimum Specification' column.
- Error messages are displayed if any issues arise during file upload or processing.

## 🌟 Future Improvements

- Add agents to check and refine the results.
- Incorporate options to use more powerful multimodal language models.
- Provide functionality to upload and process multiple brochures and datasheets.

## 📬 Contact

Feel free to reach out if you have any questions or suggestions:
- **Email**: davidkeat@graduate.utm.my
- **GitHub**: (https://github.com/renaissance2005/DDS-AI-Challenge-Project)

