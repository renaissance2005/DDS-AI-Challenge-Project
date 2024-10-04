
from langchain_community.llms import Ollama # inference model
from langchain_community.chat_models import ChatOllama # vision model
from langchain_community.embeddings import OllamaEmbeddings # embedding model
from langchain_community.document_loaders import UnstructuredPDFLoader # process unstructured PDF
from langchain_community.storage import RedisStore # document store
from langchain_community.utilities.redis import get_client

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_core.documents import Document

from langchain.retrievers.multi_vector import MultiVectorRetriever

from IPython.display import HTML, display, Markdown, Image
from operator import itemgetter
from PIL import Image
from io import BytesIO
import pandas as pd
import base64 # image processing
import os # interacting with system
import time # time calculation
import uuid

# using local model for inference, Ollama must be installed before running the code

# Initialize the model
chatgpt = Ollama(model="llama3.1:8b")


# initialize local embedding in Ollama platform
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
import chromadb

persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("private_rag")

vectorstore = Chroma(
    client=persistent_client,
    collection_name="private_rag",
    embedding_function=embeddings,
)


id_key = "doc_id"

client = get_client('redis://localhost:6379')
docstore = RedisStore(client=client) # you can use filestore, memorystory, any other DB store also

# Create the multi-vector retriever
retriever_multi_vector = MultiVectorRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    id_key=id_key,
)


# Initialize the storage layer - to store raw images, text and tables
client = get_client('redis://localhost:6379')
redis_store = RedisStore(client=client) 

# function to display image from base64 string
def plt_img_base64(img_base64):
    """Disply base64 encoded string as image"""
    # Decode the base64 string
    img_data = base64.b64decode(img_base64)
    # Create a BytesIO object
    img_buffer = BytesIO(img_data)
    # Open the image using PIL
    img = Image.open(img_buffer)
    display(img)

# function to check if the string is base64
def looks_like_base64(sb):
    """Check if the string looks like base64"""
    return re.match("^[A-Za-z0-9+/]+[=]{0,2}$", sb) is not None

# function to check whether the string is an image
def is_image_data(b64data):
    """
    Check if the base64 data is an image by looking at the start of the data
    """
    image_signatures = {
        b"\xff\xd8\xff": "jpg",
        b"\x89\x50\x4e\x47\x0d\x0a\x1a\x0a": "png",
        b"\x47\x49\x46\x38": "gif",
        b"\x52\x49\x46\x46": "webp",
    }
    try:
        header = base64.b64decode(b64data)[:8]  # Decode and get the first 8 bytes
        for sig, format in image_signatures.items():
            if header.startswith(sig):
                return True
        return False
    except Exception:
        return False


def split_image_text_types(docs):
    """
    Split base64-encoded images and texts
    """
    b64_images = []
    texts = []
    for doc in docs:
        # Check if the document is of type Document and extract page_content if so
        if isinstance(doc, Document):
            doc = doc.page_content.decode('utf-8')
        else:
            doc = doc.decode('utf-8')
        if looks_like_base64(doc) and is_image_data(doc):
            b64_images.append(doc)
        else:
            texts.append(doc)
    return {"images": b64_images, "texts": texts}
    


def multimodal_prompt_function(data_dict):
    """
    Create a multimodal prompt with both text and image context.

    This function formats the provided context from `data_dict`, which contains
    text, tables, and base64-encoded images. It joins the text (with table) portions
    and prepares the image(s) in a base64-encoded format to be included in a message.

    The formatted text and images (context) along with the user question are used to
    construct a prompt for multimodal model.
    """
    formatted_texts = "\n".join(data_dict["context"]["texts"])
    messages = ""

    # Adding image(s) to the messages if present
    if data_dict["context"]["images"]:
        for image in data_dict["context"]["images"]:
            image_message = f"Image: data:image/jpeg;base64,{image}\n"
            messages += image_message

    # Adding the text for analysis
    text_message = f"""
        You are an pre-sales specialist well-versed with the company products based on the datasheet and brochure available and
        are competent in matching the requirements from potential user or clients with the available product specification. 
        You will be given context information below which will be a mix of text, tables, and images usually of charts or graphs.
        Use this information to provide answers related to the question which should be related to minimum specification required by the user.
        Your answer should specify a value or text extracted from given document only. 
        DO NOT include preamble or your own analysis. 
        DO NOT give me the sources where you obtain your information or conclusion. 
        DO NOT make up answers, use the provided context documents below and answer the question to the best of your ability. 

        #Example 1: 
        User question: 'Weight: Maximum 3kg'
        Your answer: 'Minimum weight starts at 1.21kg'
        
        #Example 2:
        User question: 'Accessories: laptop bag'
        answer: 'Accessories: not provided'

        #Example 3:
        User question: 'Application Software: Microsoft Office'
        answer: 'Application Software: not provided'

        Let's start:

        User question:
        {data_dict['question']}

        Context documents:
        {formatted_texts}

        Answer:
    """
    
    # Combine text and image message
    messages += text_message
    return [HumanMessage(content=messages)]


# Create RAG chain
multimodal_rag = (
        {
            "context": itemgetter('context'),
            "question": itemgetter('input'),
        }
            |
        RunnableLambda(multimodal_prompt_function)
            |
        chatgpt
            |
        StrOutputParser()
)

# Pass input query to retriever and get context document elements
retrieve_docs = (itemgetter('input')
                    |
                retriever_multi_vector
                    |
                RunnableLambda(split_image_text_types))

# Below, we chain `.assign` calls. This takes a dict and successively
# adds keys-- "context" and "answer"-- where the value for each key
# is determined by a Runnable (function or chain executing at runtime).
# This helps in also having the retrieved context along with the answer generated by GPT-4
multimodal_rag_w_sources = (RunnablePassthrough.assign(context=retrieve_docs)
                                               .assign(answer=multimodal_rag)
)

# Need to improve on this function to output json format
def multimodal_rag_qa(query):
    response = multimodal_rag_w_sources.invoke({'input': query})
    print(response['answer']) 
    return response['answer']

# This part is to evaluate the response from inference model, not applicable for user uploaded excel file
# Load the Excel file
df = pd.read_excel('Sample-TenderDoc.xlsx')
evaluation_llm = Ollama(model="llama3.1:8b")

# Define the evaluation function
def evaluate_response(instruction, response, reference_answer):
    EVALUATION_PROMPT = """You are a fair evaluator language model.

    You will be given an instruction, a response to evaluate, a reference answer that gets a score of 3, and a score rubric representing a evaluation criteria are given.
    1. Write a score that is an integer between 1 and 3. You should refer to the score rubric.
    2. The output format should look as follows: "{{an integer number between 1 and 3}}"
    3. Please do not generate any other opening, closing, and explanations.
    4. Failure to give a numeric value in your output is unacceptable and will result in termination.
     
    The instruction to evaluate:
    {instruction}

    Response to evaluate:
    {response}

    Reference Answer (Score 3):
    {reference_answer}

    Score Rubrics:
    [Is the response complete, accurate, and factual based on the reference answer?]
    Score 1: The response is completely incomplete, inaccurate, and/or not factual.
    Score 2: The response is somewhat complete, accurate, and/or factual.
    Score 3: The response is completely complete, accurate, and/or factual.
    """

    eval_prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT)
    prompt = eval_prompt.format(instruction=instruction, response=response, reference_answer=reference_answer)
    
    evaluation = evaluation_llm(prompt)
    parsed_output = StrOutputParser().parse(evaluation).strip()
    
    # Extract the score from the parsed output
    try:
        score = int(parsed_output.split()[0])
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid score value: {parsed_output}") from e
    
    return score

# Iterate over the rows and generate responses and evaluations
generated_responses = []
evaluation_scores = []

for index, row in df.iterrows():
    min_spec = row['Minimum Specification']
    expected_response = row['Expected Response']
    query = f"Based on the datasheet, what is the closest specification that could meet or exceed this technical requirement: {min_spec} ? If you are asked to specify the product name, model, brand or year of manufacture, just try to get it from the document. If you cannot find, say it is not available. No need to ask for more information."

    # Generate response
    generated_response = multimodal_rag_qa(query)
    # if generated_response.startswith('Generated response for:'):
    #     generated_response = generated_response[len('Generated response for:'):].strip()
    generated_responses.append(generated_response)
    
    # Evaluate response
    try:
        score = evaluate_response(query, generated_response, expected_response)
    except ValueError as e:
        print(f"Error evaluating response for row {index}: {e}")
        score = 1  # or some default value
    evaluation_scores.append(score)

# Add the new columns to the DataFrame
df['Generated Response'] = generated_responses
df['Evaluation Score'] = evaluation_scores

# Save the updated DataFrame to a new Excel file
df.to_excel('Updated-Sample-TenderDoc.xlsx', index=False)






