
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StrOutputParser
from langchain.llms import OpenAI

def evaluate_response(instruction, response, reference_answer):
    """Evaluate a response by comparing it to a reference answer"""
    evaluation_llm = OpenAI(temperature=0)
    EVALUATION_PROMPT = """Rate the following response from 1 to 3 based on its completeness and factual accuracy:
    {instruction}
    Response: {response}
    Reference Answer: {reference_answer}
    Score 1: The response is incomplete or inaccurate.
    Score 2: The response is somewhat accurate.
    Score 3: The response is complete and accurate."""
    eval_prompt = ChatPromptTemplate.from_template(EVALUATION_PROMPT)
    prompt = eval_prompt.format(instruction=instruction, response=response, reference_answer=reference_answer)
    evaluation = evaluation_llm(prompt)
    parsed_output = StrOutputParser().parse(evaluation).strip()
    try:
        score = int(parsed_output.split()[0])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid score value: {parsed_output}")
    return score
