import ast
import re
import time
from typing import Tuple

import openai

from generator.generator import Generator


def get_llm_response(
    model: Generator, context: str, query: str
) -> Tuple[str, int, int]:
    """
    Generates an answer from a given LLM based on context and query
    returns the answer and the number of words per second and the total number of words
    Args:
        model (_type_): LLM
        context (str): context data
        query (str): question
    Returns:
        Tuple[str, int, int]: answer, words_per_second, words
    """

    init_time = time.time()
    answer_llm = model.get_answer(context, query)
    total_time = time.time() - init_time
    words_per_second = len(re.sub("[^a-zA-Z']+", " ", answer_llm).split()) / total_time
    words = len(re.sub("[^a-zA-Z']+", " ", answer_llm).split())

    return answer_llm, words_per_second, words


def define_open_ai_function() -> list:
    """
    Create OpenAI function to parse GPT answer
    Returns:
        list: functions to parse gpt answer
    """

    return [
        {
            "name": "return_rank",
            "description": "Return the answer rank",
            "parameters": {
                "type": "object",
                "properties": {
                    "rank": {"type": "string", "description": "The id rank list."},
                },
            },
            "required": ["rank"],
        }
    ]


def get_gpt_rank(true_answer: str, llm_answers: dict, openai_key: str) -> list:
    """
    Based on the true answer, it uses GPT-3.5 to rank the answers of the LLMs
    Args:
        true_answer (str): correct answer
        llm_answers (dict): LLM answers
        openai_key (str): open ai key
    Returns:
        list: rank of LLM IDs
    """

    # get a formated output from OpenAI
    functions = define_open_ai_function()

    gpt_query = f"""Based on the correct answer: {true_answer}, rank the IDs of the following three answers from the most to the least correct one:
        1 {re.sub("[^a-zA-Z0-9']+", ' ', llm_answers['llama'])}
        2 {re.sub("[^a-zA-Z0-9']+", ' ', llm_answers['mistral'])}
        3 {re.sub("[^a-zA-Z0-9']+", ' ', llm_answers['gemma'])}"""

    response = openai.ChatCompletion.create(
        api_key=openai_key,
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": gpt_query}],
        functions=functions,
        function_call={"name": "return_rank"},
    )
    response_message = response["choices"][0]["message"]
    rank = ast.literal_eval(response_message["function_call"]["arguments"])[
        "rank"
    ].split(",")
    if len(rank) == 1:
        rank = list(rank[0])

    return rank
