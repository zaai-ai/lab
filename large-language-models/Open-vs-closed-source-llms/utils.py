import ast
import re
import random
import time
from typing import Tuple

from openai import OpenAI

from generator import Generator


def get_llm_response(
    model: Generator, context: str, query: str
) -> str:
    """
    Generates an answer from a given LLM based on context and query
    returns the answer
    Args:
        model (Generator): LLM
        context (str): context data
        query (str): question
    Returns:
        str: answer
    """

    answer_llm = model.generate_answer(context, query)

    return answer_llm


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


def get_gpt_rank(true_answer: str, llm_answers: dict, openai_key: str) -> dict:
    """
    Based on the true answer, it uses GPT-3.5 to rank the answers of the LLMs
    Args:
        true_answer (str): correct answer
        llm_answers (dict): LLM answers
        openai_key (str): open ai key
    Returns:
        dict: models and its respective rank
    """

    # get a formated output from OpenAI
    functions = define_open_ai_function()

    # shuffle model names
    keys = list(llm_answers.keys())
    random.shuffle(keys)

    # set prompt for GPT
    gpt_query = (f"Based on the correct answer: {true_answer}, rank the IDs of the following sixteen answers from the "
                 f"most to the least correct one:\n")

    # add LLM answers to GPT prompt
    for idx, key in enumerate(keys):
        answer = re.sub("[^a-zA-Z0-9']+", ' ', llm_answers[key])
        gpt_query += f"        ID: {idx + 1} Answer: {answer}\n"

    # make request to GPT
    completion = OpenAI(api_key=openai_key).chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": gpt_query}],
        functions=functions,
        function_call={"name": "return_rank"},
    )
    response_message = completion.choices[0].message.function_call.arguments

    # retrive the rank and convert into dict
    rank = ast.literal_eval(response_message)["rank"].split(",")
    if len(rank) == 1:
        rank = list(rank[0])

    rank_dict = {
        key: (rank.index(str(idx + 1)) + 1) if str(idx + 1) in rank else len(rank) + 1
        for idx, key in enumerate(keys)
    }

    return rank_dict
