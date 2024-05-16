import re
import time
from typing import Tuple

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
