import ast
import re
import random
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
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


def scatterplot(dataframe: pd.DataFrame) -> None:
    """
    Create a scatterplot
    Args:
        dataframe (pd.DataFrame): dataframe to be plotted
    """
    plt.figure(figsize=(15, 8))
    sns.scatterplot(data=dataframe, x='Date', y='RAQ', hue='Type', palette=['#FA7302', '#DD4FE4'])
    for i in range(len(dataframe)):
        color = '#DD4FE4'
        if dataframe["Type"].iloc[i] == "Open-Source Models":
            color = '#FA7302'
        plt.text(dataframe['Date'].iloc[i], dataframe['RAQ'].iloc[i] + 0.5, dataframe['Model'].iloc[i], ha='center',
                 color=color)

    # Adding Titles and Labels
    plt.title('Closed-Source vs. Open-Weight Models', fontsize=22, fontweight='bold')
    plt.xlabel('Released Week', fontsize=15)
    plt.ylabel('RAQ', fontsize=15)
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    ax = plt.gca()  # Get current axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%W'))  # Format dates as YYYY-WW

    # Customizing the x-axis for better readability
    plt.xticks(rotation=45)
    plt.ylim(0, dataframe['RAQ'].max() + 2)

    plt.show()


def barplot(dataframe: pd.DataFrame) -> None:
    """
    Create a barplot
    Args:
        dataframe (pd.DataFrame): dataframe to be plotted
    """
    fig, ax = plt.subplots(figsize=(15, 9))

    sns.barplot(ax=ax, data=dataframe, x='Model', y='RAQ', hue='Type', palette=["#fa7302", "#dd4fe4"])

    ax.set_title("RAQ")
    plt.xticks(rotation=45)
    plt.ylabel('RAQ', fontsize=15)
    plt.tight_layout()

    plt.show()
