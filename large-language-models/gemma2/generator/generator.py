from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp

from base.config import Config


class Generator(Config):
    """Generator, aka LLM, to provide an answer based on some question and context"""

    def __init__(self, model) -> None:
        super().__init__()

    # template
        self.template = """
            Use the following pieces of context to answer the question at the end.
            {context}
            Question: {question}
            Answer:
        """

   # load llm from local file
        self.llm = LlamaCpp(
            model_path=f"{self.parent_path}/{self.config['generator'][model]['llm_path']}",
            n_ctx=self.config["generator"]["context_length"],
            temperature=self.config["generator"]["temperature"],
            verbose=False,
        )

        # create prompt template
        self.prompt = PromptTemplate(
            template=self.template, input_variables=["context", "question"]
        )

    def get_answer(self, context: str, question: str) -> str:
        """
        Get the answer from llm based on context and user's question
        Args:
            context (str): most similar document retrieved
            question (str): user's question
        Returns:
            str: llm answer
        """

        query_llm = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
            llm_kwargs={"max_tokens": self.config["generator"]["max_tokens"]},
            verbose=False,
        )

        return query_llm.run({"context": context, "question": question})
