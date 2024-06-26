from langchain import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain


class Generator:
    """Generator, aka LLM, to provide an answer based on some question and context"""

    def __init__(self, model: str, ngc_key: str) -> None:

        # template
        self.template = """
            Use the following pieces of context to give a succinct and clear answer to the question at the end:
            {context}
            Question: {question}
            Answer:
        """

        # llm
        self.llm = ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=ngc_key,
            model=model,
            temperature=0.1)

        # create prompt template
        self.prompt = PromptTemplate(
            template=self.template, input_variables=["context", "question"]
        )

    def generate_answer(self, context: str, question: str) -> str:
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
            llm_kwargs={"max_tokens": 2000},
        )

        answer = query_llm.invoke(
            {"context": context, "question": question}
        )
        
        return answer['text']
