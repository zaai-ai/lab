from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


class Generator:
    """Generator, aka LLM, to provide an answer based on some question and context."""

    def __init__(self, model: str, api_key: str) -> None:
        # Template for the prompt
        self.template = """
            Use the following pieces of context to give a succinct and clear answer to the question at the end:
            {context}
            Question: {question}
            Answer:
        """

        if "claude" in model:
            self.llm = ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=0.1,
            )

        elif "gpt" in model:
            self.llm = ChatOpenAI(
                base_url="https://api.openai.com/v1",
                api_key=api_key,
                model=model,
                temperature=0.1,
            )

        elif "gemini" in model:
            self.llm = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model,
                temperature=0.1,
            )

        else:
            self.llm = ChatOpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key,
                model=model,
                temperature=0.1,
            )

        # Create the prompt template
        self.prompt = PromptTemplate(
            template=self.template, input_variables=["context", "question"]
        )

    def generate_answer(self, context: str, question: str) -> str:
        """
        Get the answer from the LLM based on context and user's question.

        Args:
            context (str): Context data to be considered.
            question (str): The user's question.

        Returns:
            str: The generated answer from the LLM.
        """
        query_llm = LLMChain(
            llm=self.llm,
            prompt=self.prompt,
        )

        answer = query_llm.invoke({"context": context, "question": question})

        return answer["text"]
