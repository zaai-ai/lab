from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field


# Define pydantic class with desired data structure
class Assistant(BaseModel):
    """
    Data structure for the model's output.
    """

    answer: str = Field(
        description="A string with the fashion advice for the customer."
    )

class Assistant:
    """
    Assitant class for providing fashion advice.
    """

    def __init__(self, model: ChatVertexAI) -> None:
        """
        Initialize the Chain class by creating the chain.
        Args:
            model (ChatVertexAI): The LLM model.
        """
        super().__init__()

        parser = PydanticOutputParser(pydantic_object=Assistant)

        text_prompt = """
        You are a fashion assistant expert on understanding what a customer needs and on providing recommendations based on the items and customer message below.
        Items:
        {items}

        Customer message:
        {customer_message}

        Instructions:
        1. Read carefully the items metadata.
        2. Read carefully the customer needs.
        3. Provide a fashion advice for the customer based on the items and customer message.
        3. Return a valid JSON with the advice, the key must be 'answer' and the value must be a string with your advice.

        Provide the output as a valid JSON object without any additional formatting, such as backticks or extra text. Ensure the JSON is correctly structured according to the schema provided below.
        {format_instructions}

        Answer:
        """

        prompt = PromptTemplate.from_template(
            text_prompt, partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        self.chain = prompt | model | parser

    def get_advice(self, text: str, items: dict) -> Assistant:
        """
        Get advice from the model based on the text and items context.
        Args:
            text (str): user message.
            items (dict): items found for the customer.
        Returns:
            Assistant: The model's answer.
        """
        try:
            return self.chain.invoke({"customer_message": text, "items": items})
        except Exception as e:
            raise RuntimeError(f"Error invoking the chain: {e}")
