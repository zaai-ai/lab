from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field


# Define pydantic class with desired data structure
class AssistantOutput(BaseModel):
    """
    Data structure for the model's output.
    """

    answer: str = Field(description="A string with the fashion advice for the customer.")


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

        parser = PydanticOutputParser(pydantic_object=AssistantOutput)

        text_prompt = """
        You work for a fashion store and you are a fashion assistant expert on understanding what a customer needs.
        Based on the items that are available in the store and the customer message below, provide a fashion advice for the customer.
        Number of items: {number_of_items}
        
        Images of items:
        {items}

        Customer message:
        {customer_message}

        Instructions:
        1. Check carefully the images provided.
        2. Read carefully the customer needs.
        3. Provide a fashion advice for the customer based on the items and customer message.
        4. Return a valid JSON with the advice, the key must be 'answer' and the value must be a string with your advice.

        Provide the output as a valid JSON object without any additional formatting, such as backticks or extra text. Ensure the JSON is correctly structured according to the schema provided below.
        {format_instructions}

        Answer:
        """

        prompt = PromptTemplate.from_template(
            text_prompt, partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        self.chain = prompt | model | parser

    def get_advice(self, text: str, items: list, number_of_items: int) -> AssistantOutput:
        """
        Get advice from the model based on the text and items context.
        Args:
            text (str): user message.
            items (list): items found for the customer.
            number_of_items (int): number of items to be retrieved.
        Returns:
            AssistantOutput: The model's answer.
        """
        try:
            return self.chain.invoke({"customer_message": text, "items": items, "number_of_items": number_of_items})
        except Exception as e:
            raise RuntimeError(f"Error invoking the chain: {e}")
