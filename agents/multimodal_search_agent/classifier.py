from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field


# Define pydantic class with desired data structure
class Classifier(BaseModel):
    """
    Data structure for the model's output.
    """

    category: list = Field(
        description="A list of clothes category to search for ('t-shirt', 'pants', 'shoes', 'jersey', 'shirt', 'complete outfit')."
    )

class Classifier:
    """
    Classifier class for classification of input text.
    """

    def __init__(self, model: ChatVertexAI) -> None:
        """
        Initialize the Chain class by creating the chain.
        Args:
            model (ChatVertexAI): The LLM model.
        """
        super().__init__()

        parser = PydanticOutputParser(pydantic_object=Classifier)

        text_prompt = """
        You are a fashion assistant expert on understanding what a customer needs and on extracting the category or categories of clothes a customer wants from the given text.
        Text:
        {text}

        Instructions:
        1. Read carefully the text.
        2. Extract the category or categories of clothes the customer is looking for, it can be:
            - t-shirt if the custimer is looking for a t-shirt.
            - pants if the customer is looking for pants.
            - jacket if the customer is looking for a jacket.
            - shoes if the customer is looking for shoes.
            - jersey if the customer is looking for a jersey.
            - shirt if the customer is looking for a shirt.
            - complete outfit if the customer is looking for a complete outfit.
        3. Return a valid JSON with the categories found, the key must be 'category' and the value must be a list with the categories found.

        Provide the output as a valid JSON object without any additional formatting, such as backticks or extra text. Ensure the JSON is correctly structured according to the schema provided below.
        {format_instructions}

        Answer:
        """

        prompt = PromptTemplate.from_template(
            text_prompt, partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        self.chain = prompt | model | parser

    def classify(self, text: str) -> Classifier:
        """
        Get the category from the model based on the text context.
        Args:
            text (str): user message.
        Returns:
            Classifier: The model's answer.
        """
        try:
            return self.chain.invoke({"text": text})
        except Exception as e:
            raise RuntimeError(f"Error invoking the chain: {e}")
