from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import ChatVertexAI
from pydantic import BaseModel, Field


# Define pydantic class with desired data structure
class ClassifierOutput(BaseModel):
    """
    Data structure for the model's output.
    """

    category: list = Field(
        description="A list of clothes category to search for ('t-shirt', 'pants', 'shoes', 'jersey', 'shirt')."  #'complete_outfit'
    )
    number_of_items: int = Field(description="The number of items we should retrieve.")


# - complete_outfit if the customer is looking for a complete outfit.


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

        parser = PydanticOutputParser(pydantic_object=ClassifierOutput)

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
        3. If the customer is looking for multiple items of the same category, return the number of items we should retrieve. If not specfied but the user asked for more than 1, return 2.
        4. If the customer is looking for multiple category, the number of items should be 1.
        5. Return a valid JSON with the categories found, the key must be 'category' and the value must be a list with the categories found and 'number_of_items' with the number of items we should retrieve.

        Provide the output as a valid JSON object without any additional formatting, such as backticks or extra text. Ensure the JSON is correctly structured according to the schema provided below.
        {format_instructions}

        Answer:
        """

        prompt = PromptTemplate.from_template(
            text_prompt, partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        self.chain = prompt | model | parser

    def classify(self, text: str) -> ClassifierOutput:
        """
        Get the category from the model based on the text context.
        Args:
            text (str): user message.
        Returns:
            ClassifierOutput: The model's answer.
        """
        try:
            return self.chain.invoke({"text": text})
        except Exception as e:
            raise RuntimeError(f"Error invoking the chain: {e}")
