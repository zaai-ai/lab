import warnings

from crew_zaai.src.crew_zaai.crew import CrewZaai
from dotenv import load_dotenv


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
load_dotenv()


def run():
    """Run the crew."""

    inputs = {"topic": "AI Agents"}
    CrewZaai().crew().kickoff(inputs=inputs)


if __name__ == "__main__":
    run()
