import streamlit as st
import warnings
from dotenv import load_dotenv
from src.crew_zaai.crew import CrewZaai
import sys
import os

# Assuming your crew.py and CrewZaai class are correctly set up in crew_zaai package
# Get the directory where your streamlit_app.py is located
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "src") # Assuming 'src' is in the same directory as streamlit_app.py
if os.path.exists(src_dir):
    sys.path.insert(0, src_dir) # Add 'src' directory to Python path
else:
    src_dir_parent = os.path.dirname(current_dir) # Check if 'src' is in the parent directory
    src_dir_parent_check = os.path.join(src_dir_parent, "src")
    if os.path.exists(src_dir_parent_check):
        sys.path.insert(0, src_dir_parent_check) # Add parent 'src' if found
    else:
        print(f"Warning: 'src' directory not found in '{current_dir}' or parent directory. Import errors may occur.")


warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")
load_dotenv()

def run_crew_zaai(topic):
    """Runs the CrewZaai crew with the given topic and returns the output."""
    try:
        crew_instance = CrewZaai() # Instantiate CrewZaai class
        crew = crew_instance.crew() # Get the crew object
        result = crew.kickoff(inputs={"topic": topic}) # Run kickoff and capture the result
        return result
    except Exception as e:
        return {"error": f"An error occurred while running the crew: {e}"}

def main():
    st.title("CrewZaai AI Agent Runner")

    topic = st.text_input("Enter a Topic for the AI Agents to Explore:", "AI Agents 2024")

    if st.button("Run Crew"):
        if not topic:
            st.warning("Please enter a topic.")
        else:
            with st.spinner(text="Running CrewZaai..."):
                output = run_crew_zaai(topic)

            st.subheader("CrewZaai Output:")
            if "error" in output:
                st.error(output["error"])
            else:
                # Assuming the output is a dictionary or string that can be displayed
                st.write(output) # You might need to format this output better depending on its structure

if __name__ == "__main__":
    main()