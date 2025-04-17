import streamlit as st
import warnings
from dotenv import load_dotenv
from crew_zaai.crew import CrewZaai
import sys
import os
import json

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
    st.title("Multi Agents AI for research/Summarizer and Linkindin post") # Updated title

    topic = st.text_input("Enter a Topic for the AI Agents to Explore:", "Agentic AI 2025")

    if st.button("Button"):
        if not topic:
            st.warning("Please enter a topic.")
        else:
            with st.spinner(text="Running ..."):
                output = run_crew_zaai(topic)

            st.subheader("Task Outputs:") # Updated subheader

            if "error" in output:
                st.error(output["error"])
            else:
                if isinstance(output, dict) and "tasks_output" in output:
                    tasks_output_list = output["tasks_output"] # Get the list of TaskOutput objects

                    for i, task_output in enumerate(tasks_output_list):
                        st.subheader(f"Task {i+1}: {task_output.summary}") # Display task summary as subheader

                        try:
                            # Try to parse task_output.raw as JSON (dictionary)
                            raw_dict = json.loads(task_output.raw)
                            st.json(raw_dict) # Display as JSON if parsing is successful
                        except json.JSONDecodeError:
                            # If JSON parsing fails, display as plain text
                            st.text_area(f"Task {i+1} Output (Plain Text)", value=task_output.raw, height=1500)
                        except TypeError: # Handle cases where task_output.raw might not be a string
                            st.text_area(f"Task {i+1} Output (Plain Text, Non-String Raw)", value=str(task_output.raw), height=1500)


                else:
                    st.write(output) # Fallback to raw output if format is unexpected


if __name__ == "__main__":
    main()