from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))) # Add current directory of crew.py to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')) # Add parent directory (src) to path, assuming 'tools' is in 'src'
from tools.searx import SearxSearchTool
from tools.youtube import YouTubeTranscriptTool


@CrewBase
class CrewZaai:
    """CrewZaai crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def researcher(self) -> Agent:
        search_tool = SearxSearchTool(
            searx_host=os.getenv("SEARXNG_BASE_URL"), unsecure=False
        )

        return Agent(
            config=self.agents_config["researcher"], tools=[search_tool], verbose=True
        )

    @agent
    def summarizer(self) -> Agent:
        youtube_tool = YouTubeTranscriptTool()

        return Agent(
            config=self.agents_config["summarizer"], tools=[youtube_tool], verbose=True
        )

    @agent
    def linkedin_post_writer(self) -> Agent:
        return Agent(config=self.agents_config["linkedin_post_writer"], verbose=True)

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config["research_task"],
        )

    @task
    def summarizer_task(self) -> Task:
        return Task(
            config=self.tasks_config["summarize_task"],
        )

    @task
    def write_task(self) -> Task:
        return Task(
            config=self.tasks_config["write_task"], output_file="assets/report.html"
        )

    @crew
    def crew(self) -> Crew:
        """Creates the CrewZaai crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
