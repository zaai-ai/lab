from typing import Type, Optional
from pydantic import Field, BaseModel

from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from crewai.tools import BaseTool


class YouTubeTranscriptToolInputSchema(BaseModel):
    """
    Tool for fetching the transcript of a YouTube video using the YouTube Transcript API.
    Returns the transcript with text, start time, and duration.
    """

    video_url: str = Field(
        ..., description="URL of the YouTube video to fetch the transcript for."
    )
    language: Optional[str] = Field(
        None, description="Language code for the transcript (e.g., 'en' for English)."
    )


class YouTubeTranscriptToolOutputSchema(BaseModel):
    """
    Output schema for the YouTubeTranscriptTool. Contains the transcript text, duration, comments, and metadata.
    """

    transcript: str = Field(..., description="Transcript of the YouTube video.")
    duration: float = Field(
        ..., description="Duration of the YouTube video in seconds."
    )


class YouTubeTranscriptTool(BaseTool):
    """
    Tool for fetching the transcript of a YouTube video using the YouTube Transcript API.

    Attributes:
        input_schema (YouTubeTranscriptToolInputSchema): The schema for the input data.
        output_schema (YouTubeTranscriptToolOutputSchema): The schema for the output data.
    """

    name: str = "youtube_transcript_tool"
    description: str = (
        "A tool to perform youtube transcript extraction. "
        "Specify the url of the youtube video and optionally the language code."
    )
    args_schema: Type[BaseModel] = YouTubeTranscriptToolInputSchema

    def __init__(self):
        """
        Initializes the YouTubeTranscriptTool.
        """
        super().__init__()

    def _run(
        self, video_url: str, language: Optional[str] = None
    ) -> YouTubeTranscriptToolOutputSchema:
        """
        Runs the YouTubeTranscriptTool with the given parameters.

        Args:
            video_url (list[str]): The list of YouTube video URLs to fetch the transcript for and do have subtitles enabled.
            language (Optional[str]): The language code for the transcript (e.g., 'en' for English).

        Returns:
            YouTubeTranscriptToolOutputSchema: The output of the tool, adhering to the output schema.

        Raises:
            Exception: If fetching the transcript fails.
        """

        video_id = self.extract_video_id(video_url)
        try:
            if language:
                transcripts = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=[language]
                )
            else:
                transcripts = YouTubeTranscriptApi.get_transcript(video_id)
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            raise Exception(
                f"Failed to fetch transcript for video '{video_id}': {str(e)}"
            )

        transcript_text = " ".join([transcript["text"] for transcript in transcripts])
        total_duration = sum([transcript["duration"] for transcript in transcripts])

        return YouTubeTranscriptToolOutputSchema(
            transcript=transcript_text,
            duration=total_duration,
        )

    @staticmethod
    def extract_video_id(url: str) -> str:
        """
        Extracts the video ID from a YouTube URL.

        Args:
            url (str): The YouTube video URL.

        Returns:
            str: The extracted video ID.
        """
        return url.split("v=")[-1].split("&")[0]
