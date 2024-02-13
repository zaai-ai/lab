import os
from typing import Optional

import moviepy.editor as mp


def convert_to_wav(input_file: str, output_file: Optional[str] = None) -> None:
    """
    Converts an audio file to WAV format using FFmpeg.
    Args:
        input_file (str): The path of the input audio file to convert.
        output_file (str): The path of the output WAV file. If None, the output file will be created by replacing the input file
        extension with ".wav".
    Returns:
        None
    """
    if not output_file:
        output_file = os.path.splitext(input_file)[0] + ".wav"

    clip = mp.VideoFileClip(input_file)
    clip.audio.write_audiofile(output_file, codec="pcm_s16le")
