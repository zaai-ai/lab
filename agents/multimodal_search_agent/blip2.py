from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from transformers import AutoProcessor, Blip2TextModelWithProjection, Blip2VisionModelWithProjection

PROCESSOR = AutoProcessor.from_pretrained("Salesforce/blip2-itm-vit-g")
TEXT_MODEL = Blip2TextModelWithProjection.from_pretrained("Salesforce/blip2-itm-vit-g", torch_dtype=torch.float32).to(
    "cpu"
)
IMAGE_MODEL = Blip2VisionModelWithProjection.from_pretrained(
    "Salesforce/blip2-itm-vit-g", torch_dtype=torch.float32
).to("cpu")


def generate_embeddings(text: Optional[str] = None, image: Optional[JpegImageFile] = None) -> np.ndarray:
    """
    Generate embeddings from text or image using the Blip2 model.
    Args:
        text (Optional[str]): customer input text
        image (Optional[Image]): customer input image
    Returns:
        np.ndarray: embedding vector
    """
    if text:
        inputs = PROCESSOR(text=text, return_tensors="pt").to("cpu")
        outputs = TEXT_MODEL(**inputs)
        embedding = F.normalize(outputs.text_embeds, p=2, dim=1)[:, 0, :].detach().numpy().flatten()
    else:
        inputs = PROCESSOR(images=image, return_tensors="pt").to("cpu", torch.float16)
        outputs = IMAGE_MODEL(**inputs)
        embedding = F.normalize(outputs.image_embeds, p=2, dim=1).mean(dim=1).detach().numpy().flatten()

    return embedding
