import base64


def encode_image_to_base64(image_path: str) -> str:
    """
    Read an image file and return its Base64-encoded string.
    Args:
        image_path (str): path to the image file
    Returns:
        str: Base64-encoded string
    """
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
    return base64.b64encode(image_data).decode("utf-8")
