import subprocess
import time
import pyautogui
import base64
import google.generativeai as genai
import json
import re
import os
from dotenv import load_dotenv
from PIL import Image, ImageDraw


############################################
# 1) Environment Setup
############################################

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set GEMINI_API_KEY in your .env file.")

genai.configure(api_key=api_key)

CHROME_PATH="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SCREENSHOT_PATH = "assets/zaai_homepage.png"
SCREENSHOT_BBOXED_PATH = "assets/zaai_homepage_bboxed.png"
SCREENSHOT_BLOG_PATH = "assets/zaai_lab.png"

ZAAI_URL = "https://zaai.ai"


############################################
# 2) Utility Functions
############################################

def encode_image_to_base64(image_path: str) -> str:
    """ Read an image file and return its Base64-encoded string. """
    with open(image_path, "rb") as img_file:
        image_data = img_file.read()
    return base64.b64encode(image_data).decode("utf-8")


def extract_json_from_response(response) -> dict:
    """
    Extract JSON content from the LLM response, removing any code fence markers.
    """
    if not hasattr(response, "candidates") or not response.candidates:
        raise ValueError("Response does not contain valid candidates.")

    raw_text = response.candidates[0].content.parts[0].text
    json_str = re.sub(r"^```json|```$", "", raw_text.strip(), flags=re.MULTILINE)

    try:
        parsed_data = json.loads(json_str)
        return parsed_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}\nRaw LLM Response:\n{raw_text}")


def update_coordinates_to_pixels(detection_info: dict, width: int, height: int) -> None:
    """
    Convert normalized bounding box coordinates ([0..1000]) to actual pixel values.
    """
    for key, value in detection_info.items():
        coords = value["coordinates"]
        xmin, ymin, xmax, ymax = coords
        value["coordinates"] = [
            (xmin / 1000.0) * width,
            (ymin / 1000.0) * height,
            (xmax / 1000.0) * width,
            (ymax / 1000.0) * height
        ]


def draw_bounding_boxes(image_path: str, detection_info: dict, output_path: str, color: str = "red") -> None:
    """
    Draw bounding boxes on the image and save to output_path.
    Expects detection_info to have pixel coordinates already.
    """
    try:
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)

            for label, details in detection_info.items():
                coords = details["coordinates"]
                description = details.get("description", "")

                draw.rectangle(coords, outline=color, width=2)
                draw.text((coords[0], coords[1] - 10), label, fill=color)

            img.save(output_path)
            print(f"Image saved with bounding boxes at: {output_path}")
    except Exception as e:
        print(f"Failed to create image with bounding boxes: {e}")


def take_screenshot(output_path: str):
    """Take a screenshot of the main screen (or the active window)."""
    time.sleep(2)  # wait a bit for the page to load
    screenshot = pyautogui.screenshot()
    screenshot.save(output_path)
    print(f"Screenshot saved to {output_path}.")


############################################
# 3) Vision Agent Steps
############################################

def identify_elements_with_descriptions(image_path: str) -> list:
    """
    Step 1: Ask the model to identify clickable elements and include descriptions.
    Return a list of objects, each containing 'label' and 'description'.
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    encoded_image = encode_image_to_base64(image_path)

    prompt = """
    You are given a screenshot of a website homepage.
    Identify all the relevant clickable elements (text, buttons, icons, tabs, images, etc.)
    on the website page only (discard browser elements if they appear in the image) and provide:
      - A semantically rich name as the label (e.g., "Lab Link" or "Blog Tab")
      - A short description of its purpose on the page and any relevant visual details

    Output JSON in this format:
    {
      "elements": [
        {
          "label": "some descriptive label",
          "description": "short description with visual nuances"
        },
        ...
      ]
    }
    """

    response = model.generate_content([
        {"mime_type": "image/png", "data": encoded_image},
        prompt
    ])

    parsed_data = extract_json_from_response(response)
    if "elements" not in parsed_data:
        raise ValueError("No 'elements' field found in the JSON response.")

    return parsed_data["elements"]


def propose_bounding_boxes(image_path: str, identified_elements: list) -> dict:
    """
    Step 2: Provide the list of elements from Step 1 (labels + descriptions).
    Ask the model to propose bounding boxes in [xmin, ymin, xmax, ymax] with 0..1000 scale.
    Return a dict where keys are labels, and values have 'coordinates' + 'description'.
    We also copy the 'description' from the elements so that we keep it in the final output.
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    encoded_image = encode_image_to_base64(image_path)

    elements_json_str = json.dumps(identified_elements, indent=2)

    prompt = f"""
    The following clickable elements were identified (labels + descriptions):
    {elements_json_str}

    Propose a bounding box (in [xmin, ymin, xmax, ymax], 0..1000 scale) for each element 
    so we can locate them on the screenshot.

    Output JSON in the format:
    {{
      "<element_label>": {{
        "coordinates": [xmin, ymin, xmax, ymax],
        "description": "<the same description from above>"
      }},
      ...
    }}
    """

    response = model.generate_content([
        {"mime_type": "image/png", "data": encoded_image},
        prompt
    ])

    parsed_data = extract_json_from_response(response)
    return parsed_data

############################################
# 4) High-Level Flow
############################################

def open_chrome(url: str):
    """Open Chrome to a specific URL using a subprocess."""
    print(f"Opening Chrome at {url} ...")
    subprocess.Popen([CHROME_PATH, url])
    time.sleep(5)


def find_and_click_lab_element(bounding_box_data: dict):
    """
    Click the bounding box that should lead to the 'Lab' (or blog page).
    For simplicity, let's assume we pick the bounding box whose label
    or description references "Lab" or "Blog"
    """
    target_label = None
    for label, info in bounding_box_data.items():
        lower_desc = info["description"].lower()
        lower_label = label.lower()
        if "lab" in lower_desc or "lab" in lower_label:
            target_label = label
            break
        if "blog" in lower_desc or "blog" in lower_label:
            target_label = label
            break

    if not target_label:
        print("Could not find a bounding box that references Lab/Blog in the description.")
        return

    coords = bounding_box_data[target_label]["coordinates"]
    # coords is [xmin, ymin, xmax, ymax]
    # let's pick the down left corner of the bb
    x_center = coords[0] / 2 # bc of retina res
    y_center = coords[1] / 2 # bc of retina res

    print(f"Clicking element: {target_label}")
    pyautogui.moveTo(x_center, y_center, duration=0.5)
    pyautogui.click()


def retrieve_latest_blog_info(image_path: str) -> (str, str):
    """
    Getting the latest post title and date
    """
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")
    encoded_image = encode_image_to_base64(image_path)

    prompt = """
    You are given a screenshot of a website blog page.
    Identify the latest article and get its title and date.
    
    Output JSON in this format:
    {
      "title": "title of the article",
      "pub_date": "date of publishing of the article"
    },
    """

    response = model.generate_content([
        {"mime_type": "image/png", "data": encoded_image},
        prompt
    ])

    parsed_data = extract_json_from_response(response)
    if "title" not in parsed_data:
        raise ValueError("No 'title' field found in the JSON response.")

    return parsed_data['title'], parsed_data['pub_date']


def main():
    open_chrome(ZAAI_URL)
    take_screenshot(SCREENSHOT_PATH)

    # pass the screenshot to the 'vision agent' to get bounding boxes
    print("Identifying elements...")
    identified_elems = identify_elements_with_descriptions(SCREENSHOT_PATH)
    bounding_boxes = propose_bounding_boxes(SCREENSHOT_PATH, identified_elems)

    # convert [0..1000] coords to actual pixel coords, then draw them
    with Image.open(SCREENSHOT_PATH) as img:
        width, height = img.size
    update_coordinates_to_pixels(bounding_boxes, width, height)
    draw_bounding_boxes(SCREENSHOT_PATH, bounding_boxes, SCREENSHOT_BBOXED_PATH)

    # another agent decides which element is correct to go to the Lab (or Blog) page
    # and gets the info on the latest post
    find_and_click_lab_element(bounding_boxes)

    take_screenshot(SCREENSHOT_BLOG_PATH)
    latest_title, latest_url = retrieve_latest_blog_info(SCREENSHOT_BLOG_PATH)

    print(f"\nLatest Blog Article: {latest_title}")
    print(f"Article Date: {latest_url}")


if __name__ == "__main__":
    main()
