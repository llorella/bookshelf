import os
import base64
from openai import OpenAI
import json
from PIL import Image, ImageEnhance
from io import BytesIO
import logging

from yolo_vision import detect_objects_yolo, preprocess_image_yolo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
if not client.api_key:
    logger.error("OPENAI_API_KEY environment variable not set.")
else:
    logger.info("Initialized OpenAI client")

# --- Helper Functions ---
def encode_image_pil(pil_image):
    """Encodes a PIL image to base64 for OpenAI API."""
    buffer = BytesIO()
    pil_image.save(buffer, format="JPEG", quality=85) # Use reasonable quality
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    return encoded

# --- New GPT-4o Crop Analysis Function (Step 3) ---
def analyze_object_crop_gpt4o(crop_image, object_type_hint):
    """
    Analyzes a single image crop using GPT-4o.

    Args:
        crop_image (PIL.Image.Image): The cropped image of the object.
        object_type_hint (str): The type ('book' or 'misc') detected by YOLO.

    Returns:
        dict: A dictionary containing the analysis results from GPT-4o,
              including 'title', 'author', 'label', and 'confidence_assessment'.
              Returns a default structure on failure.
    """
    logger.info(f"Analyzing crop with GPT-4o (type hint: {object_type_hint})")
    base64_crop = encode_image_pil(crop_image)

    # Define the structure we want GPT-4o to return
    json_schema = {
        "type": "object",
        "properties": {
            "title": {"type": ["string", "null"], "description": "Detected book title, null if not a book or not visible."},
            "author": {"type": ["string", "null"], "description": "Detected book author, null if not a book or not visible."},
            "label": {"type": ["string", "null"], "description": "Descriptive label for non-book items, null if it's a book."},
            "confidence_assessment": {"type": "string", "enum": ["confident", "meh", "no clue"], "description": "Your confidence in the identification (title/author for books, label for misc)."}
        },
        "required": ["confidence_assessment"]
    }
    json_schema_string = json.dumps(json_schema)

    # Tailor the prompt based on the YOLO type hint
    if object_type_hint == 'book':
        user_prompt = "Analyze this image, likely a book spine or part of a book. Extract the title and author if clearly visible. If not, state null. Assess your confidence in identifying the text."
        system_prompt = f"""You are an expert book identifier analyzing a cropped image. Focus on identifying book details (title, author). Respond ONLY with a JSON object matching this schema: {json_schema_string}. Assess confidence based *only* on the text visibility in *this specific crop*."""
    else: # misc
        user_prompt = "Analyze this cropped image of an object found on a shelf. Provide a brief descriptive label for the object (e.g., 'white mug', 'picture frame', 'small plant'). Assess your confidence in the label."
        system_prompt = f"""You are an expert object identifier analyzing a cropped image. Identify the object with a short label. Respond ONLY with a JSON object matching this schema: {json_schema_string}. Focus only on the object in the crop."""

    default_failure_output = {
        "title": None, "author": None, "label": "GPT Error", "confidence_assessment": "no clue"
    }

    try:
        logger.info("Calling OpenAI API for crop analysis.")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_crop}", "detail": "high"}} # Use high detail for crops
                ]}
            ],
            response_format={"type": "json_object"}
        )
        logger.info("Received crop analysis response from OpenAI.")

        if response.choices and response.choices[0].message and response.choices[0].message.content:
            response_content = response.choices[0].message.content
            logger.debug(f"Raw GPT-4o crop response: {response_content[:200]}...")
            parsed_result = json.loads(response_content)

            # Validate structure and fill defaults
            parsed_result.setdefault('title', None)
            parsed_result.setdefault('author', None)
            parsed_result.setdefault('label', None)
            parsed_result.setdefault('confidence_assessment', 'no clue') # Default if missing

            # Basic validation of confidence value
            if parsed_result['confidence_assessment'] not in ["confident", "meh", "no clue"]:
                logger.warning(f"Invalid confidence_assessment received: {parsed_result['confidence_assessment']}. Setting to 'no clue'.")
                parsed_result['confidence_assessment'] = 'no clue'

            logger.info(f"GPT-4o analysis result: Confidence='{parsed_result['confidence_assessment']}', Title='{parsed_result.get('title')}', Label='{parsed_result.get('label')}'")
            return parsed_result
        else:
            raise ValueError("No valid response content from API.")

    except (json.JSONDecodeError, ValueError) as json_err:
        logger.error(f"Error parsing GPT-4o crop response: {json_err}", exc_info=True)
        if 'response_content' in locals(): logger.error(f"Raw content: {response_content[:500]}...")
        return default_failure_output
    except Exception as e:
        logger.error(f"Error analyzing crop with GPT-4o: {e}", exc_info=True)
        return default_failure_output


# --- Refactored Main Processing Function (Step 3) ---
def process_shelf_image_yolo_gpt(image_path, yolo_conf_thresh=0.3, crop_padding=5):
    """
    Processes shelf image using YOLOv8 for localization and GPT-4o for detailed
    analysis of cropped regions.

    Args:
        image_path (str): Path to the input shelf image.
        yolo_conf_thresh (float): Confidence threshold for YOLO detections.
        crop_padding (int): Pixels to add around the YOLO box for cropping.

    Returns:
        list: A list of dictionaries, each representing an analyzed object.
              Returns an empty list on major failure.
    """
    logger.info(f"Starting YOLO+GPT processing for: {image_path}")

    # 1. Run YOLO detection
    # We use the vanilla model specified in yolo_vision.py for this prototype
    yolo_detections = detect_objects_yolo(image_path, confidence_thresh=yolo_conf_thresh)

    if not yolo_detections:
        logger.warning("No objects detected by YOLO, cannot proceed.")
        return []

    # 2. Load full image with PIL (handles orientation)
    full_pil_image = preprocess_image_yolo(image_path) # Reusing the yolo preprocess here
    if full_pil_image is None:
        logger.error("Failed to load full image for cropping.")
        return []
    img_width, img_height = full_pil_image.size

    final_analyzed_objects = []
    logger.info(f"Processing {len(yolo_detections)} YOLO detections with GPT-4o...")

    for i, detection in enumerate(yolo_detections):
        yolo_type = detection['type']
        yolo_box = detection['bounding_box']
        yolo_conf = detection['confidence']
        x1, y1, x2, y2 = yolo_box

        logger.debug(f"Processing detection {i+1}: Type={yolo_type}, Box={yolo_box}, Conf={yolo_conf:.2f}")

        # 3. Crop the detected region (Step 2) with padding
        crop_x1 = max(0, x1 - crop_padding)
        crop_y1 = max(0, y1 - crop_padding)
        crop_x2 = min(img_width, x2 + crop_padding)
        crop_y2 = min(img_height, y2 + crop_padding)

        if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
            logger.warning(f"Skipping zero-area crop for box {yolo_box} after padding.")
            continue

        try:
            object_crop_pil = full_pil_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
        except Exception as e_crop:
            logger.error(f"Failed to crop image for box {yolo_box}: {e_crop}", exc_info=True)
            continue

        # 4. Analyze the crop with GPT-4o (Step 3)
        gpt_analysis = analyze_object_crop_gpt4o(object_crop_pil, yolo_type)

        # 5. Process results (Step 4) and combine data
        final_object_data = {
            "yolo_type": yolo_type,
            "yolo_confidence": yolo_conf,
            "bounding_box": yolo_box, # Original YOLO box (not the padded crop box)
            "gpt_confidence": gpt_analysis.get('confidence_assessment', 'no clue'),
            "title": gpt_analysis.get('title'),
            "author": gpt_analysis.get('author'),
            "label": gpt_analysis.get('label'),
            "needs_agentic_search": False # Default
        }

        if final_object_data['gpt_confidence'] == 'meh':
            final_object_data['needs_agentic_search'] = True
            logger.info(f"Detection {i+1} marked as 'meh', flagging for agentic search.")
        elif final_object_data['gpt_confidence'] == 'no clue':
             logger.info(f"Detection {i+1} marked as 'no clue'.")
        # else: 'confident' - data is used as-is

        final_analyzed_objects.append(final_object_data)

    logger.info(f"Finished processing. Returning {len(final_analyzed_objects)} analyzed objects.")
    return final_analyzed_objects


# --- Keep post_process_results? ---
# This function was previously used to filter/sort results from the old pipeline.
# We might need a *new* post-processing function tailored to the output of
# process_shelf_image_yolo_gpt if filtering (e.g., by combined confidence)
# or sorting is still needed downstream. For now, we comment it out.
# def post_process_results(...) -> THIS FUNCTION IS NO LONGER DIRECTLY APPLICABLE

# --- Old functions to be removed/commented out ---
# def preprocess_image(...) # Replaced by using preprocess_image_yolo
# def encode_image(...) # Replaced by encode_image_pil
# def identify_book_rows_on_processed_image(...) # Replaced by YOLO
# def analyze_book_collection_crop(...) # Replaced by analyze_object_crop_gpt4o
# def process_shelf_image(...) # Replaced by process_shelf_image_yolo_gpt

# Note: The actual removal/commenting out will happen in the edit tool call.