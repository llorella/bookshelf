import os
import base64
import requests
from openai import OpenAI
import json
from PIL import Image
from io import BytesIO
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
logger.info("Initialized OpenAI client")

def preprocess_image(image_path):
    """
    Preprocesses the image to optimize for book detection:
    - Adjusts contrast to make text more readable
    - Ensures proper resolution (768px min on shortest side for high detail)
    
    Returns:
        Tuple of (image_buffer, scale_factor, original_width, original_height)
    """
    logger.info(f"Starting image preprocessing for {image_path}")
    try:
        # Open the image
        img = Image.open(image_path)
        logger.debug(f"Successfully opened image: {image_path}")
        
        # Store original dimensions
        original_width, original_height = img.size
        logger.debug(f"Original dimensions: {original_width}x{original_height}")
        
        # Calculate new dimensions (ensuring shortest side is at least 768px for high detail)
        width, height = img.size
        scale_factor = max(768 / min(width, height), 1)
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        logger.debug(f"Calculated scale factor: {scale_factor}")
        logger.debug(f"New dimensions: {new_width}x{new_height}")
        
        # Resize if needed
        if scale_factor > 1:
            logger.info(f"Resizing image with scale factor {scale_factor}")
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Enhance contrast slightly to make text more readable
        from PIL import ImageEnhance
        logger.debug("Applying contrast enhancement")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)  # Slight contrast boost
        
        # Save to a buffer
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        logger.debug("Image saved to buffer")
        
        return buffer, scale_factor, original_width, original_height
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        # Return original image if preprocessing fails
        img = Image.open(image_path)
        width, height = img.size
        logger.warning("Falling back to original image")
        return open(image_path, 'rb'), 1.0, width, height

def encode_image(image_buffer):
    """Convert image to base64 encoding"""
    logger.debug("Converting image to base64")
    encoded = base64.b64encode(image_buffer.read()).decode('utf-8')
    logger.debug(f"Base64 encoding length: {len(encoded)}")
    return encoded

def process_shelf_image(image_path):
    """
    Process a bookshelf image to detect books and other objects.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of detected objects with metadata
    """
    logger.info(f"Processing shelf image: {image_path}")
    
    # Preprocess the image
    processed_image, scale_factor, original_width, original_height = preprocess_image(image_path)
    logger.debug(f"Image preprocessed with scale factor: {scale_factor}")
    
    # Get dimensions of processed image for coordinate reference
    try:
        # Create a copy of the image to get the processed dimensions
        img_copy = Image.open(BytesIO(processed_image.getvalue()))
        processed_width, processed_height = img_copy.size
        logger.debug(f"Processed dimensions: {processed_width}x{processed_height}")
    except:
        # Estimate processed dimensions if we can't read the buffer
        processed_width = int(original_width * scale_factor)
        processed_height = int(original_height * scale_factor)
        logger.warning(f"Using estimated processed dimensions: {processed_width}x{processed_height}")
    
    # Decide on number of divisions based on image size
    max_dimension = max(processed_width, processed_height)
    use_octants = max_dimension > 1500
    logger.debug(f"Max dimension: {max_dimension}, using octants: {use_octants}")
    
    # Generate coordinate reference for the prompt
    if use_octants:
        # Divide into eight sections (2x4 grid)
        divisions = 8
        cols = 4
        rows = 2
    else:
        # Divide into four quadrants
        divisions = 4
        cols = 2
        rows = 2
    logger.debug(f"Grid divisions: {divisions} ({rows}x{cols})")
    
    # Create coordinate reference text
    coordinate_reference = f"""
    Image dimensions: {processed_width}x{processed_height} pixels.
    
    I've divided the image into {divisions} sections for easier reference:
    """
    
    # Generate grid reference
    cell_width = processed_width / cols
    cell_height = processed_height / rows
    logger.debug(f"Cell dimensions: {cell_width}x{cell_height}")
    
    grid_text = ""
    section_number = 1
    
    for r in range(rows):
        for c in range(cols):
            top = int(r * cell_height)
            left = int(c * cell_width)
            right = int((c + 1) * cell_width)
            bottom = int((r + 1) * cell_height)
            
            logger.debug(f"Section {section_number}: ({left},{top}) to ({right},{bottom})")
            grid_text += f"\nSection {section_number}: top-left ({left},{top}), bottom-right ({right},{bottom})"
            section_number += 1
    
    coordinate_reference += grid_text
    
    # Encode the image to base64
    base64_image = encode_image(processed_image)
    logger.debug("Image encoded to base64")
    
    # Define the JSON schema for structured output
    schema = {
        "type": "object",
        "properties": {
            "objects": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["book", "misc"]
                        },
                        "bounding_box": {
                            "type": "array",
                            "items": {"type": "number"}
                        },
                        "title": {"type": ["string", "null"]},
                        "author": {"type": ["string", "null"]},
                        "isbn": {"type": ["string", "null"]},
                        "label": {"type": ["string", "null"]},
                        "confidence": {"type": "number"}
                    },
                    "required": ["type", "bounding_box", "confidence", "title", "author", "isbn", "label"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["objects"],
        "additionalProperties": False
    }
    logger.debug("JSON schema defined")
    
    try:
        logger.info("Calling OpenAI API")
        # Call OpenAI API with structured output
        response = client.responses.create(
            model="gpt-4o",  # Using GPT-4o for best vision capabilities
            input=[
                {
                    "role": "system",
                    "content": [
                        {"type": "input_text", "text": f"""
                        You are an expert at analyzing bookshelf images. Extract detailed information about all visible books and miscellaneous items.
                        
                        {coordinate_reference}
                        
                        For each BOOK:
                        - Provide the most likely title and author based on visible spine text
                        - If you can see an ISBN, include it
                        - Create a precise bounding box: [x1, y1, x2, y2] coordinates (exactly 4 values)
                          where x1,y1 is the top-left corner and x2,y2 is the bottom-right corner
                          Use pixel coordinates matching the image dimensions you're viewing ({processed_width}x{processed_height})
                        - Assign a confidence score (0.0-1.0)
                        
                        For each MISC object (non-book items):
                        - Provide a descriptive label
                        - Create a precise bounding box: [x1, y1, x2, y2] coordinates (exactly 4 values)
                          where x1,y1 is the top-left corner and x2,y2 is the bottom-right corner
                          Use pixel coordinates matching the image dimensions you're viewing ({processed_width}x{processed_height})
                        - Assign a confidence score
                        
                        Be precise with bounding boxes. Coordinates should reflect the exact position of objects in the image.
                        When referencing a book's location, you can mention which section(s) it appears in.
                        For partially visible books, include them if you can identify any text.
                        If text is unclear, provide your best guess and reduce the confidence score accordingly.
                        
                        IMPORTANT: Ensure your bounding box coordinates use the same scale as the image dimensions provided ({processed_width}x{processed_height} pixels).
                        """}
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Analyze this bookshelf image. Identify all books and miscellaneous items."},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": "high"  # Using high detail for better text recognition
                        }
                    ]
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "bookshelf_analysis",  # Required name parameter
                    "schema": schema,
                    "strict": True
                }
            }
        )
        logger.info("Received response from OpenAI API")
        
        # Parse the results
        if response.output_text:
            parsed_result = json.loads(response.output_text)
            objects = parsed_result["objects"]
            logger.info(f"Detected {len(objects)} objects")
            
            # Rescale bounding boxes to match original image dimensions
            if scale_factor != 1.0:
                logger.debug(f"Rescaling bounding boxes with scale factor: {scale_factor}")
                for obj in objects:
                    box = obj["bounding_box"]
                    original_box = box.copy()
                    # Convert from the scaled coordinates to the original image coordinates
                    # Use round() instead of int() to avoid truncation errors
                    box[0] = round(box[0] / scale_factor)  # x1
                    box[1] = round(box[1] / scale_factor)  # y1
                    box[2] = round(box[2] / scale_factor)  # x2
                    box[3] = round(box[3] / scale_factor)  # y2
                    logger.debug(f"Rescaled box from {original_box} to {box}")
            
            return objects
        else:
            # Handle potential refusal or error
            logger.error("No output text in response")
            return []
            
    except Exception as e:
        logger.error(f"Error processing image with OpenAI: {e}")
        
        # For MVP fallback, return mock data if API call fails
        logger.warning("Returning mock data due to API failure")
        return [
            {
                "bounding_box": [100, 50, 200, 300],
                "type": "book",
                "title": "Error: API call failed",
                "author": "Please try again",
                "isbn": "",
                "confidence": 0.1
            }
        ]

def post_process_results(detected_objects, image_width, image_height):
    """
    Performs post-processing on detected objects to improve quality:
    - Filters out low confidence detections
    - Ensures bounding boxes fit within image dimensions
    - Sorts objects by position (left-to-right)
    """
    logger.info(f"Post-processing {len(detected_objects)} detected objects")
    logger.debug(f"Image dimensions: {image_width}x{image_height}")
    
    # Filter out low confidence detections
    filtered_objects = [obj for obj in detected_objects if obj["confidence"] > 0.4]
    logger.debug(f"Filtered to {len(filtered_objects)} objects after confidence threshold")
    
    # Ensure bounding boxes fit within image dimensions
    # Note: We don't normalize coordinates again since they've already been 
    # rescaled to original dimensions in process_shelf_image
    for obj in filtered_objects:
        box = obj["bounding_box"]
        # Only apply limits if coordinates are outside image boundaries
        if box[0] < 0 or box[1] < 0 or box[2] > image_width or box[3] > image_height:
            original_box = box.copy()
            # Ensure bounding box coordinates are within the image boundaries
            box[0] = max(0, min(box[0], image_width))  # x1
            box[1] = max(0, min(box[1], image_height))  # y1
            box[2] = max(0, min(box[2], image_width))  # x2
            box[3] = max(0, min(box[3], image_height))  # y2
            logger.debug(f"Fixed out-of-bounds box from {original_box} to {box}")
    
    # Sort objects by x-coordinate (left to right)
    filtered_objects.sort(key=lambda obj: obj["bounding_box"][0])
    logger.info(f"Returning {len(filtered_objects)} post-processed objects")
    
    return filtered_objects