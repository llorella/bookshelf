# yolo_vision.py

import os
import cv2
from ultralytics import YOLO
from PIL import Image, ImageEnhance, ImageDraw, ImageFont, ImageOps
from io import BytesIO
import logging

# Configure logging (match level and format with your app.py if desired)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Constants ---\n# Path to the YOLOv8 model file. Start with pre-trained, plan for fine-tuned.
# Using yolov8m.pt provides a good balance of speed and accuracy for general detection.
YOLO_MODEL_PATH = 'yolov8m.pt'
# LATER, REPLACE WITH YOUR FINE-TUNED MODEL:
# YOLO_MODEL_PATH = 'path/to/your/fine_tuned_bookshelf_model.pt'

# Confidence threshold for detections - adjust based on testing
CONFIDENCE_THRESHOLD = 0.4

# Define class names expected from your *fine-tuned* model.
# IMPORTANT: You MUST align this with the classes used during your fine-tuning.
# Example assuming your fine-tuned model maps 0 -> book, 1 -> misc
# Common shelf/room items from COCO dataset that we want to detect
COCO_CLASS_IDS = {
    'book': 73,
    'misc': 1
}

# Class names for fine-tuned model 
FINE_TUNED_CLASS_NAMES = {
    0: 'book',
    1: 'misc'
}


# --- Image Preprocessing (Optional but good practice) ---
def preprocess_image_yolo(image_path):
    """
    Basic preprocessing: Ensure image is readable and in RGB format.
    YOLO models typically handle resizing internally during inference.
    Custom steps like contrast enhancement could be added here if beneficial.

    Args:
        image_path (str): Path to the image file.

    Returns:
        PIL.Image.Image: Image object in RGB format, or None on failure.
    """
    try:
        # Open image first
        img = Image.open(image_path)
        logger.info(f"Opened image: {image_path} (Size before EXIF transpose: {img.size})")

        # Explicitly apply EXIF orientation transpose
        img = ImageOps.exif_transpose(img)
        logger.info(f"Applied EXIF transpose. (Size after EXIF transpose: {img.size})")

        # Ensure it's RGB AFTER transposing (YOLO expects 3 channels)
        img = img.convert("RGB")
        logger.info(f"Converted to RGB. Final processed size: {img.size}")

        # --- Optional Enhancements (Uncomment/Add as needed) ---
        # Example: Slight contrast boost
        # enhancer = ImageEnhance.Contrast(img)
        # img = enhancer.enhance(1.1)
        # logger.debug("Applied contrast enhancement.")
        # --- End Optional Enhancements ---

        return img
    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
        return None
    except Exception as e:
        logger.error(f"Error opening or preprocessing image {image_path}: {e}", exc_info=True)
        return None


# --- YOLOv8 Inference ---
def detect_objects_yolo(image_path, model_path=YOLO_MODEL_PATH, confidence_thresh=CONFIDENCE_THRESHOLD):
    """
    Detects objects in an image using a specified YOLOv8 model.

    Args:
        image_path (str): Path to the input image.
        model_path (str): Path to the YOLOv8 model (.pt file). Can be a standard
                          name like 'yolov8m.pt' (will download if needed) or a
                          path to a custom fine-tuned model.
        confidence_thresh (float): Minimum confidence score for detected objects.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              detected object with keys: 'type', 'bounding_box', 'confidence'.
              Returns an empty list if detection fails or no objects meet the
              threshold.
    """
    logger.info(f"Starting YOLOv8 object detection for: {image_path}")
    logger.info(f"Using model: {model_path} with confidence threshold: {confidence_thresh}")

    # Check if model file exists (only strictly necessary for custom paths)
    is_standard_model = model_path in ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
    if not is_standard_model and not os.path.exists(model_path):
        logger.error(f"Custom model file not found: {model_path}")
        return []
    elif is_standard_model and not os.path.exists(model_path):
        logger.warning(f"Standard model {model_path} not found locally. YOLO will attempt to download it.")

    # Load the image using PIL for consistency
    pil_image = preprocess_image_yolo(image_path)
    if pil_image is None:
        return [] # Error already logged in preprocess_image_yolo

    try:
        # Load the YOLO model
        # If model_path is standard (e.g., 'yolov8m.pt'), Ultralytics handles download.
        # If model_path is a local path, it loads from there.
        model = YOLO(model_path)
        logger.info(f"YOLOv8 model loaded successfully from '{model_path}'")

        # Determine which class name mapping to use
        if is_standard_model:
            current_class_names = model.names
            logger.info("Using default class names from pre-trained model (COCO).")
            using_custom_classes = False
        else:
            # IMPORTANT ASSUMPTION: If using a custom path, assume it's *our* fine-tuned model
            # and use the FINE_TUNED_CLASS_NAMES map.
            current_class_names = FINE_TUNED_CLASS_NAMES
            logger.info("Using custom class names defined for fine-tuned model.")
            using_custom_classes = True
            # Verify the model's classes if possible (advanced)
            # E.g., if model.names exists on custom model, compare keys/values?
            # For now, we trust FINE_TUNED_CLASS_NAMES matches the custom model.


        # Run inference
        # Pass the PIL image directly. YOLO handles conversions.
        results = model(pil_image, conf=confidence_thresh)
        logger.info(f"YOLOv8 inference completed. Found {len(results[0].boxes)} raw detections before filtering.")

        detected_objects = []
        if results and len(results) > 0:
            # Access the Boxes object for the first (and likely only) image result
            boxes = results[0].boxes
            img_width, img_height = pil_image.size # Get dimensions for clamping

            for box in boxes:
                try:
                    # Extract information for each detected box above the threshold
                    conf = box.conf[0].item()
                    cls_id = int(box.cls[0].item())
                    xyxy = box.xyxy[0].tolist() # [x1, y1, x2, y2]

                    # Get class name from the appropriate map
                    raw_class_name = current_class_names.get(cls_id, f"unknown_id_{cls_id}")

                    # --- Map detected class to our application's types ('book', 'misc') ---
                    object_type = 'misc' # Default type
                    if using_custom_classes:
                        # If fine-tuned, directly use the name from our map if it exists
                        if cls_id in FINE_TUNED_CLASS_NAMES:
                            object_type = FINE_TUNED_CLASS_NAMES[cls_id] # Should be 'book' or 'misc'
                        else:
                             logger.warning(f"Detected class ID {cls_id} from custom model not in FINE_TUNED_CLASS_NAMES. Mapping to 'misc'.")
                    else: # Using standard pre-trained COCO model
                        # Try to find the detected class ID in our COCO_CLASS_IDS map
                        found_coco_type = None
                        for name, coco_id in COCO_CLASS_IDS.items():
                            if cls_id == coco_id:
                                found_coco_type = name # e.g., 'book', 'laptop', 'vase'
                                break 
                        
                        if found_coco_type:
                            object_type = found_coco_type
                        else:
                            # If not found in our specific map, keep default 'misc'
                            # Optionally log this for review
                            logger.debug(f"Detected COCO ID {cls_id} ('{raw_class_name}') not in our COCO_CLASS_IDS map. Assigning type 'misc'.")
                            object_type = 'misc'

                    # Format bounding box coordinates as integers and clamp to image bounds
                    x1 = max(0, int(round(xyxy[0])))
                    y1 = max(0, int(round(xyxy[1])))
                    x2 = min(img_width, int(round(xyxy[2])))
                    y2 = min(img_height, int(round(xyxy[3])))

                    # Ensure box has valid dimensions (width/height > 0)
                    if x1 >= x2 or y1 >= y2:
                        logger.warning(f"Skipping zero-area box after clamping: Orig={xyxy}, Clamped=[{x1},{y1},{x2},{y2}] for class '{raw_class_name}'")
                        continue

                    bounding_box = [x1, y1, x2, y2]

                    # Add the processed object to our list
                    detected_objects.append({
                        "type": object_type,
                        "bounding_box": bounding_box,
                        "confidence": float(conf)
                        # Optionally include raw_class_name if needed for debugging
                        # "raw_class": raw_class_name
                    })
                    logger.debug(f"Added: Type='{object_type}', RawClass='{raw_class_name}', Conf={conf:.2f}, Box={bounding_box}")

                except Exception as e_inner:
                    # Catch errors processing a single box without stopping the whole loop
                    logger.error(f"Error processing a single detection box: {e_inner}", exc_info=True)
                    continue # Skip this detection

        logger.info(f"Returning {len(detected_objects)} objects meeting threshold {confidence_thresh}.")
        return detected_objects

    except Exception as e:
        logger.error(f"An error occurred during the YOLOv8 detection process: {e}", exc_info=True)
        return [] # Return empty list on major failure

# --- Example Usage ---
if __name__ == "__main__":
    print("--- Running YOLOv8 Bookshelf Detection Example ---")

    # Directory containing test images
    test_dir = "/home/luciano/.llt/exec/bookshelf/test"
    # Directory to save visualized images
    output_dir = os.path.join(test_dir, "detections_visualized")
    os.makedirs(output_dir, exist_ok=True) # Create output dir if it doesn't exist
    print(f"Scanning directory for JPEG images: {test_dir}")
    print(f"Visualized outputs will be saved to: {output_dir}")

    # Attempt to load a default font, fall back if not available
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except IOError:
        font = ImageFont.load_default()
        print("Default DejaVuSans font not found, using PIL default font.")

    if not os.path.isdir(test_dir):
        print(f"\nERROR: Test directory not found at '{test_dir}'")
    else:
        # Find all jpeg files
        jpeg_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpeg')]

        if not jpeg_files:
            print(f"No .jpeg files found in {test_dir}")
        else:
            print(f"Found {len(jpeg_files)} JPEG files to process.")

            # --- Process each image with Default Pre-trained Model ---
            print("\n--- Processing files with Default Pre-trained YOLOv8 (yolov8m.pt) ---")

            for filename in jpeg_files:
                test_image_path = os.path.join(test_dir, filename)
                print(f"\n>>> Processing: {filename}")

                # Preprocess image using PIL (handles orientation)
                pil_image = preprocess_image_yolo(test_image_path)
                if pil_image is None:
                    print("  Skipping image due to loading error.")
                    continue

                # Detect objects using the PIL image
                detected_items_pretrained = detect_objects_yolo(test_image_path, confidence_thresh=0.3) # detect_objects_yolo uses the path internally

                # --- Visualization using PIL --- 
                draw = ImageDraw.Draw(pil_image)
                output_filename = os.path.splitext(filename)[0] + "_detected_pil.jpeg" # Add _pil to distinguish
                output_path = os.path.join(output_dir, output_filename)
                # --- End Visualization Setup --- 

                if detected_items_pretrained:
                    print(f"  Detected {len(detected_items_pretrained)} objects (type 'book' mapped from COCO, others mapped to 'misc'):")
                    # Sort by type then confidence for cleaner output
                    detected_items_pretrained.sort(key=lambda x: (x['type'], -x['confidence']))
                    for i, item in enumerate(detected_items_pretrained):
                        print(f"    {i+1}. Type: {item['type']:<5}, Confidence: {item['confidence']:.3f}, Box: {item['bounding_box']}")
                        
                        # --- Draw on Image using PIL --- 
                        box = item['bounding_box'] # Already clamped [x1, y1, x2, y2]
                        x1, y1, x2, y2 = box
                        conf = item['confidence']
                        label = f"{item['type']}: {conf:.2f}"
                        
                        # Determine color based on type
                        color = "lime" if item['type'] == 'book' else "blue" # PIL uses color names or RGB tuples
                        text_color = "white"
                        
                        # Draw rectangle (outline)
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=3) # width is line thickness
                        
                        # Calculate text size and position
                        try:
                             # Use textbbox for better size estimation if Pillow >= 9.2.0
                             text_bbox = draw.textbbox((x1, y1), label, font=font)
                             text_width = text_bbox[2] - text_bbox[0]
                             text_height = text_bbox[3] - text_bbox[1]
                        except AttributeError:
                             # Fallback for older Pillow versions
                             text_width, text_height = draw.textsize(label, font=font)

                        label_y = y1 - text_height - 5 if y1 - text_height - 5 > 0 else y1 + 2 # Position label above box, or just below if too close to top
                        text_x = x1
                        
                        # Draw label background rectangle
                        draw.rectangle([text_x, label_y, text_x + text_width + 2, label_y + text_height + 2], fill=color)
                        
                        # Draw text
                        draw.text((text_x + 1, label_y + 1), label, fill=text_color, font=font)
                        # --- End Draw on Image --- 
                else:
                    print("  No objects detected or an error occurred.")
                
                # --- Save visualized PIL image ---
                try:
                    pil_image.save(output_path, "JPEG")
                    print(f"  Saved PIL visualization to: {output_path}")
                except Exception as e_save:
                    print(f"  Error saving PIL visualized image {output_path}: {e_save}")
                # --- End Save --- 

    print("\n--- Example finished ---")

# --- Placeholder for Testing with Fine-tuned Model (Commented out as not needed now) ---
# print("\n--- 2. Testing with Fine-tuned Model (Placeholder) ---")
# # IMPORTANT: Replace this path with the actual path to your trained .pt file
# fine_tuned_model_path = "path/to/your/fine_tuned_bookshelf_model.pt" # <--- CHANGE THIS
#
# print(f"Looking for fine-tuned model at: {fine_tuned_model_path}")
# if os.path.exists(fine_tuned_model_path):
#     print("Fine-tuned model found. Running detection...")
#     detected_items_tuned = detect_objects_yolo(
#         test_image_path, # Need to adapt this if looping
#         model_path=fine_tuned_model_path,
#         confidence_thresh=0.5 # Often use a higher threshold for a tuned model
#     )
#     if detected_items_tuned:
#         print(f"Detected {len(detected_items_tuned)} objects using fine-tuned model:")
#         detected_items_tuned.sort(key=lambda x: (x['type'], -x['confidence']))
#         for i, item in enumerate(detected_items_tuned):
#             print(f"  {i+1}. Type: {item['type']:<5}, Confidence: {item['confidence']:.3f}, Box: {item['bounding_box']}")
#     else:
#         print("No objects detected with fine-tuned model or an error occurred.")
# else:
#     print("Fine-tuned model not found at the specified path. Skipping this test.")
#     print("To run this, train a YOLOv8 model using your annotated data and update the 'fine_tuned_model_path'.")
