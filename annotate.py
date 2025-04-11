import os
import cv2
import json
import numpy as np
from datetime import datetime

def annotate_images(directory_path):
    """
    Process all images in the given directory and create annotations.
    """
    # Get all image files from the directory
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(directory_path) 
                  if os.path.isfile(os.path.join(directory_path, f)) and 
                  any(f.lower().endswith(ext) for ext in image_extensions)]
    
    if not image_files:
        print(f"No image files found in {directory_path}")
        return
    
    for image_file in image_files:
        image_path = os.path.join(directory_path, image_file)
        annotations = annotate_single_image(image_path)
        
        # Save annotations to JSON
        json_filename = os.path.splitext(image_file)[0] + '_annotations.json'
        json_path = os.path.join(directory_path, json_filename)
        
        with open(json_path, 'w') as f:
            json.dump(annotations, f, indent=2)
        
        print(f"Saved annotations to {json_path}")

def annotate_single_image(image_path):
    """
    Open an image and allow the user to draw bounding boxes and label them.
    Returns annotations in a dictionary format.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return {}
    
    # Create a copy for drawing
    drawing = image.copy()
    window_name = "Annotation Tool"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)  # Adjust as needed
    
    # Image info for annotations
    height, width = image.shape[:2]
    file_name = os.path.basename(image_path)
    
    # Data structure to hold the annotations - match format used in vision.py
    annotations = {
        "objects": []
    }
    
    # Variables for mouse callback
    drawing_box = False
    start_x, start_y = -1, -1
    
    # Mouse callback function
    def draw_rectangle(event, x, y, flags, param):
        nonlocal drawing_box, start_x, start_y, drawing
        
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_box = True
            start_x, start_y = x, y
        
        elif event == cv2.EVENT_MOUSEMOVE and drawing_box:
            # Draw rectangle on a copy of the image
            temp_img = image.copy()
            cv2.rectangle(temp_img, (start_x, start_y), (x, y), (0, 255, 0), 2)
            cv2.imshow(window_name, temp_img)
        
        elif event == cv2.EVENT_LBUTTONUP:
            drawing_box = False
            end_x, end_y = x, y
            
            # Ensure coordinates are in correct order (min/max)
            x_min = min(start_x, end_x)
            y_min = min(start_y, end_y)
            x_max = max(start_x, end_x)
            y_max = max(start_y, end_y)
            
            # Calculate width and height
            box_width = x_max - x_min
            box_height = y_max - y_min
            
            # Minimum size check
            if box_width > 5 and box_height > 5:
                # Draw the final rectangle
                cv2.rectangle(drawing, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.imshow(window_name, drawing)
                
                # Prompt for object type
                cv2.putText(drawing, "Enter type in console (book/misc)", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.imshow(window_name, drawing)
                
                # Get object type
                obj_type = ""
                while obj_type not in ["book", "misc"]:
                    obj_type = input("Enter object type (book/misc): ").lower()
                
                # Initialize annotation with common fields
                annotation = {
                    "type": obj_type,
                    "bounding_box": [x_min, y_min, x_max, y_max],  # [x1, y1, x2, y2] format
                    "confidence": float(input("Enter confidence (0.0-1.0): "))
                }
                
                # Get specific fields based on object type
                if obj_type == "book":
                    annotation.update({
                        "title": input("Enter book title: "),
                        "author": input("Enter author: "),
                        "isbn": input("Enter ISBN (or leave empty): "),
                        "label": None
                    })
                else:  # misc
                    annotation.update({
                        "title": None,
                        "author": None,
                        "isbn": None,
                        "label": input("Enter label for this object: ")
                    })
                
                # Add annotation
                annotations["objects"].append(annotation)
                
                # Display label on image
                display_text = annotation["title"] if obj_type == "book" else annotation["label"]
                cv2.putText(drawing, display_text, (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.imshow(window_name, drawing)
    
    # Set mouse callback
    cv2.setMouseCallback(window_name, draw_rectangle)
    
    # Instructions
    print("\nInstructions:")
    print("1. Click and drag to create a bounding box")
    print("2. Enter object type, confidence, and details in the console when prompted")
    print("3. Press 'q' to finish annotating this image")
    print("4. Press 'c' to clear all annotations and start over\n")
    
    # Display the image
    cv2.imshow(window_name, image)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to exit
        if key == ord('q'):
            break
        
        # Press 'c' to clear annotations
        if key == ord('c'):
            drawing = image.copy()
            annotations["objects"] = []
            cv2.imshow(window_name, drawing)
    
    cv2.destroyAllWindows()
    return annotations

if __name__ == "__main__":
    # Set the directory path containing your test images
    test_directory = "/home/luciano/.llt/exec/bookshelf/test"
    
    # Run the annotation process
    annotate_images(test_directory)