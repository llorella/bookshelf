import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import uuid
import logging # Added for logging
import json
import datetime

from database import (
    init_db, create_user, create_shelf, add_shelf_image,
    add_shelf_object,
    get_shelf_objects, update_shelf_object, publish_shelf, get_shelf,
)
from vision import process_shelf_image_yolo_gpt
from yolo_vision import preprocess_image_yolo

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Configure logging
logging.basicConfig(level=logging.INFO) # Use INFO level for production, DEBUG for dev

# Add Jinja2 filters
@app.template_filter('format_datetime')
def format_datetime(value, format='%B %d, %Y at %I:%M %p'):
    if value is None:
        return ""
    if isinstance(value, str):
        try:
            value = datetime.datetime.fromisoformat(value)
        except ValueError:
            return value
    return value.strftime(format)

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize database
init_db()

# Create a test user for MVP (consider more robust user handling later)
try:
    test_user_id = create_user('testuser', 'test@example.com')
    logging.info(f"Test user created with ID: {test_user_id}")
except Exception as e:
    # This might happen if the user already exists on restart
    logging.warning(f"Could not create test user (might already exist): {e}")
    # Attempt to retrieve the existing user ID - requires a get_user_by_username function
    # For MVP, we might just assume it exists or handle login differently.
    # Setting a placeholder - THIS NEEDS A FIX for robustness
    test_user_id = "existing_testuser_placeholder_id"

@app.route('/')
def index():
    return render_template('index.html', user_id=test_user_id)

# Updated upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            app.logger.warning("No file part in request")
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            app.logger.warning("No selected file")
            return redirect(request.url)

        if file:
            try:
                # Save the uploaded file
                filename = secure_filename(str(uuid.uuid4()) + os.path.splitext(file.filename)[1])
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                app.logger.info(f"File saved to {filepath}")

                # Create relative URL for the image
                image_url = '/uploads/' + filename

                # Create a new shelf
                shelf_id = create_shelf(test_user_id)
                app.logger.info(f"Created shelf with ID: {shelf_id}")

                # --- Get Processed Dimensions ---
                processed_pil_image = preprocess_image_yolo(filepath)
                if processed_pil_image is None:
                    raise ValueError("Failed to preprocess image to get dimensions.")
                processed_width, processed_height = processed_pil_image.size
                app.logger.info(f"Image processed dimensions (after potential EXIF rotation): {processed_width}x{processed_height}")
                # --- End Get Dimensions ---

                # Add image to shelf WITH dimensions
                add_shelf_image(shelf_id, image_url, processed_width, processed_height)
                app.logger.info(f"Added image {image_url} with dimensions to shelf {shelf_id}")

                # Process the image using the new YOLO+GPT pipeline
                app.logger.info("Starting YOLO+GPT image processing...")
                # Pass filepath, the function internally preprocesses again if needed
                analyzed_objects = process_shelf_image_yolo_gpt(filepath)
                app.logger.info(f"Pipeline returned {len(analyzed_objects)} analyzed objects.")

                # --- No Post-processing applied yet --- 
                # Optional: Add call to post_process_yolo_gpt_results if implemented
                # processed_objects = post_process_yolo_gpt_results(analyzed_objects)
                # app.logger.info(f"Post-processing resulted in {len(processed_objects)} objects.")
                processed_objects = analyzed_objects # Use raw results for now

                # --- Remove collection creation logic --- 
                # collection_temp_id_map = {}
                # for i, coll_data in enumerate(processed_collections):
                #     real_collection_id = add_shelf_collection(shelf_id, coll_data['bounding_box'], coll_data.get('label'))
                #     temp_id = coll_data.get('temp_id', f'fallback_temp_col_{i}')
                #     collection_temp_id_map[temp_id] = real_collection_id
                #     app.logger.info(f"Added collection {real_collection_id} (Temp: {temp_id}) to DB.")

                # Add detected objects to database
                # No longer need to link to collection_id here
                objects_added_count = 0
                for obj_data in processed_objects:
                    try:
                        # Pass None for collection_id
                        # obj_data structure comes from process_shelf_image_yolo_gpt
                        add_shelf_object(shelf_id, None, obj_data)
                        objects_added_count += 1
                    except Exception as db_err:
                        app.logger.error(f"Error adding object to DB: {db_err}. Object data: {obj_data}", exc_info=True)
                        # Decide if you want to continue or stop processing
                
                app.logger.info(f"Added {objects_added_count} objects to the database for shelf {shelf_id}.")

                # Redirect to edit page (Template needs update later)
                return redirect(url_for('edit_shelf', shelf_id=shelf_id))

            except Exception as e:
                app.logger.error(f"Error during file upload and processing: {e}", exc_info=True)
                # Render an error page or return JSON error
                return render_template('error.html', error="An error occurred during image processing."), 500

    return render_template('upload.html')

# Updated API ingest route
@app.route('/api/shelf/ingest', methods=['POST'])
def api_ingest():
    data = request.json
    user_id = data.get('user_id')
    image_urls = data.get('image_urls', [])

    if not user_id or not image_urls:
        return jsonify({"error": "Missing required fields"}), 400

    try:
        # Create a new shelf
        shelf_id = create_shelf(user_id)
        app.logger.info(f"[API] Created shelf {shelf_id} for user {user_id}")

        # Process only the first image for simplicity in MVP
        image_url = image_urls[0]
        # Get the image path (same logic as before)
        if image_url.startswith('/uploads/'):
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_url.split('/')[-1])
        else:
            image_path = image_url 
            app.logger.warning(f"[API] Processing image from non-standard path: {image_path}")
            # If it's an external URL, you'd download it here first
        
        if not os.path.exists(image_path):
             app.logger.error(f"[API] Image path not found: {image_path}")
             return jsonify({"error": "Image specified by URL not found"}), 400

        # --- Get Processed Dimensions ---
        processed_pil_image = preprocess_image_yolo(image_path)
        if processed_pil_image is None:
            raise ValueError("Failed to preprocess image to get dimensions.")
        processed_width, processed_height = processed_pil_image.size
        app.logger.info(f"[API] Image processed dimensions: {processed_width}x{processed_height}")
        # --- End Get Dimensions ---

        # Add image to shelf WITH dimensions
        # For API, image_url might be the path or original URL, depends on requirements
        add_shelf_image(shelf_id, image_url, processed_width, processed_height)
        app.logger.info(f"[API] Added image {image_url} with dimensions to shelf {shelf_id}")

        # Process the image using new pipeline
        app.logger.info("[API] Starting YOLO+GPT image processing...")
        analyzed_objects = process_shelf_image_yolo_gpt(image_path)
        app.logger.info(f"[API] Pipeline returned {len(analyzed_objects)} analyzed objects.")

        # --- No Post-processing applied yet ---
        processed_objects = analyzed_objects # Use raw results for now

        # --- Remove collection logic ---
        # collection_temp_id_map = {}
        # ...

        # Store objects
        objects_added_count = 0
        for obj_data in processed_objects:
            try:
                # Pass None for collection_id
                add_shelf_object(shelf_id, None, obj_data)
                objects_added_count += 1
            except Exception as db_err:
                 app.logger.error(f"[API] Error adding object to DB: {db_err}. Object data: {obj_data}", exc_info=True)
        app.logger.info(f"[API] Added {objects_added_count} objects to DB for shelf {shelf_id}.")

        # Prepare response (get_shelf now returns flat object list)
        final_shelf_data = get_shelf(shelf_id)
        if not final_shelf_data:
             return jsonify({"error": "Failed to retrieve shelf data after processing"}), 500

        # Return the shelf data including the flat list of objects
        return jsonify(final_shelf_data)

    except Exception as e:
        app.logger.error(f"[API] Error during shelf ingestion: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during ingestion."}), 500


@app.route('/edit/<shelf_id>', methods=['GET'])
def edit_shelf(shelf_id):
    # get_shelf now returns flat object list under 'objects' key
    shelf = get_shelf(shelf_id)
    if not shelf:
        app.logger.warning(f"Shelf not found for editing: {shelf_id}")
        return "Shelf not found", 404

    # Log the data being sent to the template
    app.logger.debug(f"Data for edit_shelf template (shelf_id={shelf_id}): {json.dumps(shelf, indent=2)}")

    # The template 'edit_shelf.html' needs to be updated separately
    # to handle the new flat shelf['objects'] structure instead of nested collections.
    return render_template('edit_shelf.html', shelf=shelf)

# --- Removed Collection Update Route --- 
# @app.route('/api/collection/update', methods=['POST'])
# def api_update_collection():
#    # ... (This functionality is removed as collections are not managed this way now)
#    return jsonify({"error": "Collection API endpoint is disabled in current version"}), 404

@app.route('/api/shelf/update', methods=['POST'])
def api_update():
    data = request.json
    shelf_id = data.get('shelf_id')
    # API expects a list of objects, potentially modified
    # These objects should now correspond to the flat list structure
    objects_to_update = data.get('objects', [])
    publish = data.get('publish', False)

    if not shelf_id:
        return jsonify({"error": "Missing shelf_id"}), 400

    try:
        app.logger.info(f"[API Update] Updating shelf {shelf_id}")
        # Update each object sent in the request
        update_count = 0
        for obj_update_data in objects_to_update:
            object_id = obj_update_data.get('object_id') # Expecting object_id from the client
            if object_id:
                # Pass the whole dict (excluding object_id) to update function
                update_payload = {k: v for k, v in obj_update_data.items() if k != 'object_id'}
                update_shelf_object(object_id, update_payload)
                update_count += 1
            else:
                app.logger.warning(f"[API Update] Received object data without object_id: {obj_update_data}")
        app.logger.info(f"[API Update] Processed updates for {update_count} objects.")

        # Publish if requested
        if publish:
            publish_shelf(shelf_id)
            app.logger.info(f"[API Update] Published shelf {shelf_id}")

        return jsonify({
            "status": "success",
            "shelf_id": shelf_id
        })
    except Exception as e:
        app.logger.error(f"[API Update] Error updating shelf {shelf_id}: {e}", exc_info=True)
        return jsonify({"error": "An internal error occurred during update."}), 500

@app.route('/view/<shelf_id>')
def view_shelf(shelf_id):
    # get_shelf now returns flat object list under 'objects' key
    shelf = get_shelf(shelf_id)
    if not shelf:
        return "Shelf not found", 404

    # The view template 'view_shelf.html' also needs updating separately.
    return render_template('view_shelf.html', shelf=shelf)

@app.route('/api/shelf/view/<shelf_id>')
def api_view_shelf(shelf_id):
    shelf = get_shelf(shelf_id) # Already returns the new structure
    if not shelf:
        return jsonify({"error": "Shelf not found"}), 404

    # The get_shelf function already structures the data correctly
    return jsonify(shelf)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.errorhandler(Exception)
def handle_exception(e):
    # Log the error using Flask's logger
    app.logger.error(f"Unhandled exception: {str(e)}", exc_info=True) # Include traceback

    # Create a user-friendly error message
    # Check for specific OpenAI errors if the client library raises them specifically
    # (Requires knowing the specific exception types, e.g., openai.APIError)
    # if isinstance(e, openai.APIError): # Example
    #     error_message = "There was an issue analyzing your bookshelf image with the AI service. Please try again later."
    if "OpenAI" in str(e): # Fallback check in string representation
        error_message = "There was an issue analyzing your bookshelf image. Please try again later."
    else:
        error_message = "An unexpected server error occurred. Please try again or contact support."

    # Return JSON for API requests, HTML otherwise
    if request.path.startswith('/api/'):
        return jsonify({"error": error_message}), 500
    else:
        return render_template('error.html', error=error_message), 500

if __name__ == '__main__':
    # Use environment variable for debug mode, default to False for safety
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    app.run(debug=debug_mode, host='0.0.0.0', port=5000) # Bind to 0.0.0.0 to be accessible externally if needed

