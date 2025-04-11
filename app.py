import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import uuid

from database import init_db, create_user, create_shelf, add_shelf_image, add_shelf_objects, get_shelf_objects, update_shelf_object, publish_shelf, get_shelf
from vision import process_shelf_image

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize database
init_db()

# Create a test user for MVP
test_user_id = create_user('testuser', 'test@example.com')

@app.route('/')
def index():
    return render_template('index.html', user_id=test_user_id)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
            
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            # Save the uploaded file
            filename = secure_filename(str(uuid.uuid4()) + os.path.splitext(file.filename)[1])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Create relative URL for the image
            image_url = '/uploads/' + filename
            
            # Create a new shelf
            shelf_id = create_shelf(test_user_id)
            
            # Add image to shelf
            add_shelf_image(shelf_id, image_url)
            
            # Process the image (in a real app, this might be async)
            objects = process_shelf_image(image_url)
            
            # Add detected objects to database
            add_shelf_objects(shelf_id, objects)
            
            # Redirect to edit page
            return redirect(url_for('edit_shelf', shelf_id=shelf_id))
    
    return render_template('upload.html')

@app.route('/api/shelf/ingest', methods=['POST'])
def api_ingest():
    data = request.json
    user_id = data.get('user_id')
    image_urls = data.get('image_urls', [])
    
    if not user_id or not image_urls:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Create a new shelf
    shelf_id = create_shelf(user_id)
    
    # Process only the first image for simplicity in MVP
    image_url = image_urls[0]
    add_shelf_image(shelf_id, image_url)
    
    # Process the image
    objects = process_shelf_image(image_url)
    
    # Add objects to database
    add_shelf_objects(shelf_id, objects)
    
    # Prepare response
    response_objects = []
    db_objects = get_shelf_objects(shelf_id)
    
    for obj in db_objects:
        response_obj = {
            "object_id": obj['id'],
            "type": obj['type'],
            "bounding_box": obj['bounding_box'],
            "status": obj['status'],
            "confidence": obj['confidence']
        }
        
        if obj['type'] == 'book':
            response_obj.update({
                "title": obj['title'],
                "author": obj['author'],
                "isbn": obj['isbn']
            })
        else:
            response_obj.update({
                "label": obj['label']
            })
            
        response_objects.append(response_obj)
    
    return jsonify({
        "shelf_id": shelf_id,
        "objects": response_objects
    })

@app.route('/edit/<shelf_id>', methods=['GET'])
def edit_shelf(shelf_id):
    shelf = get_shelf(shelf_id)
    if not shelf:
        return "Shelf not found", 404
    
    return render_template('edit_shelf.html', shelf=shelf)

@app.route('/api/shelf/update', methods=['POST'])
def api_update():
    data = request.json
    shelf_id = data.get('shelf_id')
    objects = data.get('objects', [])
    publish = data.get('publish', False)
    
    if not shelf_id:
        return jsonify({"error": "Missing shelf_id"}), 400
    
    # Update each object
    for obj in objects:
        object_id = obj.pop('object_id')
        update_shelf_object(object_id, obj)
    
    # Publish if requested
    if publish:
        publish_shelf(shelf_id)
    
    return jsonify({
        "status": "success",
        "shelf_id": shelf_id
    })

@app.route('/view/<shelf_id>')
def view_shelf(shelf_id):
    shelf = get_shelf(shelf_id)
    if not shelf:
        return "Shelf not found", 404
    
    return render_template('view_shelf.html', shelf=shelf)

@app.route('/api/shelf/view/<shelf_id>')
def api_view_shelf(shelf_id):
    shelf = get_shelf(shelf_id)
    if not shelf:
        return jsonify({"error": "Shelf not found"}), 404
    
    # Format response according to API spec
    response = {
        "shelf_id": shelf['id'],
        "user_id": shelf['user_id'],
        "objects": [],
        "created_at": shelf['created_at'],
        "updated_at": shelf['updated_at']
    }
    
    for obj in shelf['objects']:
        response_obj = {
            "object_id": obj['id'],
            "type": obj['type'],
            "bounding_box": obj['bounding_box'],
            "tags": obj['tags'],
            "notes": obj['note']
        }
        
        if obj['type'] == 'book':
            response_obj.update({
                "title": obj['title'],
                "author": obj['author'],
                "isbn": obj['isbn']
            })
        else:
            response_obj.update({
                "label": obj['label']
            })
            
        response['objects'].append(response_obj)
    
    return jsonify(response)

# Serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)