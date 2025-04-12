import sqlite3
import uuid
import json
from datetime import datetime
import os
import logging # Import logging

logger = logging.getLogger(__name__) # Setup logger for this module

# Ensure the instance directory exists
if not os.path.exists('instance'):
    os.makedirs('instance')

def get_db():
    db = sqlite3.connect('instance/bookshelf.db')
    db.row_factory = sqlite3.Row
    return db

def init_db():
    # Initialize DB from schema file
    try:
        with open('schema.sql', 'r') as f:
            schema = f.read()
        db = get_db()
        db.executescript(schema)
        db.commit()
        db.close()
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {e}")
        # Optionally raise the exception or handle it more gracefully
        raise

def create_user(username, email=None):
    user_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    db = get_db()
    db.execute(
        'INSERT INTO users (id, username, email, created_at) VALUES (?, ?, ?, ?)',
        (user_id, username, email, now)
    )
    db.commit()
    db.close()
    
    return user_id

def create_shelf(user_id):
    shelf_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    db = get_db()
    db.execute(
        'INSERT INTO shelves (id, user_id, status, created_at, updated_at) VALUES (?, ?, ?, ?, ?)',
        (shelf_id, user_id, 'draft', now, now)
    )
    db.commit()
    db.close()
    
    return shelf_id

# Function to add a shelf collection
# def add_shelf_collection(...): # Commented out as it's unused now
#    ...

def add_shelf_image(shelf_id, image_url, processed_width=None, processed_height=None): # Added width/height params
    image_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    db = get_db()
    db.execute(
        # Added processed_width and processed_height columns
        'INSERT INTO shelf_images (id, shelf_id, image_url, processed_width, processed_height, uploaded_at) VALUES (?, ?, ?, ?, ?, ?)',
        (image_id, shelf_id, image_url, processed_width, processed_height, now)
    )
    db.commit()
    db.close()
    
    return image_id

# Modified to accept collection_id
def add_shelf_object(shelf_id, collection_id, obj_data):
    db = get_db()
    now = datetime.now().isoformat()
    object_id = str(uuid.uuid4())

    # Extract data from the new pipeline's output structure
    obj_type = obj_data.get('yolo_type', 'misc')
    bounding_box_json = json.dumps(obj_data['bounding_box'])
    status = 'auto_detected'
    confidence = obj_data.get('yolo_confidence', 0.0)
    # Get GPT analysis details
    title = obj_data.get('title', None)
    author = obj_data.get('author', None)
    label = obj_data.get('label', None)
    gpt_confidence = obj_data.get('gpt_confidence') # String: 'confident', 'meh', 'no clue'
    needs_agentic_search = obj_data.get('needs_agentic_search', False) # Boolean
    needs_agentic_search_int = 1 if needs_agentic_search else 0 # Convert to integer for DB

    # Common fields for insertion
    common_fields = '''id, shelf_id, collection_id, type, bounding_box, status, confidence,
                       gpt_confidence, needs_agentic_search, created_at, updated_at'''
    common_values = (
        object_id, shelf_id, collection_id, obj_type, bounding_box_json, status,
        confidence, gpt_confidence, needs_agentic_search_int, now, now
    )

    # Simplified logic: Insert common fields + type-specific fields if they exist
    # The CHECK constraint in the schema now validates the 'type' field.
    fields_to_insert = list(common_fields.split(', '))
    values_to_insert = list(common_values)

    if title:
        fields_to_insert.append('title')
        values_to_insert.append(title)
    if author:
        fields_to_insert.append('author')
        values_to_insert.append(author)
    # We don't get ISBN from the current pipeline, but keep the column
    # fields_to_insert.append('isbn')
    # values_to_insert.append(None)
    if label:
        fields_to_insert.append('label')
        values_to_insert.append(label)

    # Construct the SQL query dynamically
    fields_str = ', '.join(fields_to_insert)
    placeholders = ', '.join(['?'] * len(values_to_insert))
    sql = f'INSERT INTO shelf_objects ({fields_str}) VALUES ({placeholders})'

    try:
        db.execute(sql, tuple(values_to_insert))
        db.commit()
        logger.debug(f"Successfully added object {object_id} of type {obj_type} to DB.")
    except Exception as e:
        db.rollback() # Rollback on error
        logger.error(f"Error executing insert for object {object_id}: {e}", exc_info=True)
        logger.error(f"SQL: {sql}")
        logger.error(f"Values: {values_to_insert}")
        # Re-raise the exception to be handled by the calling function (in app.py)
        raise e
    finally:
        db.close()

    return object_id

def get_shelf_objects(shelf_id):
    db = get_db()
    # Select objects, join with annotations, order by position
    objects = db.execute(
        '''SELECT so.*, oa.note, oa.tags
           FROM shelf_objects so
           LEFT JOIN object_annotations oa ON so.id = oa.object_id
           WHERE so.shelf_id = ?
           ORDER BY CAST(json_extract(so.bounding_box, '$[1]') AS REAL), -- Order primarily by vertical position (y1)
                    CAST(json_extract(so.bounding_box, '$[0]') AS REAL)  -- Then by horizontal position (x1)
           ''',
        (shelf_id,)
    ).fetchall()

    result = []
    for obj in objects:
        obj_dict = dict(obj)
        obj_dict['bounding_box'] = json.loads(obj_dict['bounding_box'])
        # Handle annotations (note and tags)
        obj_dict['note'] = obj_dict.get('note', '') # Default to empty string if NULL
        # Parse tags JSON, default to empty list if NULL or invalid JSON
        tags_json = obj_dict.get('tags')
        try:
            obj_dict['tags'] = json.loads(tags_json) if tags_json else []
        except json.JSONDecodeError:
            logger.warning(f"Could not decode tags JSON for object {obj_dict['id']}: {tags_json}")
            obj_dict['tags'] = []
        # Ensure needs_agentic_search is boolean for consistency upstream, although template handles 0/1 okay
        # Convert integer 0/1 back to boolean True/False
        obj_dict['needs_agentic_search'] = bool(obj_dict.get('needs_agentic_search', 0))
        result.append(obj_dict)

    db.close()
    return result

# Update shelf object details, including logging changes and handling annotations
def update_shelf_object(object_id, updates):
    db = get_db()
    now = datetime.now().isoformat()
    
    # Get current object data for logging changes
    current = db.execute('SELECT * FROM shelf_objects WHERE id = ?', (object_id,)).fetchone()
    
    if not current:
        print(f"Warning: Object with ID {object_id} not found for update.")
        db.close()
        return

    # Update fields
    update_query_parts = []
    update_values = []
    
    for field, value in updates.items():
        if field in ['notes', 'tags', 'object_id']: # Exclude non-column fields and ID
            continue  
            
        # Check if field exists in the table (simple check, might need improvement for robustness)
        if field in current.keys():
            # Only log and update if the value has actually changed
            current_value_str = str(current[field])
            new_value_str = str(value)
            if current_value_str != new_value_str:
                # Log the change
                db.execute(
                    'INSERT INTO correction_logs (id, object_id, field, old_value, new_value, timestamp) VALUES (?, ?, ?, ?, ?, ?)',
                    (str(uuid.uuid4()), object_id, field, current_value_str, new_value_str, now)
                )
                
                # Prepare part of the UPDATE statement
                update_query_parts.append(f"{field} = ?")
                update_values.append(value)
        else:
            print(f"Warning: Field '{field}' not found in shelf_objects table, skipping update for this field.")

    # Update the object if there are changes
    if update_query_parts:
        update_query_parts.append("updated_at = ?")
        update_values.append(now)
        update_values.append(object_id)
        
        update_sql = f"UPDATE shelf_objects SET {', '.join(update_query_parts)} WHERE id = ?"
        db.execute(update_sql, tuple(update_values))
    
    # Handle annotations
    if 'notes' in updates or 'tags' in updates:
        notes = updates.get('notes', '')
        # Ensure tags are stored as a JSON string
        tags_list = updates.get('tags', [])
        if not isinstance(tags_list, list):
            tags_list = [] # Default to empty list if invalid format
        tags = json.dumps(tags_list)
        
        # Check if annotation exists
        existing = db.execute('SELECT id FROM object_annotations WHERE object_id = ?', (object_id,)).fetchone()
        
        if existing:
            db.execute(
                'UPDATE object_annotations SET note = ?, tags = ? WHERE object_id = ?',
                (notes, tags, object_id)
            )
        else:
            db.execute(
                'INSERT INTO object_annotations (id, object_id, note, tags, created_at) VALUES (?, ?, ?, ?, ?)',
                (str(uuid.uuid4()), object_id, notes, tags, now)
            )
    
    db.commit()
    db.close()

# Function to update collection details (e.g., label)
# def update_shelf_collection(...):
#    ...

def publish_shelf(shelf_id):
    db = get_db()
    now = datetime.now().isoformat()
    
    db.execute(
        'UPDATE shelves SET status = ?, updated_at = ? WHERE id = ?',
        ('published', now, shelf_id)
    )
    
    db.commit()
    db.close()

def get_shelf(shelf_id):
    """Fetches shelf details and all associated objects (flat list)."""
    db = get_db()
    shelf_data = db.execute(
        # Select processed dimensions along with image URL
        '''SELECT s.*, 
           (SELECT json_group_array(json_object('id', i.id, 'image_url', i.image_url, 'processed_width', i.processed_width, 'processed_height', i.processed_height)) 
            FROM shelf_images i WHERE i.shelf_id = s.id) as images 
           FROM shelves s 
           WHERE s.id = ?''',
        (shelf_id,)
    ).fetchone()
    
    if not shelf_data:
        db.close()
        return None
        
    shelf_dict = dict(shelf_data)
    # Ensure images is parsed correctly and handle potential null dimensions
    try:
        shelf_dict['images'] = json.loads(shelf_dict['images']) if shelf_dict['images'] else []
        # Add default dimensions if they are missing in the DB for some reason
        for img in shelf_dict['images']:
            img.setdefault('processed_width', None)
            img.setdefault('processed_height', None)
    except (json.JSONDecodeError, TypeError):
         logger.error(f"Failed to parse images JSON for shelf {shelf_id}", exc_info=True)
         shelf_dict['images'] = [] # Default to empty list on error

    # Fetch all objects for this shelf using the existing function
    shelf_objects = get_shelf_objects(shelf_id) 
    shelf_dict['objects'] = shelf_objects # Add objects as a flat list
            
    db.close()
    logger.debug(f"get_shelf returning data for {shelf_id} with {len(shelf_objects)} objects.")
    return shelf_dict