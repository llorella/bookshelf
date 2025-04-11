import sqlite3
import uuid
import json
from datetime import datetime
import os

# Ensure the instance directory exists
if not os.path.exists('instance'):
    os.makedirs('instance')

def get_db():
    db = sqlite3.connect('instance/bookshelf.db')
    db.row_factory = sqlite3.Row
    return db

def init_db():
    with open('schema.sql', 'r') as f:
        schema = f.read()
    
    db = get_db()
    db.executescript(schema)
    db.commit()
    db.close()

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

def add_shelf_image(shelf_id, image_url):
    image_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    db = get_db()
    db.execute(
        'INSERT INTO shelf_images (id, shelf_id, image_url, uploaded_at) VALUES (?, ?, ?, ?)',
        (image_id, shelf_id, image_url, now)
    )
    db.commit()
    db.close()
    
    return image_id

def add_shelf_objects(shelf_id, objects):
    db = get_db()
    now = datetime.now().isoformat()
    
    for obj in objects:
        object_id = str(uuid.uuid4())
        
        if obj['type'] == 'book':
            db.execute(
                '''INSERT INTO shelf_objects 
                (id, shelf_id, type, bounding_box, status, title, author, isbn, confidence, created_at, updated_at) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    object_id, 
                    shelf_id, 
                    obj['type'], 
                    json.dumps(obj['bounding_box']), 
                    'auto_detected',
                    obj.get('title', ''),
                    obj.get('author', ''),
                    obj.get('isbn', ''),
                    obj.get('confidence', 0.0),
                    now,
                    now
                )
            )
        else:  # misc
            db.execute(
                '''INSERT INTO shelf_objects 
                (id, shelf_id, type, bounding_box, status, label, confidence, created_at, updated_at) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    object_id, 
                    shelf_id, 
                    obj['type'], 
                    json.dumps(obj['bounding_box']), 
                    'auto_detected',
                    obj.get('label', ''),
                    obj.get('confidence', 0.0),
                    now,
                    now
                )
            )
        
    db.commit()
    db.close()

def get_shelf_objects(shelf_id):
    db = get_db()
    objects = db.execute(
        'SELECT * FROM shelf_objects WHERE shelf_id = ?',
        (shelf_id,)
    ).fetchall()
    
    result = []
    for obj in objects:
        obj_dict = dict(obj)
        obj_dict['bounding_box'] = json.loads(obj_dict['bounding_box'])
        result.append(obj_dict)
    
    db.close()
    return result

def update_shelf_object(object_id, updates):
    db = get_db()
    now = datetime.now().isoformat()
    
    # Get current object data for logging changes
    current = db.execute('SELECT * FROM shelf_objects WHERE id = ?', (object_id,)).fetchone()
    
    # Update fields
    for field, value in updates.items():
        if field in ['notes', 'tags']:
            continue  # Handle annotations separately
            
        if field in current and current[field] != value:
            # Log the change
            db.execute(
                'INSERT INTO correction_logs (id, object_id, field, old_value, new_value, timestamp) VALUES (?, ?, ?, ?, ?, ?)',
                (str(uuid.uuid4()), object_id, field, str(current[field]), str(value), now)
            )
            
            # Update the field
            db.execute(
                f'UPDATE shelf_objects SET {field} = ?, updated_at = ? WHERE id = ?',
                (value, now, object_id)
            )
    
    # Handle annotations
    if 'notes' in updates or 'tags' in updates:
        notes = updates.get('notes', '')
        tags = json.dumps(updates.get('tags', []))
        
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
    db = get_db()
    
    shelf = db.execute('SELECT * FROM shelves WHERE id = ?', (shelf_id,)).fetchone()
    if not shelf:
        db.close()
        return None
    
    result = dict(shelf)
    
    # Get objects
    objects = db.execute('''
        SELECT o.*, a.note, a.tags 
        FROM shelf_objects o
        LEFT JOIN object_annotations a ON o.id = a.object_id
        WHERE o.shelf_id = ?
    ''', (shelf_id,)).fetchall()
    
    result['objects'] = []
    for obj in objects:
        obj_dict = dict(obj)
        obj_dict['bounding_box'] = json.loads(obj_dict['bounding_box'])
        if obj_dict['tags']:
            obj_dict['tags'] = json.loads(obj_dict['tags'])
        else:
            obj_dict['tags'] = []
        result['objects'].append(obj_dict)
    
    # Get image
    image = db.execute('SELECT image_url FROM shelf_images WHERE shelf_id = ?', (shelf_id,)).fetchone()
    if image:
        result['image_url'] = image['image_url']
    else:
        result['image_url'] = None
    
    db.close()
    return result