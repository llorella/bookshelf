DROP TABLE IF EXISTS object_annotations;
DROP TABLE IF EXISTS correction_logs;
DROP TABLE IF EXISTS shelf_objects;
DROP TABLE IF EXISTS shelf_images;
DROP TABLE IF EXISTS shelf_collections;
DROP TABLE IF EXISTS shelves;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    created_at TEXT NOT NULL
);

CREATE TABLE shelves (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE shelf_collections (
    id TEXT PRIMARY KEY,
    shelf_id TEXT NOT NULL,
    bounding_box TEXT NOT NULL,
    label TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (shelf_id) REFERENCES shelves (id)
);

CREATE TABLE shelf_images (
    id TEXT PRIMARY KEY,
    shelf_id TEXT NOT NULL,
    image_url TEXT NOT NULL,
    processed_width INTEGER,
    processed_height INTEGER,
    uploaded_at TEXT NOT NULL,
    FOREIGN KEY (shelf_id) REFERENCES shelves (id)
);

CREATE TABLE shelf_objects (
    id TEXT PRIMARY KEY,
    shelf_id TEXT NOT NULL,
    collection_id TEXT,
    type TEXT NOT NULL CHECK(type IN ('book', 'misc', 'vase', 'clock', 'laptop', 'cell phone', 'remote', 'mouse', 'keyboard', 'teddy bear', 'bottle', 'cup', 'chair', 'potted plant')),
    bounding_box TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'auto_detected',
    title TEXT,
    author TEXT,
    isbn TEXT,
    label TEXT,
    confidence REAL,
    gpt_confidence TEXT,
    needs_agentic_search INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (shelf_id) REFERENCES shelves (id),
    FOREIGN KEY (collection_id) REFERENCES shelf_collections (id)
);

CREATE TABLE correction_logs (
    id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (object_id) REFERENCES shelf_objects (id)
);

CREATE TABLE object_annotations (
    id TEXT PRIMARY KEY,
    object_id TEXT NOT NULL UNIQUE,
    note TEXT,
    tags TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (object_id) REFERENCES shelf_objects (id)
);

CREATE INDEX idx_shelves_user_id ON shelves (user_id);
CREATE INDEX idx_shelf_collections_shelf_id ON shelf_collections (shelf_id);
CREATE INDEX idx_shelf_images_shelf_id ON shelf_images (shelf_id);
CREATE INDEX idx_shelf_objects_shelf_id ON shelf_objects (shelf_id);
CREATE INDEX idx_shelf_objects_collection_id ON shelf_objects (collection_id);
CREATE INDEX idx_correction_logs_object_id ON correction_logs (object_id);
CREATE INDEX idx_object_annotations_object_id ON object_annotations (object_id);