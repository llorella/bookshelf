DROP TABLE IF EXISTS users;
DROP TABLE IF EXISTS shelves;
DROP TABLE IF EXISTS shelf_images;
DROP TABLE IF EXISTS shelf_objects;
DROP TABLE IF EXISTS object_annotations;
DROP TABLE IF EXISTS correction_logs;

CREATE TABLE users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE,
    created_at TEXT
);

CREATE TABLE shelves (
    id TEXT PRIMARY KEY,
    user_id TEXT REFERENCES users(id),
    status TEXT CHECK(status IN ('draft', 'published')),
    title TEXT DEFAULT 'Untitled Shelf',
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE shelf_images (
    id TEXT PRIMARY KEY,
    shelf_id TEXT REFERENCES shelves(id),
    image_url TEXT NOT NULL,
    uploaded_at TEXT
);

CREATE TABLE shelf_objects (
    id TEXT PRIMARY KEY,
    shelf_id TEXT REFERENCES shelves(id),
    type TEXT CHECK(type IN ('book', 'misc')),
    bounding_box TEXT, -- JSON string
    status TEXT CHECK(status IN ('auto_detected', 'confirmed', 'edited', 'unknown')),
    title TEXT,
    author TEXT,
    isbn TEXT,
    label TEXT, -- for misc objects
    confidence REAL,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE object_annotations (
    id TEXT PRIMARY KEY,
    object_id TEXT REFERENCES shelf_objects(id),
    note TEXT,
    tags TEXT, -- JSON string array
    created_at TEXT
);

CREATE TABLE correction_logs (
    id TEXT PRIMARY KEY,
    object_id TEXT REFERENCES shelf_objects(id),
    field TEXT,
    old_value TEXT,
    new_value TEXT,
    reason TEXT,
    timestamp TEXT
);