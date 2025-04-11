"""
Simplified mock of vision processing for MVP.
In a production system, this would integrate with a real computer vision API.
"""

def process_shelf_image(image_url):
    """
    Mock vision processing that returns hardcoded detected objects
    """
    # In a real implementation, this would call a vision API
    # For MVP, we'll return mock data
    
    # Simulate detection with hardcoded sample data
    objects = [
        {
            "bounding_box": [100, 50, 200, 300],
            "type": "book",
            "title": "Dune",
            "author": "Frank Herbert",
            "isbn": "0441172717",
            "confidence": 0.92
        },
        {
            "bounding_box": [210, 50, 280, 300],
            "type": "book",
            "title": "1984",
            "author": "George Orwell",
            "isbn": "0451524934",
            "confidence": 0.89
        },
        {
            "bounding_box": [290, 80, 350, 250],
            "type": "misc",
            "label": "Small plant",
            "confidence": 0.78
        },
        {
            "bounding_box": [360, 50, 450, 300],
            "type": "book",
            "title": "The Great Gatsby",
            "author": "F. Scott Fitzgerald",
            "isbn": "9780743273565",
            "confidence": 0.85
        },
        {
            "bounding_box": [460, 100, 520, 270],
            "type": "misc",
            "label": "Coffee mug",
            "confidence": 0.82
        }
    ]
    
    return objects