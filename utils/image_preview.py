import os

# Assuming the static directory is at the root of the project,
# and this file is in utils/
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
PLACEHOLDER_IMAGE_NAME = "preview.jpg"
PLACEHOLDER_IMAGE_PATH = os.path.join(STATIC_DIR, PLACEHOLDER_IMAGE_NAME)

def get_placeholder_image_path() -> str:
    """
    Returns the absolute path to the placeholder preview image.

    Checks if the placeholder image exists. If not, it prints a warning
    but still returns the expected path, allowing UI components to link to it.
    The actual image file should be present at static/preview.jpg.
    """
    if not os.path.exists(PLACEHOLDER_IMAGE_PATH):
        print(f"Warning: Placeholder image not found at {PLACEHOLDER_IMAGE_PATH}")
        print("Please ensure 'static/preview.jpg' exists.")
    return PLACEHOLDER_IMAGE_PATH

if __name__ == "__main__":
    path = get_placeholder_image_path()
    print(f"Placeholder image path: {path}")

    # Verify if the directory and file exist (optional check)
    if os.path.exists(path):
        print("Placeholder image file exists.")
    else:
        print("Placeholder image file does NOT exist. Make sure 'static/preview.jpg' is present.")

    # Check if static directory exists
    if os.path.isdir(STATIC_DIR):
        print(f"Static directory '{STATIC_DIR}' exists.")
    else:
        print(f"Static directory '{STATIC_DIR}' does NOT exist.")

    # Check if static/preview.jpg exists by listing files
    expected_image_file = os.path.join(STATIC_DIR, PLACEHOLDER_IMAGE_NAME)
    if os.path.isfile(expected_image_file):
        print(f"Verified: '{expected_image_file}' is a file.")
    else:
        print(f"Warning: '{expected_image_file}' is not a file or does not exist.")
