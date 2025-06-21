import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import os # For __main__ example

# --- Global Variables for Model and Processor ---
FLORENCE_MODEL = None
FLORENCE_PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = 'microsoft/Florence-2-base' # Using base as it's generally smaller/faster

# --- Initialization Function ---
def initialize_florence_model() -> bool:
    """
    Initializes the Florence-2 model and processor.
    Loads them into global variables FLORENCE_MODEL and FLORENCE_PROCESSOR.

    Returns:
        bool: True if initialization was successful, False otherwise.
    """
    global FLORENCE_MODEL, FLORENCE_PROCESSOR

    if FLORENCE_MODEL is not None and FLORENCE_PROCESSOR is not None:
        print("Florence-2 model and processor already initialized.")
        return True

    print(f"Initializing Florence-2 model ('{MODEL_ID}') on device: {DEVICE}...")
    try:
        # Note: The original script from PRITHIVSAKTHIUR/Image-Captioning-Florence2
        # included a subprocess call to attempt `pip install flash-attn --no-build-isolation`.
        # flash-attn can provide significant speedups on compatible hardware (NVIDIA GPUs).
        # For cleaner integration and to avoid side-effects from a utility module,
        # that automatic installation is omitted here.
        # Users wanting to leverage flash-attn should install it manually in their environment:
        # e.g., pip install flash-attn --no-build-isolation (check flash-attn docs for current guidance)
        # If installed and compatible, Transformers should automatically attempt to use it.

        FLORENCE_MODEL = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()
        FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("Florence-2 model and processor loaded successfully.")
        return True
    except ImportError as e:
        print(f"Error: Missing dependencies for Florence-2. Please ensure 'transformers' and 'torch' are installed. Details: {e}")
        FLORENCE_MODEL = None
        FLORENCE_PROCESSOR = None
        return False
    except Exception as e:
        print(f"Error loading Florence-2 model or processor: {e}")
        FLORENCE_MODEL = None
        FLORENCE_PROCESSOR = None
        return False

# --- Captioning Function ---
def get_caption_for_image(image: Image.Image, task_prompt: str = "<MORE_DETAILED_CAPTION>") -> str:
    """
    Generates a caption for the given PIL Image using the loaded Florence-2 model.

    Args:
        image (PIL.Image.Image): The image to describe.
        task_prompt (str): The task prompt for Florence-2 (e.g., "<MORE_DETAILED_CAPTION>", "<CAPTION>", "<OD>")

    Returns:
        str: The generated caption, or an error message if captioning fails or model is not loaded.
    """
    if FLORENCE_MODEL is None or FLORENCE_PROCESSOR is None:
        error_msg = "Error: Florence-2 model is not initialized. Please call initialize_florence_model() first."
        print(error_msg)
        return error_msg

    if not isinstance(image, Image.Image):
        try:
            # Attempt to convert if common array type, though PIL Image is expected.
            image = Image.fromarray(image)
        except Exception as e:
            return f"Error: Input is not a valid PIL Image and could not be converted. Details: {e}"

    print(f"Generating caption for image with task: {task_prompt}...")
    try:
        inputs = FLORENCE_PROCESSOR(text=task_prompt, images=image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            generated_ids = FLORENCE_MODEL.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False, # As per original script
                do_sample=False,      # As per original script
                num_beams=3,          # As per original script
            )

        # Decode and post-process
        # Using skip_special_tokens=False as in the reference app.py for generate,
        # but True might be more common for final captions. Let's stick to reference.
        generated_text = FLORENCE_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # The post_process_generation method is crucial for Florence-2
        processed_output = FLORENCE_PROCESSOR.post_process_generation(
            generated_text,
            task=task_prompt, # Use the same task prompt as input
            image_size=(image.width, image.height)
        )
        # The output is a dict, and the key is the task prompt itself
        caption = processed_output.get(task_prompt, "Error: Caption not found in processed output.")

        print(f"Caption generated: {caption[:100]}...") # Print start of caption
        return caption
    except Exception as e:
        error_msg = f"Error during Florence-2 image captioning: {e}"
        print(error_msg)
        return error_msg

# --- Main block for testing ---
if __name__ == "__main__":
    print("--- Testing Florence-2 Captioner Utility ---")

    # 1. Initialize model
    if not initialize_florence_model():
        print("Florence-2 model initialization failed. Exiting test.")
        exit()

    # 2. Load a sample image
    #    For this test, we'll try to use the 'static/preview.jpg' if it exists.
    #    Otherwise, users running this directly would need to provide their own image path.
    sample_image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "preview.jpg")

    if not os.path.exists(sample_image_path):
        print(f"Sample image not found at '{sample_image_path}'.")
        print("Please create a 'static/preview.jpg' or modify the path in this test script.")
        # Attempt to create a dummy image for basic pipeline testing if no image found
        try:
            print("Creating a dummy red 100x100 image for testing...")
            dummy_image = Image.new('RGB', (100, 100), color = 'red')
            img_to_caption = dummy_image
            print("Using dummy image for captioning test.")
        except Exception as e:
            print(f"Failed to create dummy image: {e}. Exiting test.")
            exit()
    else:
        try:
            img_to_caption = Image.open(sample_image_path)
            print(f"Successfully loaded sample image from '{sample_image_path}'.")
        except Exception as e:
            print(f"Error loading sample image '{sample_image_path}': {e}. Exiting test.")
            exit()

    # 3. Get caption for the image
    print(f"\nAttempting to get caption for the image using task: <MORE_DETAILED_CAPTION>")
    caption_detailed = get_caption_for_image(img_to_caption, task_prompt="<MORE_DETAILED_CAPTION>")
    print(f"\n<MORE_DETAILED_CAPTION> result:\n{caption_detailed}")

    # Test with a different task prompt, e.g., <CAPTION> for shorter caption
    # Or <OD> for object detection like results
    # print(f"\nAttempting to get caption for the image using task: <CAPTION>")
    # caption_simple = get_caption_for_image(img_to_caption, task_prompt="<CAPTION>")
    # print(f"\n<CAPTION> result:\n{caption_simple}")

    # print(f"\nAttempting to get object detection like results using task: <OD>")
    # od_results = get_caption_for_image(img_to_caption, task_prompt="<OD>")
    # print(f"\n<OD> result:\n{od_results}")

    print("\n--- Florence-2 Captioner Utility Test Complete ---")
