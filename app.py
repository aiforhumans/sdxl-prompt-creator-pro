import gradio as gr
from composer.composer import PromptComposer
from utils.lmstudio_client import LMStudioClient
from utils.image_preview import get_placeholder_image_path
from models.prompt_schema import CinematicPrompt # To use for type hinting and structure
import os

# --- Application Setup ---
# Initialize LMStudioClient (points to http://localhost:1234/v1 by default)
# The README instructs to have LM Studio running at this address.
try:
    lm_client = LMStudioClient()
    # Perform a quick test call to see if LM Studio is available
    # This is a simple way to check; a more robust app might have a status indicator
    # Test with a very short, simple prompt
    test_response = lm_client.generate_text("System: You are a health check assistant.", "User: Ping.", max_tokens=10)
    if "Error:" in test_response:
        print(f"Warning: LMStudioClient test call failed. {test_response}")
        print("Please ensure LM Studio is running at http://localhost:1234/v1 and a model is loaded.")
        # Fallback or error state could be handled here. For now, app will run but composer may fail.
    else:
        print("LMStudioClient initialized and test call successful.")
except Exception as e:
    print(f"Critical Error: Could not initialize LMStudioClient: {e}")
    print("The application may not function correctly. Ensure LM Studio is running.")
    # In a production app, you might exit or provide a non-functional UI state.
    # For this example, we'll let it proceed so Gradio UI can still load.
    lm_client = None

# Instantiate PromptComposer with the client
if lm_client:
    prompt_composer = PromptComposer(lm_client=lm_client)
else:
    # Provide a non-functional composer if client failed, so Gradio can still launch
    class MockComposer:
        def compose_prompt(self, character_name: str) -> CinematicPrompt:
            # Return a dummy prompt object with error messages
            error_prompt = CinematicPrompt(
                character=dict(description=f"Error: LM Studio client not available for {character_name}"),
                environment=dict(setting="Error state"),
                camera=dict(shot_type="Error state"),
                lighting=dict(description="Error state"),
                style=dict(genre="Error state"),
                subject_focus="Error",
                ambiance_atmosphere="Error",
            )
            error_prompt.final_prompt_string = "Error: LM Studio client not available. Cannot generate prompt."
            return error_prompt
    prompt_composer = MockComposer()
    print("Using MockComposer as LMStudioClient initialization failed.")


# Get the placeholder image path
placeholder_image = get_placeholder_image_path()
if not os.path.exists(placeholder_image):
    print(f"Warning: Placeholder image not found at {placeholder_image}")
    # Gradio might show a broken image link if this file is missing.

# --- Gradio Interface Function ---
def generate_cinematic_prompt(character_name: str, enable_comfyui_preview: bool):
    """
    Main function for Gradio to generate prompt and handle preview.
    """
    if not character_name.strip():
        return "Please enter a character name.", {}, placeholder_image, "Character name cannot be empty."

    print(f"Generating prompt for character: {character_name}")

    try:
        structured_prompt: CinematicPrompt = prompt_composer.compose_prompt(character_name)
        prompt_string = structured_prompt.final_prompt_string
        prompt_json = structured_prompt.model_dump()

        # For now, ComfyUI preview is just the placeholder.
        # Later, this is where comfyui_integration.generate_image would be called if enable_comfyui_preview is True.
        current_image_preview = placeholder_image

        if enable_comfyui_preview:
            # This is where you would call the actual ComfyUI integration
            # from comfyui_integration.generate_image import generate_image_preview_comfy
            # current_image_preview = generate_image_preview_comfy(prompt_string)
            # This is where you would call the actual ComfyUI integration
            from comfyui_integration.generate_image import generate_image_preview_comfy # Import the stub

            print("ComfyUI preview checkbox is enabled. Calling ComfyUI integration stub...")
            # Call the stub, which currently returns the placeholder path
            current_image_preview = generate_image_preview_comfy(prompt_string if (prompt_string and "Error:" not in prompt_string) else "Test prompt for ComfyUI stub if main prompt failed")
            status_message = "Prompt generated. ComfyUI preview stub called (returns placeholder)."
        else:
            status_message = "Prompt generated. ComfyUI preview disabled."

        print(f"Generated prompt string: {prompt_string[:100]}...")
        return prompt_string, prompt_json, current_image_preview, status_message

    except Exception as e:
        print(f"Error during prompt generation: {e}")
        error_message = f"An error occurred: {e}"
        return ("Error generating prompt. Check console for details.",
                {"error": str(e)},
                placeholder_image,
                error_message)

# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# SDXL Prompt Creator Studio PRO")
    gr.Markdown("Generate cinematic prompts for Stable Diffusion XL from a character name using AI modules via LM Studio.")

    with gr.Row():
        with gr.Column(scale=2):
            character_name_input = gr.Textbox(
                label="Character Name",
                placeholder="e.g., 'A stoic space marine', 'A whimsical forest fairy', 'Gandalf the Grey'"
            )
            enable_comfyui_checkbox = gr.Checkbox(
                label="Enable ComfyUI Preview (Placeholder)",
                value=False # Default to False
            )
            generate_button = gr.Button("🎨 Generate Cinematic Prompt", variant="primary")

            status_output = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Column(scale=3):
            gr.Markdown("### 🖼️ Preview")
            image_output = gr.Image(label="Generated Image Preview", value=placeholder_image, type="filepath", interactive=False, height=300) # Use filepath for local images

    gr.Markdown("---")

    with gr.Accordion("📝 Generated Prompt Details", open=True):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("#### Full Prompt String (for SDXL)")
                prompt_string_output = gr.Textbox(label="SDXL Prompt", lines=10, interactive=False, show_copy_button=True)
            with gr.Column(scale=1):
                gr.Markdown("#### Structured Prompt (JSON)")
                prompt_json_output = gr.JSON(label="Structured Prompt Data",) # Removed interactive=False, JSON is not directly interactive

    # --- Event Handling ---
    generate_button.click(
        fn=generate_cinematic_prompt,
        inputs=[character_name_input, enable_comfyui_checkbox],
        outputs=[prompt_string_output, prompt_json_output, image_output, status_output]
    )

    gr.Markdown(
        """
        **Note:**
        - Ensure LM Studio is running at `http://localhost:1234/v1` with a model loaded.
        - ComfyUI preview is currently a placeholder. Actual integration is a future step.
        """
    )

if __name__ == "__main__":
    print("Launching Gradio App...")
    # Ensure the placeholder image path is correct before launching
    print(f"Using placeholder image from: {placeholder_image}")
    if not os.path.exists(placeholder_image):
        print(f"CRITICAL WARNING: Placeholder image {placeholder_image} not found. The image preview will be broken.")
        # You could create a dummy placeholder if it's missing, e.g., a small black square.
        # For now, we proceed.

    demo.launch()
    print("Gradio App launched. Access it in your browser (usually http://127.0.0.1:7860 or http://localhost:7860).")
