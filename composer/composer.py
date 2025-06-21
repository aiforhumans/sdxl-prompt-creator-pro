from models.prompt_schema import (
    CinematicPrompt,
    CharacterDetails,
    EnvironmentDetails,
    CameraSetup,
    LightingDetails,
    ArtisticStyle,
)
from utils.lmstudio_client import LMStudioClient


import json
import os
from typing import Optional, List, Dict, Any

class PromptComposer:
    def __init__(self, lm_client, traits_file_path: str = "data/character_traits.json"):
        self.lm_client = lm_client
        self.character_traits_kb: Dict[str, Any] = {}
        self._load_character_traits(traits_file_path)

    def _load_character_traits(self, traits_file_path: str):
        try:
            # Construct path relative to this file's location if necessary,
            # or assume traits_file_path is absolute or relative to execution dir.
            # For simplicity, let's assume it's relative to project root for now as per plan.
            if os.path.exists(traits_file_path):
                with open(traits_file_path, 'r') as f:
                    self.character_traits_kb = json.load(f)
                print(f"Successfully loaded character traits from {traits_file_path}")
            else:
                print(f"Warning: Character traits file not found at {traits_file_path}. Proceeding without knowledge base.")
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {traits_file_path}. Proceeding without knowledge base.")
            self.character_traits_kb = {}
        except Exception as e:
            print(f"An unexpected error occurred while loading {traits_file_path}: {e}. Proceeding without knowledge base.")
            self.character_traits_kb = {}

    def _get_known_traits(self, character_name: str) -> Optional[Dict[str, Any]]:
        # Case-insensitive matching for character names in the knowledge base
        for known_name, traits in self.character_traits_kb.items():
            if known_name.lower() == character_name.lower():
                return traits
        return None

    def _generate_character_visuals(self, character_name: str, known_traits: Optional[Dict[str, Any]] = None) -> str:
        system_prompt = "You are an AI assistant helping describe character visuals for an image generation prompt. Focus on appearance, age, and key features. Elaborate on any provided known traits."
        user_prompt_parts = [f"Describe the character '{character_name}'."]
        if known_traits and "description_keywords" in known_traits:
            keywords = ", ".join(known_traits["description_keywords"])
            user_prompt_parts.append(f"Incorporate these known iconic traits: {keywords}.")
        user_prompt = " ".join(user_prompt_parts)
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_outfit_details(self, character_name: str, character_visuals: str, known_traits: Optional[Dict[str, Any]] = None) -> str:
        system_prompt = "You are an AI assistant helping describe outfit details for a character based on their visuals for an image generation prompt. Elaborate on any provided known traits."
        user_prompt_parts = [f"Given the character '{character_name}' who looks like: '{character_visuals}', describe their outfit."]
        if known_traits and "outfit_keywords" in known_traits:
            keywords = ", ".join(known_traits["outfit_keywords"])
            user_prompt_parts.append(f"Known iconic outfit elements include: {keywords}.")
        user_prompt = " ".join(user_prompt_parts)
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_character_expression(self, character_name: str, character_visuals: str, known_traits: Optional[Dict[str, Any]] = None) -> str:
        system_prompt = "You are an AI assistant helping describe the facial expression or emotion of a character for an image generation prompt. Elaborate on any provided known traits."
        user_prompt_parts = [f"What is the expression of '{character_name}' who looks like: '{character_visuals}'?"]
        if known_traits and "expression_keywords" in known_traits:
            keywords = ", ".join(known_traits["expression_keywords"])
            user_prompt_parts.append(f"Known expressions or typical demeanor includes: {keywords}.")
        user_prompt = " ".join(user_prompt_parts)
        return self.lm_client.generate_text(system_prompt, user_prompt)

    # Other _generate methods remain similar for now, but could also accept known_traits if relevant
    # For example, environment might be influenced by a character's typical settings.

    def _generate_environment_setting(self, character_name: str, known_traits: Optional[Dict[str, Any]] = None) -> str:
        system_prompt = "You are an AI assistant creating a compelling environment/setting for a character in an image generation prompt."
        user_prompt_parts = [f"Describe a suitable setting for the character '{character_name}'."]
        if known_traits and "environment_keywords" in known_traits: # Assuming we might add this to JSON
             keywords = ", ".join(known_traits["environment_keywords"])
             user_prompt_parts.append(f"Consider their typical environments such as: {keywords}.")
        user_prompt = " ".join(user_prompt_parts)
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_environment_mood(self, setting_description: str) -> str:
        system_prompt = "You are an AI assistant defining the mood of an environment based on its description."
        user_prompt = f"For the setting '{setting_description}', what is the emotional atmosphere or tone?"
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_time_of_day(self, setting_description: str) -> str:
        system_prompt = "You are an AI assistant determining an appropriate time of day for a described environment."
        user_prompt = f"For the setting '{setting_description}', what is the time of day?"
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_camera_shot_type(self, character_name: str, setting_description: str) -> str:
        system_prompt = "You are an AI assistant selecting a cinematic camera shot type (e.g., close-up, medium shot, full shot) for a character in a scene."
        user_prompt = f"For '{character_name}' in '{setting_description}', what is a good camera shot type?"
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_camera_angle(self, character_name: str, shot_type: str) -> str:
        system_prompt = "You are an AI assistant selecting a camera angle (e.g., eye-level, low angle) for a character."
        user_prompt = f"For '{character_name}' with a '{shot_type}', suggest a camera angle."
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_camera_lens(self, shot_type: str, style_genre: str) -> str:
        system_prompt = "You are an AI assistant suggesting a camera lens (e.g., 35mm, wide-angle) appropriate for a shot type and artistic style."
        user_prompt = f"For a '{shot_type}' in a '{style_genre}' style, what lens would be suitable?"
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_lighting_description(self, character_name: str, setting_description: str, time_of_day: str) -> str:
        system_prompt = "You are an AI assistant describing the lighting of a scene."
        user_prompt = f"Describe the lighting for '{character_name}' in '{setting_description}' during '{time_of_day}'."
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_lighting_temperature(self, lighting_description: str) -> str:
        system_prompt = "You are an AI assistant describing the lighting temperature."
        user_prompt = f"For lighting described as '{lighting_description}', what is its color temperature (e.g. cool, warm)?"
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_artistic_genre(self, character_name: str, known_traits: Optional[Dict[str, Any]] = None) -> str:
        system_prompt = "You are an AI assistant determining an artistic genre (e.g., photorealistic, fantasy art) for a character."
        user_prompt_parts = [f"What artistic genre best fits '{character_name}'?"]
        if known_traits and "genre_keywords" in known_traits: # Assuming we might add this
            keywords = ", ".join(known_traits["genre_keywords"])
            user_prompt_parts.append(f"Consider their typical genre: {keywords}.")
        user_prompt = " ".join(user_prompt_parts)
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_artistic_influences(self, genre: str) -> str:
        system_prompt = "You are an AI assistant suggesting artistic influences (artists, movies) for a given genre."
        user_prompt = f"For the genre '{genre}', suggest some artistic influences."
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_subject_focus(self, character_description: str, setting: str) -> str:
        system_prompt = "You are an AI assistant defining the main subject or focus of an image, given a character and setting."
        user_prompt = f"The character is '{character_description}' in '{setting}'. What should be the main subject focus of the image?"
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_ambiance_atmosphere(self, character_description: str, setting: str, mood: str) -> str:
        system_prompt = "You are an AI assistant describing the overall ambiance and atmosphere to convey in an image."
        user_prompt = f"Given the character '{character_description}', the setting '{setting}', and mood '{mood}', describe the overall ambiance and atmosphere."
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def compose_prompt(self, character_name: str) -> CinematicPrompt:
        known_traits = self._get_known_traits(character_name)
        if known_traits:
            print(f"Found known traits for character: {character_name}")
        else:
            print(f"No known traits found for character: {character_name}. Proceeding with pure LLM generation.")

        # Generate components using LLM (via client), now passing known_traits
        char_visuals = self._generate_character_visuals(character_name, known_traits)
        char_outfit = self._generate_outfit_details(character_name, char_visuals, known_traits)
        char_expression = self._generate_character_expression(character_name, char_visuals, known_traits)

        env_setting = self._generate_environment_setting(character_name, known_traits) # Pass traits here too
        env_mood = self._generate_environment_mood(env_setting)
        env_time_of_day = self._generate_time_of_day(env_setting)

        art_genre = self._generate_artistic_genre(character_name, known_traits) # Pass traits here too
        art_influences = self._generate_artistic_influences(art_genre)

        cam_shot_type = self._generate_camera_shot_type(character_name, env_setting)
        cam_angle = self._generate_camera_angle(character_name, cam_shot_type)
        cam_lens = self._generate_camera_lens(cam_shot_type, art_genre)

        light_desc = self._generate_lighting_description(character_name, env_setting, env_time_of_day)
        light_temp = self._generate_lighting_temperature(light_desc)

        subject = self._generate_subject_focus(char_visuals, env_setting)
        ambiance = self._generate_ambiance_atmosphere(char_visuals, env_setting, env_mood)

        negative_prompts_list = []
        if known_traits and "negative_prompt_keywords" in known_traits:
            negative_prompts_list.extend(known_traits["negative_prompt_keywords"])
            print(f"Adding negative prompt keywords from KB: {negative_prompts_list}")


        # Assemble into Pydantic model
        prompt_data = CinematicPrompt(
            character=CharacterDetails(
                description=char_visuals,
                outfit=char_outfit,
                expression=char_expression,
            ),
            environment=EnvironmentDetails(
                setting=env_setting,
                mood=env_mood,
                time_of_day=env_time_of_day,
            ),
            camera=CameraSetup(
                shot_type=cam_shot_type,
                angle=cam_angle,
                lens=cam_lens,
                # composition_notes will be added later or be part of LLM generation for camera
            ),
            lighting=LightingDetails(
                description=light_desc,
                temperature=light_temp,
            ),
            style=ArtisticStyle(
                genre=art_genre,
                influences=art_influences,
                # additional_details can be added later
            ),
            subject_focus=subject,
            ambiance_atmosphere=ambiance,
            negative_prompt_elements=negative_prompts_list # Use list from KB
        )

        prompt_data.generate_prompt_string() # This will populate final_prompt_string
        return prompt_data

if __name__ == "__main__":
    # Example Usage
    print("Attempting to initialize LMStudioClient for composer test.")
    print("Ensure LM Studio is running at http://localhost:1234/v1 and a model is loaded.")

    actual_lm_client = None
    try:
        actual_lm_client = LMStudioClient()
        print("LMStudioClient initialized.")
    except Exception as e:
        print(f"Failed to initialize LMStudioClient: {e}. Using a dummy client for composer structure test.")
        class DummyClient:
            def generate_text(self, system_prompt, user_prompt, max_tokens=150):
                # Simulate some variation based on prompt
                if "Marge Simpson" in user_prompt:
                    if "visuals" in system_prompt: return "Marge Simpson with tall blue hair, yellow skin."
                    if "outfit" in system_prompt: return "Wearing her iconic green dress and red pearls."
                    if "expression" in system_prompt: return "A patient, motherly smile."
                elif "Gandalf" in user_prompt:
                     if "visuals" in system_prompt: return "Gandalf the Grey, an old wizard with a long grey beard."
                     if "outfit" in system_prompt: return "Grey robes, pointed hat, and a wooden staff."
                return f"Dummy LLM response for: {user_prompt[:50]}..."
        actual_lm_client = DummyClient()

    # Assuming 'data/character_traits.json' is in the parent directory of 'composer'
    # Or if script is run from project root, 'data/character_traits.json' is correct.
    # For robustness, construct path from this script's location if needed,
    # but current PromptComposer init also assumes 'data/character_traits.json' from root.
    composer = PromptComposer(lm_client=actual_lm_client, traits_file_path="data/character_traits.json")

    characters_to_test = [
        "Marge Simpson", # Should use knowledge base
        "Gandalf the Grey", # Should use knowledge base
        "A futuristic space explorer", # Should NOT use knowledge base (pure LLM)
        "wonder woman" # Should use knowledge base (testing case insensitivity)
    ]

    for character_name in characters_to_test:
        print(f"\n--- Composing prompt for: {character_name} ---")
        structured_prompt = composer.compose_prompt(character_name)

        print("\n--- Structured Output (Pydantic Model from Composer) ---")
        # Ensure the model_dump_json is available, it's a pydantic feature.
        # If structured_prompt can be None or not a Pydantic model, add checks.
        if hasattr(structured_prompt, 'model_dump_json'):
            print(structured_prompt.model_dump_json(indent=2))
        else:
            print("Error: structured_prompt is not a valid Pydantic model.")

        print("\n--- Final Composed Prompt String from Composer ---")
        print(structured_prompt.final_prompt_string if hasattr(structured_prompt, 'final_prompt_string') else "Error: No final prompt string.")
