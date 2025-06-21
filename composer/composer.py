from models.prompt_schema import (
    CinematicPrompt,
    CharacterDetails,
    EnvironmentDetails,
    CameraSetup,
    LightingDetails,
    ArtisticStyle,
)
from utils.lmstudio_client import LMStudioClient


class PromptComposer:
    def __init__(self, lm_client): # lm_client should be an instance of LMStudioClient or similar
        self.lm_client = lm_client

    def _generate_character_visuals(self, character_name: str) -> str:
        system_prompt = "You are an AI assistant helping describe character visuals for an image generation prompt. Focus on appearance, age, and key features."
        user_prompt = f"Describe the character '{character_name}'."
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_outfit_details(self, character_name: str, character_visuals: str) -> str:
        system_prompt = "You are an AI assistant helping describe outfit details for a character based on their visuals for an image generation prompt."
        user_prompt = f"Given the character '{character_name}' who looks like: '{character_visuals}', describe their outfit."
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_character_expression(self, character_name: str, character_visuals: str) -> str:
        system_prompt = "You are an AI assistant helping describe the facial expression or emotion of a character for an image generation prompt."
        user_prompt = f"What is the expression of '{character_name}' who looks like: '{character_visuals}'?"
        return self.lm_client.generate_text(system_prompt, user_prompt)

    def _generate_environment_setting(self, character_name: str) -> str:
        system_prompt = "You are an AI assistant creating a compelling environment/setting for a character in an image generation prompt."
        user_prompt = f"Describe a suitable setting for the character '{character_name}'."
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

    def _generate_artistic_genre(self, character_name: str) -> str:
        system_prompt = "You are an AI assistant determining an artistic genre (e.g., photorealistic, fantasy art) for a character."
        user_prompt = f"What artistic genre best fits '{character_name}'?"
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
        # Generate components using LLM (via client)
        char_visuals = self._generate_character_visuals(character_name)
        char_outfit = self._generate_outfit_details(character_name, char_visuals)
        char_expression = self._generate_character_expression(character_name, char_visuals)

        env_setting = self._generate_environment_setting(character_name)
        env_mood = self._generate_environment_mood(env_setting)
        env_time_of_day = self._generate_time_of_day(env_setting)

        art_genre = self._generate_artistic_genre(character_name)
        art_influences = self._generate_artistic_influences(art_genre)

        cam_shot_type = self._generate_camera_shot_type(character_name, env_setting)
        cam_angle = self._generate_camera_angle(character_name, cam_shot_type)
        cam_lens = self._generate_camera_lens(cam_shot_type, art_genre)

        light_desc = self._generate_lighting_description(character_name, env_setting, env_time_of_day)
        light_temp = self._generate_lighting_temperature(light_desc)

        subject = self._generate_subject_focus(char_visuals, env_setting)
        ambiance = self._generate_ambiance_atmosphere(char_visuals, env_setting, env_mood)

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
            # negative_prompt_elements can be added later or be fixed
        )

        prompt_data.generate_prompt_string() # This will populate final_prompt_string
        return prompt_data

if __name__ == "__main__":
    # Example Usage (requires a placeholder or real LM client)
    # This will now attempt to connect to a running LM Studio instance.
    print("Attempting to initialize LMStudioClient for composer test.")
    print("Ensure LM Studio is running at http://localhost:1234/v1 and a model is loaded.")
    try:
        actual_lm_client = LMStudioClient()
        # Quick check if client can connect (optional, client handles errors internally too)
        # This is a simplified check; proper would be a dedicated health check endpoint or a dummy call
        # For now, we rely on the generate_text calls to show success/failure.
        print("LMStudioClient initialized. Attempting to compose prompt...")
    except Exception as e:
        print(f"Failed to initialize LMStudioClient: {e}")
        print("Composer test will likely fail or use dummy data if client is not functional.")
        # Fallback to a dummy client if real one fails for local testing of composer structure
        class DummyClient:
            def generate_text(self, sp, up, max_tokens=50): return f"Dummy response for {up[:30]}"
        actual_lm_client = DummyClient()

    composer = PromptComposer(lm_client=actual_lm_client)

    character_name = "Gandalf the Grey"
    print(f"\nComposing prompt for: {character_name}\n")

    structured_prompt = composer.compose_prompt(character_name)

    print("\n--- Structured Output (Pydantic Model) ---")
    print(structured_prompt.model_dump_json(indent=2))

    print("\n--- Final Composed Prompt String ---")
    print(structured_prompt.final_prompt_string)

    # Example 2
    character_name_2 = "Cyberpunk Detective Kaito"
    print(f"\nComposing prompt for: {character_name_2}\n")
    structured_prompt_2 = composer.compose_prompt(character_name_2)
    print("\n--- Structured Output (Pydantic Model) ---")
    print(structured_prompt_2.model_dump_json(indent=2))
    print("\n--- Final Composed Prompt String ---")
    print(structured_prompt_2.final_prompt_string)
