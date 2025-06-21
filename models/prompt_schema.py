from pydantic import BaseModel, Field
from typing import Optional

class CharacterDetails(BaseModel):
    description: str = Field(..., description="Detailed description of the character's appearance, age, key features.")
    outfit: Optional[str] = Field(None, description="Specific attire or costume of the character.")
    expression: Optional[str] = Field(None, description="The character's facial expression or emotion.")

class EnvironmentDetails(BaseModel):
    setting: str = Field(..., description="The overall location or backdrop of the scene.")
    mood: Optional[str] = Field(None, description="The emotional atmosphere or tone of the environment (e.g., ominous, serene, chaotic).")
    time_of_day: Optional[str] = Field(None, description="e.g., golden hour, twilight, midday, midnight.")

class CameraSetup(BaseModel):
    shot_type: str = Field(..., description="Type of camera shot (e.g., close-up, medium shot, cowboy shot, full shot, long shot, establishing shot).")
    angle: Optional[str] = Field(None, description="Camera angle (e.g., eye-level, low angle, high angle, bird's-eye view, worm's-eye view).")
    lens: Optional[str] = Field(None, description="Camera lens type or focal length (e.g., wide-angle, 35mm, 50mm, 85mm, telephoto, fisheye).")
    composition_notes: Optional[str] = Field(None, description="Specific compositional elements, e.g., rule of thirds, leading lines, depth of field.")

class LightingDetails(BaseModel):
    description: str = Field(..., description="Description of the lighting setup (e.g., softbox, rim lighting, natural light, volumetric lighting, cinematic lighting).")
    temperature: Optional[str] = Field(None, description="Color temperature of the light (e.g., cool, warm, blue hour).")

class ArtisticStyle(BaseModel):
    genre: str = Field(..., description="Primary genre (e.g., photorealistic, fantasy art, anime, comic book, impressionistic, surreal).")
    influences: Optional[str] = Field(None, description="Specific artists, movies, or art movements as style references.")
    additional_details: Optional[str] = Field(None, description="Any other stylistic choices like color grading, film grain, etc.")

class CinematicPrompt(BaseModel):
    character: CharacterDetails
    environment: EnvironmentDetails
    camera: CameraSetup
    lighting: LightingDetails
    style: ArtisticStyle
    # Overall prompt elements
    subject_focus: str = Field(..., description="The main subject or focus of the image.")
    ambiance_atmosphere: str = Field(..., description="Overall ambiance and atmosphere to convey.")
    negative_prompt_elements: Optional[list[str]] = Field(default_factory=list, description="Elements to exclude or avoid.")
    # Final composed prompt string
    final_prompt_string: Optional[str] = Field(None, description="The fully composed prompt string ready for SDXL.")

    def generate_prompt_string(self) -> str:
        """Generates a coherent string from the structured prompt details."""
        parts = []

        # Character
        char_parts = [self.character.description]
        if self.character.outfit:
            char_parts.append(f"wearing {self.character.outfit}")
        if self.character.expression:
            char_parts.append(f"with a {self.character.expression} expression")
        parts.append(", ".join(char_parts))

        # Environment
        env_parts = [f"in a {self.environment.setting}"]
        if self.environment.mood:
            env_parts.append(f"creating a {self.environment.mood} mood")
        if self.environment.time_of_day:
            env_parts.append(f"during {self.environment.time_of_day}")
        parts.append(" ".join(env_parts))

        # Subject and Ambiance
        parts.append(f"Focusing on {self.subject_focus}.")
        parts.append(f"The overall atmosphere is {self.ambiance_atmosphere}.")

        # Camera
        cam_parts = [f"{self.camera.shot_type}"]
        if self.camera.angle:
            cam_parts.append(f"{self.camera.angle}")
        if self.camera.lens:
            cam_parts.append(f"using a {self.camera.lens} lens")
        if self.camera.composition_notes:
            cam_parts.append(f"with {self.camera.composition_notes}")
        parts.append("Camera: " + ", ".join(cam_parts) + ".")

        # Lighting
        light_parts = [self.lighting.description]
        if self.lighting.temperature:
            light_parts.append(f"with {self.lighting.temperature} tones")
        parts.append("Lighting: " + ", ".join(light_parts) + ".")

        # Style
        style_parts = [f"Style: {self.style.genre}"]
        if self.style.influences:
            style_parts.append(f"influenced by {self.style.influences}")
        if self.style.additional_details:
            style_parts.append(self.style.additional_details)
        parts.append(". ".join(style_parts) + ".")

        # Common SDXL terms
        parts.append("ultra-detailed, 8k, photorealistic, cinematic composition")

        prompt_str = " ".join(parts)

        if self.negative_prompt_elements:
            # Basic handling for negative prompts, can be refined.
            # Some UIs use --neg, others have a dedicated field.
            # For a string, this is a common way.
            prompt_str += " --neg " + ", ".join(self.negative_prompt_elements)

        self.final_prompt_string = prompt_str
        return prompt_str

if __name__ == "__main__":
    # Example Usage:
    sample_prompt = CinematicPrompt(
        character=CharacterDetails(
            description="A rugged space pirate, mid-30s, cybernetic eye",
            outfit="worn leather jacket, utility belt with gadgets",
            expression="determined smirk"
        ),
        environment=EnvironmentDetails(
            setting="dimly lit starship bridge, holographic displays flickering",
            mood="tense and expectant",
            time_of_day="artificial night"
        ),
        camera=CameraSetup(
            shot_type="medium close-up",
            angle="slight low angle",
            lens="35mm anamorphic",
            composition_notes="character slightly off-center, bokeh background"
        ),
        lighting=LightingDetails(
            description="key light from a console screen, blue rim light from a viewport",
            temperature="cool with warm highlights"
        ),
        style=ArtisticStyle(
            genre="sci-fi realism",
            influences="Blade Runner, Mass Effect",
            additional_details="subtle film grain, high contrast"
        ),
        subject_focus="the pirate's cybernetic eye and their reaction to an off-screen event",
        ambiance_atmosphere="a mix of high-tech grit and suspenseful exploration",
        negative_prompt_elements=["cartoonish", "blurry", "low resolution"]
    )
    print("--- Structured Prompt ---")
    print(sample_prompt.model_dump_json(indent=2))
    print("\n--- Generated Prompt String ---")
    generated_string = sample_prompt.generate_prompt_string()
    print(generated_string)
    assert sample_prompt.final_prompt_string == generated_string

    # Example 2: Minimal
    sample_minimal_prompt = CinematicPrompt(
        character=CharacterDetails(description="A young witch"),
        environment=EnvironmentDetails(setting="enchanted forest"),
        camera=CameraSetup(shot_type="full shot"),
        lighting=LightingDetails(description="moonlight"),
        style=ArtisticStyle(genre="fantasy art"),
        subject_focus="the witch casting a spell",
        ambiance_atmosphere="mystical and magical"
    )
    print("\n--- Minimal Structured Prompt ---")
    print(sample_minimal_prompt.model_dump_json(indent=2))
    print("\n--- Minimal Generated Prompt String ---")
    print(sample_minimal_prompt.generate_prompt_string())

    # Example 3: Character focused, let LLM fill more
    sample_character_focus = CinematicPrompt(
        character=CharacterDetails(
            description="Grizzled old wizard with a long white beard",
            outfit="star-patterned blue robes",
            expression="wise and thoughtful"
        ),
        environment=EnvironmentDetails(
            setting="a cluttered study filled with ancient books and magical artifacts",
        ),
        camera=CameraSetup(
            shot_type="medium shot",
        ),
        lighting=LightingDetails(
            description="warm light from a fireplace and glowing crystals",
        ),
        style=ArtisticStyle(
            genre="classic fantasy illustration",
        ),
        subject_focus="the wizard poring over an ancient tome",
        ambiance_atmosphere="scholarly magic and ancient secrets"
    )
    print("\n--- Character Focus Structured Prompt ---")
    print(sample_character_focus.model_dump_json(indent=2))
    print("\n--- Character Focus Generated Prompt String ---")
    print(sample_character_focus.generate_prompt_string())
