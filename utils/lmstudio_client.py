import requests
import json

class LMStudioClient:
    def __init__(self, base_url: str = "http://localhost:1234/v1"):
        self.base_url = base_url
        self.chat_completions_url = f"{self.base_url}/chat/completions"

    def generate_text(self, system_prompt: str, user_prompt: str, model: str = "local-model", max_tokens: int = 500, temperature: float = 0.7, stream: bool = False) -> str:
        """
        Generates text using the LM Studio API.

        Args:
            system_prompt: The system message to guide the AI.
            user_prompt: The user's message or query.
            model: The model to use (LM Studio uses "local-model" for any loaded model).
            max_tokens: Maximum number of tokens to generate.
            temperature: Controls randomness (0.0 to 2.0).
            stream: Whether to stream the response (not fully handled here, returns concatenated string).

        Returns:
            The AI's generated text response.
            Returns an error message string if the request fails.
        """
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream
        }

        try:
            response = requests.post(self.chat_completions_url, headers=headers, json=payload, timeout=120) # 120 seconds timeout
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)

            completion_data = response.json()

            if not completion_data.get("choices") or not completion_data["choices"][0].get("message") or not completion_data["choices"][0]["message"].get("content"):
                return "Error: Unexpected response structure from LM Studio API."

            generated_text = completion_data["choices"][0]["message"]["content"]
            return generated_text.strip()

        except requests.exceptions.RequestException as e:
            error_message = f"Error connecting to LM Studio or during request: {e}"
            print(error_message) # Also print for server-side visibility
            return f"Error: {error_message}" # Return error message to caller
        except json.JSONDecodeError:
            error_message = "Error: Could not decode JSON response from LM Studio API."
            print(error_message)
            return f"Error: {error_message}"
        except KeyError:
            error_message = "Error: 'choices' or 'content' not found in LM Studio API response."
            print(error_message)
            return f"Error: {error_message}"


if __name__ == "__main__":
    # This is a test that will only work if LM Studio is running with a model loaded.
    # And the server is accessible at http://localhost:1234

    print("Attempting to connect to LM Studio...")
    print(f"Ensure LM Studio is running and a model is loaded, accessible at http://localhost:1234/v1.")

    client = LMStudioClient()

    system_prompt_test = "You are a helpful AI assistant."
    user_prompt_test = "Explain the concept of '量子力学' (quantum mechanics) in simple terms for a beginner, in one sentence."

    print(f"\nSending test prompt to LM Studio:")
    print(f"System: {system_prompt_test}")
    print(f"User: {user_prompt_test}")

    response_text = client.generate_text(system_prompt_test, user_prompt_test, max_tokens=100)

    print("\n--- LM Studio Response ---")
    if response_text.startswith("Error:"):
        print(f"Test failed. {response_text}")
        print("Please ensure LM Studio is running, a model is loaded, and the server is enabled.")
    else:
        print(response_text)
        print("\nTest successful if you see a relevant response above.")

    print("\n--- Testing another prompt ---")
    system_prompt_story = "You are a master storyteller."
    user_prompt_story = "Tell me a very short story (2-3 sentences) about a brave knight and a friendly dragon."

    print(f"System: {system_prompt_story}")
    print(f"User: {user_prompt_story}")

    response_story = client.generate_text(system_prompt_story, user_prompt_story, max_tokens=150)
    print("\n--- LM Studio Response ---")
    if response_story.startswith("Error:"):
        print(f"Test failed. {response_story}")
    else:
        print(response_story)
        print("\nTest successful if you see a relevant story above.")

    print("\n--- Testing error handling for a non-existent server (example) ---")
    client_bad = LMStudioClient(base_url="http://localhost:1235/v1") # Assuming this port is not used
    response_bad = client_bad.generate_text("Test", "Test")
    print("\n--- LM Studio Response (should be an error) ---")
    if response_bad.startswith("Error:"):
        print(f"Successfully caught error: {response_bad}")
    else:
        print(f"Test failed, expected an error but got: {response_bad}")
