from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust as needed for your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration for Google Gemini API ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Choose a suitable Gemini model. 'gemini-1.5-flash-latest' is a fast and versatile model.
# For a simpler, purely text-based model, 'gemini-pro' was common, but check current recommendations.
GEMINI_MODEL_NAME = "gemini-1.5-flash-latest" # Or "gemini-pro" if you prefer and it's available for your key
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GOOGLE_API_KEY}"

if not GOOGLE_API_KEY:
    print("Error: GOOGLE_API_KEY not found in environment variables.")
    # You might want to exit or raise an exception here if the key is critical for startup
else:
    print("Google API Key loaded (first 8 chars):", GOOGLE_API_KEY[:8] if GOOGLE_API_KEY else "None")


class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
def chat(req: ChatRequest):
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="Google API Key not configured.")

    # Payload for Gemini API (text-only input)
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": req.message}
                ]
            }
        ],
        # Optional: Configure generation parameters
        # "generationConfig": {
        #     "temperature": 0.7,
        #     "maxOutputTokens": 150,
        # }
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        result = response.json()

        # --- Parse Gemini API Response ---
        # The structure can vary slightly based on model and whether streaming is used.
        # For a typical non-streaming generateContent response:
        if "candidates" in result and result["candidates"]:
            first_candidate = result["candidates"][0]
            if "content" in first_candidate and "parts" in first_candidate["content"] and first_candidate["content"]["parts"]:
                generated_text = first_candidate["content"]["parts"][0].get("text", "")
                if not generated_text and "functionCall" in first_candidate["content"]["parts"][0]:
                     # Handle function calling if you intend to use it, otherwise it's an unexpected response for chat
                    print("Warning: Received a function call, not text.")
                    generated_text = f"Received a function call: {first_candidate['content']['parts'][0]['functionCall']}"
                elif not generated_text:
                    generated_text = "No text content found in the response."

                # Check for safety ratings and finish reason if needed
                finish_reason = first_candidate.get("finishReason", "UNKNOWN")
                if finish_reason == "SAFETY":
                    print("Warning: Response blocked due to safety reasons.")
                    # You might want to return a specific message or handle this differently
                    # For now, we'll return what text might have been partially generated or an empty string
                    # if it's fully blocked before any text part.
                    safety_ratings = first_candidate.get("safetyRatings", [])
                    print(f"Safety Ratings: {safety_ratings}")
                    # Depending on your policy, you might want to return a generic message or the (potentially empty) text.
                    # If generated_text is empty due to full block, this will be the case.
                    if not generated_text:
                         generated_text = "Response blocked due to safety settings."


                return {"reply": generated_text.strip()}
            else:
                raise Exception("Invalid response structure from Gemini: Missing content or parts.")
        elif "promptFeedback" in result and "blockReason" in result["promptFeedback"]:
            block_reason = result["promptFeedback"]["blockReason"]
            error_message = f"Prompt blocked due_to {block_reason}."
            if "safetyRatings" in result["promptFeedback"]:
                error_message += f" Safety Ratings: {result['promptFeedback']['safetyRatings']}"
            print(f"Gemini API error: {error_message}")
            raise Exception(error_message)
        else:
            raise Exception(f"Unexpected response structure from Gemini: {result}")

    except requests.exceptions.HTTPError as http_err:
        error_detail = f"HTTP error occurred: {http_err}"
        try:
            # Try to get more details from the response body if available
            error_body = http_err.response.json()
            error_detail += f" - {error_body.get('error', {}).get('message', str(error_body))}"
        except ValueError: # JSONDecodeError
            error_detail += f" - {http_err.response.text}"
        print(f"Gemini API error: {error_detail}")
        raise HTTPException(status_code=http_err.response.status_code, detail=error_detail)
    except Exception as e:
        print(f"Gemini API error or internal error: {e}")
        # It's good practice to avoid sending raw internal error messages to the client
        # For debugging, you log the full error 'e'
        # For the client, you send a more generic message or a sanitized version of 'e'
        raise HTTPException(status_code=500, detail=f"Error processing your request: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    # Make sure to create a .env file with your GOOGLE_API_KEY
    # Example .env content:
    # GOOGLE_API_KEY=AIzaSyYOUR_ACTUAL_API_KEY_HERE
    uvicorn.run(app, host="0.0.0.0", port=8000)