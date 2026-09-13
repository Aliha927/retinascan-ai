import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Header
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from supabase import create_client, Client
import json
from datetime import datetime
from typing import Optional
from pathlib import Path

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Safeguard initialization to prevent crash on startup if missing
if not SUPABASE_URL or not SUPABASE_KEY:
    print("[WARNING] SUPABASE_URL or SUPABASE_KEY missing in environment!")
    supabase = None
else:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

if not GEMINI_API_KEY:
    print("[WARNING] GEMINI_API_KEY missing in environment!")
    gemini_client = None
else:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="RetinaScan AI API - Hackathon Edition", version="3.0")

# Replace your current CORS setup with this:
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://retinascan-ai-three.vercel.app",
]
g
# If ALLOWED_ORIGINS env var exists, add those explicitly as well
env_origins = os.getenv("ALLOWED_ORIGINS")
if env_origins:
    origins.extend([origin.strip() for origin in env_origins.split(",") if origin.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",  # Automatically permits all Vercel deployments!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
from model import load_dr_model, predict_dr
from utils import preprocess_image, generate_gradcam_overlay

model = load_dr_model()


def get_current_user_email(authorization: Optional[str]) -> str:
    """
    Verifies the Supabase JWT sent by the frontend (Authorization: Bearer <token>).
    Returns the logged-in user's email, or 'anonymous' if no valid token was sent.
    This does NOT block unauthenticated requests (kept permissive for demo speed) —
    tighten this to raise HTTPException(401) if you want to require login.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return "anonymous"
    token = authorization.split(" ", 1)[1]
    try:
        user_response = supabase.auth.get_user(token)
        return user_response.user.email
    except Exception:
        return "anonymous"


@app.post("/api/screen")
async def screen_retina(
    file: UploadFile = File(...),
    patient_name: str = Form("Anonymous Patient"),
    language: str = Form("english"),
    authorization: Optional[str] = Header(None)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Upload an image.")

    user_email = get_current_user_email(authorization)
    image_bytes = await file.read()

    # 1. Preprocess & Predict using CNN
    original_image, input_tensor = preprocess_image(image_bytes)
    predicted_class, confidence = predict_dr(model, input_tensor)

    # Safety Guardrail Check
    warning_flag = confidence < 75.0

    # 2. Generate Grad-CAM Visual Explanation
    gradcam_bytes = generate_gradcam_overlay(original_image, model, input_tensor)
    gradcam_base64 = base64.b64encode(gradcam_bytes).decode("utf-8")

    # 3. Generative AI Reporting Layer (Using Gemini)
    prompt = f"""
    You are an expert clinical ophthalmology assistant.
    A deep learning screening model analyzed a retinal fundus image with the following results:
    - Classification: {predicted_class}
    - Model Confidence: {confidence:.2f}%
    - Safety Warning Triggered: {warning_flag}
    - Requested Patient Language: {language}

    Generate structured JSON output containing:
    1. "clinical_report": A precise clinical summary for the referring physician detailing findings and recommendations.
    2. "patient_report": A compassionate, plain-language explanation for the patient.
    3. "urdu_report": The patient explanation translated accurately into fluent Urdu or Roman Urdu script.

    Never state this as a confirmed diagnosis — always frame it as an AI-assisted
    screening result that requires clinician confirmation.

    JSON Structure format required:
    {{
      "clinical_report": "...",
      "patient_report": "...",
      "urdu_report": "..."
    }}
    """

    try:
        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            ),
        )
        reports = json.loads(response.text)
    except Exception as e:
        reports = {
            "clinical_report": f"Diagnosis: {predicted_class} ({confidence:.1f}%). Note: {str(e)}",
            "patient_report": f"Your scan indicates {predicted_class}. Please see a specialist.",
            "urdu_report": f"آپ کی اسکریننگ کا نتیجہ: {predicted_class}. براہ کرم ماہر امراض چشم سے رجوع کریں۔"
        }

    # Save to Supabase (persistent, cloud-hosted Postgres)
    try:
        supabase.table("scans").insert({
            "user_email": user_email,
            "patient_name": patient_name,
            "prediction": predicted_class,
            "confidence": round(confidence, 2),
            "clinical_report": reports.get("clinical_report"),
            "patient_report": reports.get("patient_report"),
            "urdu_report": reports.get("urdu_report"),
            "timestamp": datetime.now().isoformat()
        }).execute()
    except Exception as e:
        print(f"[WARNING] Failed to save scan to Supabase: {e}")

    return {
        "prediction": predicted_class,
        "confidence": round(confidence, 2),
        "warning_flag": warning_flag,
        "gradcam_image": f"data:image/jpeg;base64,{gradcam_base64}",
        "clinical_report": reports.get("clinical_report"),
        "patient_report": reports.get("patient_report"),
        "urdu_report": reports.get("urdu_report")
    }


@app.get("/api/history")
async def get_history(authorization: Optional[str] = Header(None)):
    user_email = get_current_user_email(authorization)
    try:
        query = supabase.table("scans").select("*").order("id", desc=True).limit(10)
        if user_email != "anonymous":
            query = query.eq("user_email", user_email)
        result = query.execute()
        return result.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
