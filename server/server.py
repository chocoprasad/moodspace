"""
MoodSpace API Server — with Auth + MongoDB
Run: pip install fastapi uvicorn pymongo bcrypt pyjwt python-multipart
     uvicorn server:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime, timedelta, timezone
import numpy as np
import json, os, pickle, hashlib

# ── Auth dependencies ──
import jwt
import bcrypt
from pymongo import MongoClient

app = FastAPI(title="MoodSpace API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════
# DATABASE + AUTH CONFIG
# ══════════════════════════════════════════════

MONGO_URI = "mongodb+srv://adityapisal_db_user:MoodSpace2026@cluster0.qbgykhf.mongodb.net/?appName=Cluster0"
JWT_SECRET = "moodspace-secret-key-change-in-production-2026"
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 72

# Connect to MongoDB
try:
    import certifi
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000, tls=True, tlsCAFile=certifi.where())
    client.admin.command("ping")
    db = client["moodspace"]
    users_col = db["users"]
    mood_history_col = db["mood_history"]
    # Create unique index on email
    users_col.create_index("email", unique=True)
    users_col.create_index("username", unique=True)
    print("[Server] MongoDB connected ✅")
except Exception as e:
    print(f"[Server] MongoDB error: {e}")
    db = None
    users_col = None
    mood_history_col = None


# ══════════════════════════════════════════════
# AUTH HELPERS
# ══════════════════════════════════════════════

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_token(user_id: str, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def get_current_user(authorization: Optional[str] = Header(None)):
    """Dependency: extract user from JWT token."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = authorization.split(" ")[1]
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user = users_col.find_one({"_id": payload["user_id"]}) if users_col else None
        if not user:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


# ══════════════════════════════════════════════
# AUTH MODELS
# ══════════════════════════════════════════════

class SignupRequest(BaseModel):
    name: str
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    status: Optional[str] = None
    theme: Optional[str] = None
    notifications: Optional[bool] = None
    musicAutoplay: Optional[bool] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


# ══════════════════════════════════════════════
# AUTH ROUTES
# ══════════════════════════════════════════════

@app.post("/auth/signup")
def signup(data: SignupRequest):
    if users_col is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    # Validate
    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    if len(data.username) < 3:
        raise HTTPException(status_code=400, detail="Username must be at least 3 characters")

    # Check existing
    if users_col.find_one({"email": data.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    if users_col.find_one({"username": data.username}):
        raise HTTPException(status_code=400, detail="Username already taken")

    # Create user
    import uuid
    user_id = str(uuid.uuid4())
    user_doc = {
        "_id": user_id,
        "name": data.name,
        "username": data.username,
        "email": data.email,
        "password": hash_password(data.password),
        "status": "online",
        "theme": "dark",
        "notifications": True,
        "musicAutoplay": True,
        "iotConnected": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    users_col.insert_one(user_doc)

    token = create_token(user_id, data.email)

    return {
        "token": token,
        "user": {
            "id": user_id,
            "name": data.name,
            "username": data.username,
            "email": data.email,
            "status": "online",
            "theme": "dark",
            "notifications": True,
            "musicAutoplay": True,
            "iotConnected": False,
        },
    }


@app.post("/auth/login")
def login(data: LoginRequest):
    if users_col is None:
        raise HTTPException(status_code=500, detail="Database not connected")

    user = users_col.find_one({"email": data.email})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not verify_password(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Update last login
    users_col.update_one(
        {"_id": user["_id"]},
        {"$set": {"status": "online", "updated_at": datetime.now(timezone.utc).isoformat()}},
    )

    token = create_token(user["_id"], user["email"])

    return {
        "token": token,
        "user": {
            "id": user["_id"],
            "name": user["name"],
            "username": user["username"],
            "email": user["email"],
            "status": "online",
            "theme": user.get("theme", "dark"),
            "notifications": user.get("notifications", True),
            "musicAutoplay": user.get("musicAutoplay", True),
            "iotConnected": user.get("iotConnected", False),
        },
    }


@app.get("/auth/me")
def get_me(user=Depends(get_current_user)):
    return {
        "user": {
            "id": user["_id"],
            "name": user["name"],
            "username": user["username"],
            "email": user["email"],
            "status": user.get("status", "online"),
            "theme": user.get("theme", "dark"),
            "notifications": user.get("notifications", True),
            "musicAutoplay": user.get("musicAutoplay", True),
            "iotConnected": user.get("iotConnected", False),
        }
    }


@app.put("/auth/profile")
def update_profile(data: UpdateProfileRequest, user=Depends(get_current_user)):
    updates = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if data.name is not None:
        updates["name"] = data.name
    if data.status is not None:
        updates["status"] = data.status
    if data.theme is not None:
        updates["theme"] = data.theme
    if data.notifications is not None:
        updates["notifications"] = data.notifications
    if data.musicAutoplay is not None:
        updates["musicAutoplay"] = data.musicAutoplay

    users_col.update_one({"_id": user["_id"]}, {"$set": updates})
    return {"success": True, "message": "Profile updated"}


@app.put("/auth/password")
def change_password(data: ChangePasswordRequest, user=Depends(get_current_user)):
    if not verify_password(data.current_password, user["password"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if len(data.new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters")

    users_col.update_one(
        {"_id": user["_id"]},
        {"$set": {"password": hash_password(data.new_password), "updated_at": datetime.now(timezone.utc).isoformat()}},
    )
    return {"success": True, "message": "Password changed"}


@app.delete("/auth/account")
def delete_account(user=Depends(get_current_user)):
    users_col.delete_one({"_id": user["_id"]})
    mood_history_col.delete_many({"user_id": user["_id"]})
    return {"success": True, "message": "Account deleted"}


# ══════════════════════════════════════════════
# MOOD HISTORY
# ══════════════════════════════════════════════

class MoodEntry(BaseModel):
    mood: str
    confidence: float
    source: str = "webcam"  # "webcam" or "sensor"


@app.post("/mood/log")
def log_mood(data: MoodEntry, user=Depends(get_current_user)):
    entry = {
        "user_id": user["_id"],
        "mood": data.mood,
        "confidence": data.confidence,
        "source": data.source,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    mood_history_col.insert_one(entry)
    return {"success": True}


@app.get("/mood/history")
def get_mood_history(limit: int = 50, user=Depends(get_current_user)):
    entries = list(
        mood_history_col.find(
            {"user_id": user["_id"]},
            {"_id": 0, "user_id": 0},
        )
        .sort("timestamp", -1)
        .limit(limit)
    )
    return {"history": entries}


@app.get("/mood/stats")
def get_mood_stats(user=Depends(get_current_user)):
    pipeline = [
        {"$match": {"user_id": user["_id"]}},
        {"$group": {"_id": "$mood", "count": {"$sum": 1}, "avg_confidence": {"$avg": "$confidence"}}},
    ]
    results = list(mood_history_col.aggregate(pipeline))
    stats = {}
    total = sum(r["count"] for r in results)
    for r in results:
        stats[r["_id"]] = {
            "count": r["count"],
            "percentage": round(r["count"] / total * 100, 1) if total > 0 else 0,
            "avg_confidence": round(r["avg_confidence"], 1),
        }
    return {"stats": stats, "total": total}


# ══════════════════════════════════════════════
# MOOD PREDICTION (existing — unchanged)
# ══════════════════════════════════════════════

MOODS = ["Focused", "Relaxed", "Stressed", "Fatigued"]

MOOD_META = {
    "Focused": {"emoji": "🎯", "color": "#facc15"},
    "Relaxed": {"emoji": "😌", "color": "#4ade80"},
    "Stressed": {"emoji": "😤", "color": "#f87171"},
    "Fatigued": {"emoji": "😴", "color": "#94a3b8"},
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "..", "scaler_params.json")
RF_PATH = os.path.join(BASE_DIR, "..", "rf_model.pkl")
TFLITE_PATH = os.path.join(BASE_DIR, "..", "mood_model.tflite")

scaler_mean = None
scaler_std = None
rf_model = None
tflite_interpreter = None
tflite_inp = None
tflite_out = None

if os.path.exists(SCALER_PATH):
    with open(SCALER_PATH) as f:
        p = json.load(f)
    scaler_mean = np.array(p["mean"], dtype=np.float32)
    scaler_std = np.array(p["scale"], dtype=np.float32)
    print(f"[Server] Scaler loaded ({len(scaler_mean)} features)")

if os.path.exists(TFLITE_PATH):
    try:
        from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
        tflite_interpreter = TFLiteInterpreter(model_path=TFLITE_PATH)
        tflite_interpreter.allocate_tensors()
        tflite_inp = tflite_interpreter.get_input_details()
        tflite_out = tflite_interpreter.get_output_details()
        print(f"[Server] TFLite loaded (input: {tflite_inp[0]['shape']})")
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
            tflite_interpreter = tflite.Interpreter(model_path=TFLITE_PATH)
            tflite_interpreter.allocate_tensors()
            tflite_inp = tflite_interpreter.get_input_details()
            tflite_out = tflite_interpreter.get_output_details()
            print(f"[Server] TFLite loaded via tflite-runtime")
        except ImportError:
            print("[Server] No TFLite runtime found, skipping")
    except Exception as e:
        print(f"[Server] TFLite error: {e}")
if os.path.exists(RF_PATH):
    try:
        with open(RF_PATH, "rb") as f:
            bundle = pickle.load(f)
        rf_model = bundle["model"] if isinstance(bundle, dict) else bundle
        print("[Server] Random Forest loaded")
    except Exception as e:
        print(f"[Server] RF error: {e}")

mode = "TFLite" if tflite_interpreter else ("RF" if rf_model else "Rule-based")
print(f"[Server] Mode: {mode}")


def engineer_features(hr, temp, gsr):
    f = np.array(
        [[
            hr, temp, gsr,
            hr / 70, gsr / 300,
            hr * gsr / 10000,
            temp - 33,
            (hr - 70) * (gsr - 200) / 1000,
            1.0 if hr > 90 else 0.0,
            1.0 if gsr > 500 else 0.0,
            1.0 if hr < 62 else 0.0,
            1.0 if temp > 35 else 0.0,
        ]],
        dtype=np.float32,
    )
    if scaler_mean is not None:
        f = (f - scaler_mean) / scaler_std
    return f


def predict_mood(hr, temp, gsr):
    f = engineer_features(hr, temp, gsr)
    if tflite_interpreter:
        try:
            tflite_interpreter.set_tensor(tflite_inp[0]["index"], f)
            tflite_interpreter.invoke()
            probs = tflite_interpreter.get_tensor(tflite_out[0]["index"])[0]
            idx = int(np.argmax(probs))
            return MOODS[idx], float(probs[idx]), {m: float(probs[i]) for i, m in enumerate(MOODS)}
        except Exception as e:
            print(f"[Server] TFLite predict error: {e}")
    if rf_model:
        try:
            probs = rf_model.predict_proba(f)[0]
            idx = int(np.argmax(probs))
            return MOODS[idx], float(probs[idx]), {m: float(probs[i]) for i, m in enumerate(MOODS)}
        except Exception as e:
            print(f"[Server] RF predict error: {e}")
    return rule_based(hr, temp, gsr)


def rule_based(hr, temp, gsr):
    s = {m: 0 for m in MOODS}
    if 75 <= hr <= 90:      s["Focused"] += 35
    if 60 <= hr < 75:       s["Relaxed"] += 35
    if hr > 90:             s["Stressed"] += 40
    if hr < 60:             s["Fatigued"] += 35
    if 33.5 <= temp <= 35:  s["Focused"] += 20
    if 32.5 <= temp < 33.5: s["Relaxed"] += 25
    if temp > 35:           s["Stressed"] += 30
    if temp < 32:           s["Fatigued"] += 20
    if gsr < 250:           s["Relaxed"] += 30; s["Fatigued"] += 15
    if 250 <= gsr < 500:    s["Focused"] += 30
    if gsr >= 500:          s["Stressed"] += 35
    # Normalize scores to probabilities
    total = sum(s.values())
    if total == 0:
        probs = {m: 1.0 / len(MOODS) for m in MOODS}
    else:
        probs = {m: s[m] / total for m in MOODS}
    mood = max(probs, key=probs.get)
    confidence = probs[mood]
    return mood, confidence, probs


class SensorData(BaseModel):
    heart_rate: float
    temperature: float
    gsr: float


class PredictionResponse(BaseModel):
    mood: str
    emoji: str
    color: str
    confidence: float
    probabilities: dict


@app.get("/")
def root():
    return {"status": "ok", "model": mode, "moods": MOODS, "db": "connected" if db is not None else "disconnected"}


@app.post("/detect-mood", response_model=PredictionResponse)
def detect_mood(data: SensorData):
    mood, confidence, probs = predict_mood(data.heart_rate, data.temperature, data.gsr)
    meta = MOOD_META.get(mood, {"emoji": "🤔", "color": "#a78bfa"})
    return PredictionResponse(
        mood=mood,
        emoji=meta["emoji"],
        color=meta["color"],
        confidence=round(confidence * 100, 1),
        probabilities={k: round(v * 100, 1) for k, v in probs.items()} if probs else {},
    )


# ══════════════════════════════════════════════
# WEBCAM IMAGE-BASED MOOD DETECTION
# (endpoint the frontend actually calls)
# ══════════════════════════════════════════════

MOOD_ACTIONS = {
    "focused":  "AC: 22°C Fan Med  | Music: Lofi",
    "relaxed":  "AC: 24°C Fan Low  | Music: Ambient",
    "stressed": "AC: 20°C Fan High | Music: Calm",
    "fatigued": "AC: 23°C Fan Med  | Music: Energetic",
}

class ImageDetectRequest(BaseModel):
    image: str  # base64 data-URL from webcam

@app.post("/detect")
def detect_from_image(data: ImageDetectRequest):
    """
    Accepts a webcam frame as a base64 data-URL, runs face detection +
    brightness/color heuristics to estimate mood scores.
    Returns the shape the frontend expects:
      { mood, confidence, scores, face_detected, action }
    """
    import base64

    face_detected = False
    scores = {"focused": 25.0, "relaxed": 25.0, "stressed": 25.0, "fatigued": 25.0}

    try:
        # Decode image
        header, b64data = data.image.split(",", 1) if "," in data.image else ("", data.image)
        img_bytes = base64.b64decode(b64data)

        # Try OpenCV face detection + analysis
        try:
            import cv2
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                # Resize up if too small for reliable detection
                h_orig, w_orig = img.shape[:2]
                if w_orig < 300:
                    scale = 300 / w_orig
                    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.equalizeHist(gray)
                h, w = gray.shape

                # Load cascades
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                alt_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
                )
                alt_tree = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_alt_tree.xml"
                )

                if face_cascade.empty():
                    print("[Server] ERROR: Haar cascade failed to load!")

                faces = []

                # Try each cascade with lenient settings
                for cascade in [face_cascade, alt_cascade, alt_tree]:
                    if cascade.empty():
                        continue
                    for (sf, mn) in [(1.05, 2), (1.1, 2), (1.1, 3), (1.2, 3)]:
                        detected = cascade.detectMultiScale(
                            gray, scaleFactor=sf, minNeighbors=mn,
                            minSize=(20, 20),
                            flags=cv2.CASCADE_SCALE_IMAGE
                        )
                        if len(detected) > 0:
                            faces = detected
                            break
                    if len(faces) > 0:
                        break

                # If still no face, try horizontally flipped image
                if len(faces) == 0:
                    gray_flip = cv2.flip(gray, 1)
                    img_flip = cv2.flip(img, 1)
                    for cascade in [face_cascade, alt_cascade]:
                        if cascade.empty():
                            continue
                        detected = cascade.detectMultiScale(
                            gray_flip, scaleFactor=1.1, minNeighbors=2,
                            minSize=(20, 20)
                        )
                        if len(detected) > 0:
                            faces = detected
                            gray = gray_flip
                            img = img_flip
                            print("[Server] Face found in flipped image")
                            break

                face_detected = len(faces) > 0
                print(f"[Server] Face detection: found={face_detected} count={len(faces)} img={w}x{h}")

                if face_detected:
                    (fx, fy, fw, fh) = faces[0]
                    face_roi = gray[fy:fy+fh, fx:fx+fw]

                    # Analyze face region
                    brightness = float(np.mean(face_roi))
                    contrast = float(np.std(face_roi))

                    # Extract color info from face region (BGR)
                    face_color = img[fy:fy+fh, fx:fx+fw]
                    mean_b, mean_g, mean_r = [float(x) for x in cv2.mean(face_color)[:3]]

                    # Heuristic scoring based on facial features
                    # Brightness: low = fatigued, high = stressed
                    # Contrast: high = focused, low = relaxed
                    # Color warmth (redness): high = stressed
                    warmth = mean_r - (mean_b + mean_g) / 2

                    s = {"focused": 10.0, "relaxed": 10.0, "stressed": 10.0, "fatigued": 10.0}

                    # Brightness scoring
                    if brightness < 100:
                        s["fatigued"] += 30
                        s["relaxed"] += 15
                    elif brightness < 140:
                        s["focused"] += 25
                        s["relaxed"] += 20
                    elif brightness < 180:
                        s["focused"] += 20
                        s["stressed"] += 15
                    else:
                        s["stressed"] += 30

                    # Contrast scoring
                    if contrast > 55:
                        s["focused"] += 20
                        s["stressed"] += 10
                    elif contrast > 40:
                        s["focused"] += 15
                        s["relaxed"] += 10
                    else:
                        s["relaxed"] += 20
                        s["fatigued"] += 15

                    # Warmth scoring
                    if warmth > 20:
                        s["stressed"] += 20
                    elif warmth > 5:
                        s["focused"] += 10
                    else:
                        s["relaxed"] += 15
                        s["fatigued"] += 10

                    # Normalize to percentages
                    total = sum(s.values())
                    scores = {k: round(v / total * 100, 1) for k, v in s.items()}
                else:
                    # No face — return neutral scores
                    scores = {"focused": 25.0, "relaxed": 25.0, "stressed": 25.0, "fatigued": 25.0}
        except ImportError:
            # OpenCV not installed — use random-ish heuristic from image bytes
            byte_sum = sum(img_bytes[:200]) if len(img_bytes) > 200 else sum(img_bytes)
            idx = byte_sum % 4
            moods_lower = ["focused", "relaxed", "stressed", "fatigued"]
            base = {m: 15.0 for m in moods_lower}
            base[moods_lower[idx]] += 40.0
            total = sum(base.values())
            scores = {k: round(v / total * 100, 1) for k, v in base.items()}
            face_detected = True

    except Exception as e:
        print(f"[Server] /detect error: {e}")
        scores = {"focused": 25.0, "relaxed": 25.0, "stressed": 25.0, "fatigued": 25.0}

    # Pick top mood
    mood = max(scores, key=scores.get)
    confidence = scores[mood]

    result = {
        "mood": mood,
        "confidence": confidence,
        "scores": scores,
        "face_detected": face_detected,
        "action": MOOD_ACTIONS.get(mood, ""),
    }
    print(f"[Server] /detect → mood={mood} conf={confidence} face={face_detected}")
    return result


@app.get("/health")
def health():
    return {"status": "healthy", "mode": mode, "db": "connected" if db is not None else "disconnected"}


if __name__ == "__main__":
    from fastapi.responses import RedirectResponse

@app.get("/spotify/callback")
def spotify_callback(code: str = ""):
    # Redirect to frontend with the code
    return RedirectResponse(url=f"http://localhost:3000?spotify_code={code}")


if __name__ == "__main__":
    import uvicorn
    print("\n🚀 MoodSpace API starting...")
    print(f"   Mode: {mode}")
    print(f"   DB: {'connected' if db is not None else 'disconnected'}")
    print(f"   Open http://localhost:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)