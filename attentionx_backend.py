import os
import whisper
import librosa
import numpy as np
import cv2
import mediapipe as mp
import requests
from moviepy.editor import VideoFileClip

# FFmpeg path (keep yours)
os.environ["PATH"] += r";C:\Users\pandian\Downloads\ffmpeg-2026-04-16-git-5abc240a27-essentials_build\ffmpeg-2026-04-16-git-5abc240a27-essentials_build\bin"

# =========================
# TRANSCRIPTION
# =========================
def transcribe_video(video_path):
    print("🎧 Transcribing video...")
    model = whisper.load_model("tiny")
    result = model.transcribe(video_path)
    return result["text"]

# =========================
# GEMINI (SAFE HTTP CALL)
# =========================
def get_gemini_peaks(transcript):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return []

    url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"

    prompt = f"""
    Extract 3 key timestamps (mm:ss format) from this transcript.

    Example output:
    0:30
    1:10
    2:05

    Transcript:
    {transcript[:2000]}
    """

    body = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }

    try:
        res = requests.post(url, json=body)
        data = res.json()

        text = data["candidates"][0]["content"]["parts"][0]["text"]

        lines = text.strip().split("\n")

        peaks = []
        for line in lines:
            if ":" in line:
                peaks.append({"time": line.strip()})

        return peaks[:3]

    except Exception as e:
        print("Gemini failed:", e)
        return []

# =========================
# AUDIO FALLBACK
# =========================
def get_audio_peaks(video_path):
    y, sr = librosa.load(video_path)

    length = len(y)

    peaks = []
    for i in [0.2, 0.5, 0.8]:
        sec = int((length * i) / sr)
        peaks.append({"time": f"{sec//60}:{sec%60:02d}"})

    return peaks

# =========================
# MEDIAPIPE FACE CROP
# =========================
def smart_crop(video_path, start, end, output_path):
    clip = VideoFileClip(video_path).subclip(start, end)

    temp = "temp.mp4"
    clip.write_videofile(temp, verbose=False, logger=None)

    cap = cv2.VideoCapture(temp)

    mp_face = mp.solutions.face_detection
    face = mp_face.FaceDetection(min_detection_confidence=0.5)

    centers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face.process(rgb)

        if result.detections:
            box = result.detections[0].location_data.relative_bounding_box
            cx = int((box.xmin + box.width / 2) * w)
            centers.append(cx)

    cap.release()

    if not centers:
        centers = [clip.w // 2]

    avg_center = int(sum(centers) / len(centers))

    target_w = int(clip.h * 9 / 16)
    x1 = max(0, avg_center - target_w // 2)
    x2 = x1 + target_w

    final = clip.crop(x1=x1, y1=0, x2=x2, y2=clip.h).resize((1080, 1920))

    final.write_videofile(output_path, verbose=False, logger=None)

    clip.close()
    final.close()

    return output_path

# =========================
# FALLBACK CROP
# =========================
def smart_crop_fallback(video_path, start, end, output_path):
    clip = VideoFileClip(video_path).subclip(start, end)

    target_w = int(clip.h * 9 / 16)
    x_center = clip.w // 2

    x1 = max(0, x_center - target_w // 2)
    x2 = x1 + target_w

    final = clip.crop(x1=x1, y1=0, x2=x2, y2=clip.h).resize((1080, 1920))

    final.write_videofile(output_path, verbose=False, logger=None)

    clip.close()
    final.close()

    return output_path

# =========================
# MAIN PIPELINE
# =========================
def process_video_full_pipeline(video_path, api_key=None):
    print("\n🚀 Starting Pipeline...\n")

    transcript = transcribe_video(video_path)

    # Try Gemini first
    peaks = get_gemini_peaks(transcript)

    if not peaks:
        print("⚠️ Using audio fallback")
        peaks = get_audio_peaks(video_path)

    clips = []

    for i, p in enumerate(peaks[:3]):
        m, s = map(int, p["time"].split(":"))
        start = m * 60 + s
        end = start + 10

        output = f"clip_{i}.mp4"

        try:
            path = smart_crop(video_path, start, end, output)
        except Exception as e:
            print("⚠️ Crop fallback:", e)
            path = smart_crop_fallback(video_path, start, end, output)

        clips.append({
            "clip_number": i + 1,
            "path": path
        })

    return {
        "transcript": transcript,
        "peaks": peaks,
        "clips": clips
    }