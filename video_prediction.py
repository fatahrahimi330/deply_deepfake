import tempfile
from pathlib import Path

import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "weights" / "model.torchscript"
NUM_FRAMES = 16
PREPROCESS = transforms.Compose([
    transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711)),
])


def video_to_frames(path, num_frames):
    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(path)
    frames = []
    frame_id = 0
    try:
        while len(frames) < num_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % 10 == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                boxes = detector.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 5)
                if len(boxes):
                    x, y, w, h = max(boxes, key=lambda box: box[2] * box[3])
                    mx, my = int(w * .25), int(h * .25)
                    face = rgb[max(0, y-my):min(rgb.shape[0], y+h+my),
                               max(0, x-mx):min(rgb.shape[1], x+w+mx)]
                    if face.size:
                        frames.append(cv2.resize(face, (224, 224)))
            frame_id += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("No faces detected in the video.")
    return np.stack(frames)


@st.cache_resource(show_spinner=False)
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(str(MODEL_PATH), map_location=device).eval()
    return model, device


def predict_video(path):
    model, device = load_model()
    frames = video_to_frames(str(path), num_frames=NUM_FRAMES)
    batch = torch.stack([PREPROCESS(Image.fromarray(frame).convert("RGB"))
                         for frame in frames]).to(device)
    with torch.inference_mode():
        probabilities = model(batch).float().softmax(dim=1).cpu()
    return frames, probabilities


def main():
    st.set_page_config(page_title="Deepfake Video Prediction", layout="wide")
    st.title("Deepfake Video Prediction")
    st.caption("The video score is the mean of the model's face-frame probabilities.")
    if not MODEL_PATH.exists():
        st.error("Model not found. Run `git lfs pull` to download it.")
        return
    try:
        _, device = load_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}. Check that Git LFS files were downloaded.")
        return
    st.sidebar.success(f"Model loaded on {device}")
    source = st.sidebar.radio("Video source", ("Upload video", "Use local demo video"))
    temporary_path = None
    if source == "Upload video":
        uploaded = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is None:
            return
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            temporary_path = Path(tmp.name)
        path = temporary_path
    else:
        videos = sorted(p for p in BASE_DIR.iterdir() if p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
        if not videos:
            st.info("No local demo videos found.")
            return
        path = st.sidebar.selectbox("Choose a local video", videos, format_func=lambda p: p.name)
    try:
        st.video(path.read_bytes())
        if st.button("Run Prediction", type="primary"):
            with st.spinner("Extracting faces and scoring frames..."):
                frames, probabilities = predict_video(path)
            real, fake = probabilities.mean(dim=0).tolist()
            st.subheader("FAKE" if fake > real else "REAL")
            cols = st.columns(2)
            cols[0].metric("Fake probability", f"{fake:.2%}")
            cols[1].metric("Real probability", f"{real:.2%}")
            st.dataframe([{"Frame": i + 1, "Fake probability": float(p[1]), "Real probability": float(p[0])}
                          for i, p in enumerate(probabilities)], hide_index=True)
            st.image(list(frames), caption=[f"Frame {i + 1}" for i in range(len(frames))], width=120)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
