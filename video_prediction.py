# -*- coding: utf-8 -*-

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch

from CNN_ViT_BiLSTM import CNN_ViT_BiLSTM
from GradCAM import GradCAM
from helper import preprocess_frames, val_transform, video_to_frames
from visvalization import overlay_heatmap


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_model.pth"
LOCAL_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
NUM_FRAMES = 16


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@st.cache_resource(show_spinner=False)
def load_model():
    device = get_device()
    model = CNN_ViT_BiLSTM()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Grad-CAM needs gradients on the final CNN head.
    for param in model.cnn.conv_head.parameters():
        param.requires_grad = True

    return model, device


def predict_video(video_path):
    model, device = load_model()

    frames = video_to_frames(str(video_path), num_frames=NUM_FRAMES)
    input_tensor = preprocess_frames(frames, val_transform).to(device)

    grad_cam = GradCAM(model, target_layer=model.cnn.conv_head)
    try:
        cam_list, prob_fake, prob_real, pred_label = grad_cam.generate(
            input_tensor, target="fake"
        )
    finally:
        grad_cam.remove_hooks()

    scores = [float(cam.mean()) for cam in cam_list]
    return {
        "frames": frames,
        "cam_list": cam_list,
        "prob_fake": prob_fake,
        "prob_real": prob_real,
        "pred_label": pred_label,
        "scores": scores,
    }


def build_gradcam_figure(frames, cam_list, prob_fake, prob_real, pred_label):
    total_frames = len(frames)
    fig, axes = plt.subplots(2, total_frames, figsize=(total_frames * 1.9, 5.5))
    fig.patch.set_facecolor("#111111")

    title_color = "#ff4d4f" if pred_label == "fake" else "#2fbf71"
    fig.suptitle(
        f"Prediction: {pred_label.upper()}   Fake {prob_fake:.1%} | Real {prob_real:.1%}",
        color=title_color,
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for idx in range(total_frames):
        overlay, cam_resized = overlay_heatmap(frames[idx], cam_list[idx])
        activation_score = cam_resized.mean()

        top_ax = axes[0, idx]
        top_ax.imshow(frames[idx])
        top_ax.set_title(f"F{idx + 1}", color="white", fontsize=8)
        top_ax.axis("off")

        bottom_ax = axes[1, idx]
        bottom_ax.imshow(overlay)
        bottom_ax.set_title(
            f"{activation_score:.2f}",
            color="tomato" if activation_score > 0.35 else "lightgray",
            fontsize=8,
        )
        bottom_ax.axis("off")

    plt.tight_layout()
    return fig


def build_top_frame_figures(frames, cam_list, prob_fake, pred_label, top_k=3):
    ranked_indices = sorted(
        range(len(cam_list)),
        key=lambda idx: float(cam_list[idx].mean()),
        reverse=True,
    )[:top_k]

    figures = []
    for rank, idx in enumerate(ranked_indices, start=1):
        overlay, _ = overlay_heatmap(frames[idx], cam_list[idx])
        score = float(cam_list[idx].mean())

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
        fig.patch.set_facecolor("#111111")
        fig.suptitle(
            (
                f"Rank {rank} | Frame {idx + 1} | "
                f"Activation {score:.3f} | Fake prob {prob_fake:.1%}"
            ),
            color="#ff4d4f" if pred_label == "fake" else "#2fbf71",
            fontsize=11,
        )

        axes[0].imshow(frames[idx])
        axes[0].set_title("Original", color="white", fontsize=9)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM", color="white", fontsize=9)
        axes[1].axis("off")

        plt.tight_layout()
        figures.append(fig)

    return figures


def save_uploaded_video(uploaded_file):
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def get_demo_videos():
    return sorted(
        [
            path
            for path in BASE_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in LOCAL_VIDEO_EXTENSIONS
        ]
    )


def render_sidebar():
    st.sidebar.header("Input Source")
    source = st.sidebar.radio(
        "Choose a video source",
        ("Upload video", "Use local demo video"),
    )

    if source == "Upload video":
        uploaded = st.sidebar.file_uploader(
            "Upload a video",
            type=["mp4", "avi", "mov", "mkv"],
        )
        if uploaded is None:
            return None, None
        saved_path = save_uploaded_video(uploaded)
        return saved_path, uploaded.name

    demo_videos = get_demo_videos()
    if not demo_videos:
        return None, None

    selected_name = st.sidebar.selectbox(
        "Choose a local video",
        [path.name for path in demo_videos],
    )
    selected_path = next(path for path in demo_videos if path.name == selected_name)
    return selected_path, selected_name


def main():
    st.set_page_config(
        page_title="Deepfake Video Prediction",
        page_icon="🎥",
        layout="wide",
    )

    st.title("Deepfake Video Prediction")
    st.caption(
        f"Upload a video or choose a local sample to run the trained model on {NUM_FRAMES} face frames and inspect Grad-CAM explanations."
    )

    if not MODEL_PATH.exists():
        st.error(f"Model file not found: {MODEL_PATH}")
        return

    try:
        _, device = load_model()
    except Exception as exc:
        st.error(f"Failed to load model: {exc}")
        return

    st.sidebar.success(f"Model loaded on `{device}`")
    video_path, video_name = render_sidebar()

    if video_path is None:
        st.info("Select or upload a video from the sidebar to begin.")
        return

    left_col, right_col = st.columns([1.1, 0.9])
    with left_col:
        st.subheader("Video Preview")
        st.video(video_path.read_bytes())

    if st.button("Run Prediction", type="primary", use_container_width=True):
        with st.spinner("Extracting faces, running the model, and generating Grad-CAM..."):
            try:
                result = predict_video(video_path)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                return

        pred_label = result["pred_label"]
        prob_fake = result["prob_fake"]
        prob_real = result["prob_real"]

        with right_col:
            st.subheader("Prediction")
            verdict = "FAKE" if pred_label == "fake" else "REAL"
            verdict_color = "red" if pred_label == "fake" else "green"
            st.markdown(
                f"### <span style='color:{verdict_color}'>{verdict}</span>",
                unsafe_allow_html=True,
            )

            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Fake Probability", f"{prob_fake:.2%}")
            metric_col2.metric("Real Probability", f"{prob_real:.2%}")

            st.caption(f"Source: `{video_name}`")

            score_rows = [
                {
                    "Frame": idx + 1,
                    "Activation Score": round(score, 4),
                }
                for idx, score in enumerate(result["scores"])
            ]
            st.dataframe(score_rows, use_container_width=True, hide_index=True)

        st.subheader("Frame-by-Frame Grad-CAM")
        grid_figure = build_gradcam_figure(
            result["frames"],
            result["cam_list"],
            prob_fake,
            prob_real,
            pred_label,
        )
        st.pyplot(grid_figure, use_container_width=True)
        plt.close(grid_figure)

        st.subheader("Most Suspicious Frames")
        top_figures = build_top_frame_figures(
            result["frames"],
            result["cam_list"],
            prob_fake,
            pred_label,
        )
        for figure in top_figures:
            st.pyplot(figure, use_container_width=True)
            plt.close(figure)


if __name__ == "__main__":
    main()
