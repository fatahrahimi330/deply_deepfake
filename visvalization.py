# ═════════════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ═════════════════════════════════════════════════════════════════════════════
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from CNN_ViT_BiLSTM import CNN_ViT_BiLSTM
import matplotlib.pyplot as plt
import os


def overlay_heatmap(frame_rgb, cam, alpha=0.45):
    """Overlay Grad-CAM heatmap on a frame."""
    h, w = frame_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    heatmap_bgr = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    blended = (alpha * heatmap_rgb + (1 - alpha) * frame_rgb).astype(np.uint8)
    return blended, cam_resized


def save_gradcam_grid(frames, cam_list, prob_fake, prob_real, pred_label,
                      output_path="gradcam_output.png"):
    """
    Two-row grid:
      Row 1 — original face frames
      Row 2 — Grad-CAM overlays with per-frame activation score
    """
    T = len(frames)
    fig, axes = plt.subplots(2, T, figsize=(T * 2.5, 6))
    fig.patch.set_facecolor("#111111")

    color = "#ff4444" if pred_label == "fake" else "#44dd88"
    fig.suptitle(
        f"Prediction: {pred_label.upper()}    "
        f"Fake {prob_fake:.1%}  |  Real {prob_real:.1%}",
        color=color, fontsize=13, fontweight="bold", y=1.02
    )

    for t in range(T):
        overlay, cam_up = overlay_heatmap(frames[t], cam_list[t])
        act_score = cam_up.mean()

        # Original frame
        ax = axes[0, t]
        ax.imshow(frames[t])
        ax.set_title(f"F{t+1}", color="white", fontsize=7)
        ax.axis("off")

        # Grad-CAM overlay
        ax = axes[1, t]
        ax.imshow(overlay)
        ax.set_title(
            f"{act_score:.2f}",
            color="tomato" if act_score > 0.35 else "gray",
            fontsize=7
        )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[✓] Saved grid → {output_path}")


def save_top_suspicious_frames(frames, cam_list, prob_fake, pred_label,
                                top_k=3, out_dir="gradcam_top"):
    """Save top-k most activated frames side-by-side (original | heatmap)."""
    os.makedirs(out_dir, exist_ok=True)
    scores = [cam.mean() for cam in cam_list]
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    for rank, idx in enumerate(top_idx):
        overlay, _ = overlay_heatmap(frames[idx], cam_list[idx])
        color = "#ff4444" if pred_label == "fake" else "#44dd88"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.5))
        fig.patch.set_facecolor("#111111")
        fig.suptitle(
            f"{pred_label.upper()}  |  Frame {idx+1}  |  "
            f"Activation: {scores[idx]:.3f}  |  Fake prob: {prob_fake:.1%}",
            color=color, fontsize=10
        )
        ax1.imshow(frames[idx]); ax1.set_title("Original", color="white", fontsize=9); ax1.axis("off")
        ax2.imshow(overlay);     ax2.set_title("Grad-CAM", color="white", fontsize=9); ax2.axis("off")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"rank{rank+1}_frame{idx+1}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()
        print(f"[✓] Saved top frame {rank+1} → {out_path}")