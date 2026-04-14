

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

classes = ["fake", "real"]
mtcnn = MTCNN(keep_all=False, device="cpu")
# ═════════════════════════════════════════════════════════════════════════════
# HELPERS 
# ═════════════════════════════════════════════════════════════════════════════

def expand_box(x1, y1, x2, y2, w, h, margin_ratio=0.25):
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * margin_ratio), int(bh * margin_ratio)
    return max(0, x1-mx), max(0, y1-my), min(w, x2+mx), min(h, y2+my)


def video_to_frames(video_path, num_frames=16, img_size=224,
                    frame_skip=10, margin_ratio=0.25):
    cap = cv2.VideoCapture(video_path)
    frames, frame_id = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id % frame_skip == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = mtcnn.detect(rgb)
            if boxes is not None and len(boxes) > 0:
                x1, y1, x2, y2 = map(int, boxes[0])
                h, w, _ = rgb.shape
                x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, margin_ratio)
                face = rgb[y1:y2, x1:x2]
                if face.size > 0:
                    frames.append(cv2.resize(face, (img_size, img_size)))
        frame_id += 1
        if len(frames) == num_frames:
            break
    cap.release()
    if not frames:
        raise ValueError("No faces detected!")
    while len(frames) < num_frames:
        frames.append(frames[-1])
    return np.stack(frames)


val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def preprocess_frames(frames, transform):
    tensors = torch.stack([transform(Image.fromarray(f)) for f in frames])
    return tensors.unsqueeze(0)  # (1, T, C, H, W)