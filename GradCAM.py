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

class GradCAM:
    """
    Grad-CAM for CNN_ViT_BiLSTM.
    
    Architecture flow:
      (B, T, C, H, W)
           ↓  flatten to (B*T, C, H, W)
       EfficientNet-B0 → conv_head → (B*T, 1280, H_feat, W_feat)
           ↓  global_pool + flatten → (B*T, 1280)
       concat ViT CLS → (B*T, 1280+768)
           ↓  reshape → (B, T, 2048)
       BiLSTM → FC → scalar logit
    
    We hook into conv_head to get spatial activation maps per frame.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        self._fwd = target_layer.register_forward_hook(self._save_activation)
        self._bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        # output: (B*T, 1280, H_feat, W_feat)
        self.activations = output.detach().clone()

    def _save_gradient(self, module, grad_input, grad_output):
        # grad_output[0]: (B*T, 1280, H_feat, W_feat)
        self.gradients = grad_output[0].detach().clone()

    def generate(self, input_tensor, target="fake"):
        """
        Args:
            input_tensor : (1, T, C, H, W) on device, requires_grad=True
            target       : "fake" or "real" — which class to explain

        Returns:
            cam_per_frame : list of T numpy arrays (H, W) in [0, 1]
            prob_fake     : float
            prob_real     : float
            pred_label    : str
        """
        self.model.zero_grad()

        # Forward pass
        logit = self.model(input_tensor)          # (1, 1)
        prob_real = torch.sigmoid(logit).item()
        prob_fake = 1.0 - prob_real
        pred_label = "real" if prob_real > 0.5 else "fake"

        # Backward pass
        # High logit → REAL, Low logit → FAKE
        # To explain FAKE: minimize logit (negate), to explain REAL: maximize
        if target == "fake":
            logit.backward(torch.ones_like(logit) * -1)  # gradient toward fake
        else:
            logit.backward(torch.ones_like(logit))        # gradient toward real

        # ── Compute CAM ──────────────────────────────────────────────────────
        grads = self.gradients    # (B*T, 1280, H_feat, W_feat)
        acts  = self.activations  # (B*T, 1280, H_feat, W_feat)

        # Global average pool gradients → importance weights per channel
        weights = grads.mean(dim=(2, 3), keepdim=True)  # (B*T, 1280, 1, 1)

        # Weighted combination of activation maps
        cam = (weights * acts).sum(dim=1)  # (B*T, H_feat, W_feat)
        cam = F.relu(cam)                  # only keep positive influence

        # Normalize per frame to [0, 1]
        cam_per_frame = []
        T = cam.shape[0]
        for t in range(T):
            c = cam[t].cpu().numpy()
            c_min, c_max = c.min(), c.max()
            c = (c - c_min) / (c_max - c_min + 1e-8)
            cam_per_frame.append(c)

        return cam_per_frame, prob_fake, prob_real, pred_label

    def remove_hooks(self):
        self._fwd.remove()
        self._bwd.remove()
