import torch
import torch.nn as nn
import timm

class CNN_ViT_BiLSTM(nn.Module):
    def __init__(self, cnn_model='efficientnet_b0', vit_model='vit_base_patch16_224', lstm_hidden=256, lstm_layers=1):
        super(CNN_ViT_BiLSTM, self).__init__()

        # Use the checkpoint weights we load later instead of fetching backbone weights at runtime.
        self.cnn = timm.create_model(cnn_model, pretrained=False)
        self.cnn.reset_classifier(0)  # remove classifier to get features
        cnn_feature_dim = self.cnn.num_features

        # Freeze CNN
        for param in self.cnn.parameters():
            param.requires_grad = False

        self.vit = timm.create_model(vit_model, pretrained=False)
        self.vit.reset_classifier(0)
        vit_feature_dim = self.vit.num_features

         # Freeze ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        # BiLSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=cnn_feature_dim + vit_feature_dim, # Corrected input size
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True
        )


        # Fully connected layer for binary output
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden*2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        """
        x: shape (B, T, C, H, W)
        B = batch size, T = number of frames per video
        """
        B, T, C, H, W = x.shape
        x_flat = x.view(B*T, C, H, W) # Flatten B and T dimensions for CNN and ViT processing

        # CNN feature extraction from original images
        cnn_raw_feat = self.cnn.forward_features(x_flat)
        # Apply global pooling and flatten for CNN
        cnn_feat = self.cnn.global_pool(cnn_raw_feat).flatten(1)

        # ViT feature extraction from original images (parallel processing)
        vit_raw_feat = self.vit.forward_features(x_flat)
        # Extract CLS token for ViT
        vit_feat = vit_raw_feat[:, 0]

        # Concatenate features
        combined_feat = torch.cat((cnn_feat, vit_feat), dim=1) # Combine features

        # Reshape to sequence for LSTM
        seq_feat = combined_feat.view(B, T, -1)

        # BiLSTM
        lstm_out, _ = self.lstm(seq_feat)  # lstm_out: (B, T, 2*hidden)
        lstm_last = lstm_out[:, -1, :]     # take last frame output

        # FC layer
        out = self.fc(lstm_last)
        return out
