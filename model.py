import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, padding=4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout1d(0.2)

    def forward(self, x):
        return self.dropout(self.relu(self.bn(self.conv(x))))


class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBnRelu1d(in_channels, out_channels)
        self.conv2 = ConvBnRelu1d(out_channels, out_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        features = self.conv2(self.conv1(x))
        return features, self.pool(features)


class StackDecoder3p(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv1d(channels, skip_channels, kernel_size=9, padding=4)
            for channels in in_channels
        ])
        self.aggregate = ConvBnRelu1d(skip_channels * 5, out_channels)

    def forward(self, *features):
        projected = [projection(feature) for projection, feature in zip(self.projections, features)]
        return self.aggregate(torch.cat(projected, dim=1))


class ECGUNet3p(nn.Module):
    def __init__(self, n_channels=32, n_classes=4):
        super().__init__()
        filters = [n_channels * (2 ** level) for level in range(5)]
        self.filters = filters
        skip_channels = filters[0]
        decoder_channels = skip_channels * 5

        self.down1 = StackEncoder(1, filters[0])
        self.down2 = StackEncoder(filters[0], filters[1])
        self.down3 = StackEncoder(filters[1], filters[2])
        self.down4 = StackEncoder(filters[2], filters[3])
        self.middle = nn.Sequential(
            ConvBnRelu1d(filters[3], filters[4]),
            ConvBnRelu1d(filters[4], filters[4]),
        )

        self.up4 = StackDecoder3p(filters, skip_channels, decoder_channels)
        self.up3 = StackDecoder3p(filters[:3] + [decoder_channels] + filters[4:], skip_channels, decoder_channels)
        self.up2 = StackDecoder3p(filters[:2] + [decoder_channels] * 2 + filters[4:], skip_channels, decoder_channels)
        self.up1 = StackDecoder3p(filters[:1] + [decoder_channels] * 3 + filters[4:], skip_channels, decoder_channels)
        self.segment = nn.Conv1d(decoder_channels, n_classes, kernel_size=1)

    def forward_encoder(self, x):
        enc1, x = self.down1(x)
        enc2, x = self.down2(x)
        enc3, x = self.down3(x)
        enc4, x = self.down4(x)
        enc5 = self.middle(x)
        return enc1, enc2, enc3, enc4, enc5

    def forward_decoder(self, enc1, enc2, enc3, enc4, enc5):
        dec5 = enc5
        dec4 = self.up4(
            F.max_pool1d(enc1, kernel_size=8, stride=8),
            F.max_pool1d(enc2, kernel_size=4, stride=4),
            F.max_pool1d(enc3, kernel_size=2, stride=2),
            enc4,
            F.interpolate(dec5, size=enc4.shape[-1], mode='linear', align_corners=False),
        )
        dec3 = self.up3(
            F.max_pool1d(enc1, kernel_size=4, stride=4),
            F.max_pool1d(enc2, kernel_size=2, stride=2),
            enc3,
            F.interpolate(dec4, size=enc3.shape[-1], mode='linear', align_corners=False),
            F.interpolate(dec5, size=enc3.shape[-1], mode='linear', align_corners=False),
        )
        dec2 = self.up2(
            F.max_pool1d(enc1, kernel_size=2, stride=2),
            enc2,
            F.interpolate(dec3, size=enc2.shape[-1], mode='linear', align_corners=False),
            F.interpolate(dec4, size=enc2.shape[-1], mode='linear', align_corners=False),
            F.interpolate(dec5, size=enc2.shape[-1], mode='linear', align_corners=False),
        )
        dec1 = self.up1(
            enc1,
            F.interpolate(dec2, size=enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(dec3, size=enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(dec4, size=enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(dec5, size=enc1.shape[-1], mode='linear', align_corners=False),
        )
        return self.segment(dec1)

    def forward(self, x, **kwargs):
        return self.forward_decoder(*self.forward_encoder(x))


class ECGSegmenterFeatureClassifier(nn.Module):
    """P-present classifier over explicit segmenter-derived window features.

    This is the cross-task classifier path: the segmenter remains responsible
    for dense morphology, while this small MLP learns the beat-level
    P-present/P-absent decision from pooled segmentation evidence.
    """

    def __init__(self, feature_dim=13):
        super().__init__()
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.1),
            nn.Linear(16, 2),
        )

    def forward(self, features):
        return self.net(features)


class ECGSegmenterTokenMILClassifier(nn.Module):
    """Attention-MIL classifier over regional segmenter evidence tokens."""

    def __init__(self, token_dim=8, context_dim=15, hidden_dim=32, attn_dim=16):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.attn_v = nn.Linear(hidden_dim, attn_dim)
        self.attn_u = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1, bias=False)
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.1),
            nn.Linear(16, 2),
        )

    def forward(self, tokens, context):
        token_hidden = self.token_proj(tokens)
        gated = torch.tanh(self.attn_v(token_hidden)) * torch.sigmoid(self.attn_u(token_hidden))
        attn_logits = self.attn_w(gated).squeeze(-1)
        attn = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(token_hidden * attn.unsqueeze(-1), dim=1)
        context_hidden = self.context_proj(context)
        return self.head(torch.cat([pooled, context_hidden], dim=1))


class ECGRecordBagMILClassifier(nn.Module):
    """Record-level attention-MIL classifier over beat evidence vectors."""

    def __init__(self, instance_dim, hidden_dim=64, attn_dim=32):
        super().__init__()
        self.instance_proj = nn.Sequential(
            nn.LayerNorm(instance_dim),
            nn.Linear(instance_dim, hidden_dim),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.attn_v = nn.Linear(hidden_dim, attn_dim)
        self.attn_u = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1, bias=False)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.2),
            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(p=0.1),
            nn.Linear(16, 2),
        )

    def forward(self, instances, mask, return_attention=False):
        hidden = self.instance_proj(instances)
        gated = torch.tanh(self.attn_v(hidden)) * torch.sigmoid(self.attn_u(hidden))
        attn_logits = self.attn_w(gated).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(hidden * attn.unsqueeze(-1), dim=1)
        logits = self.head(pooled)
        if return_attention:
            return logits, attn
        return logits
