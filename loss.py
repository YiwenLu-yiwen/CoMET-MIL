import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0):
        super().__init__()
        self.gamma = gamma
        
    def forward(self, input, target):
        '''
        input x: (B,C,N)
        target y: (B,N)
        CE_loss -log(y_pred): (B,N)
        y_pred: (B,N)
        '''
        CE_loss = F.cross_entropy(input, target, reduction='none')
        y_pred = torch.exp(-CE_loss)
        Focal_loss = torch.mean((1 - y_pred) ** self.gamma * CE_loss)

        return Focal_loss


class DistillationLoss(nn.Module):
    """Soft knowledge distillation via KL divergence.

    Only samples where the teacher's confidence (max softmax prob) exceeds
    ``threshold`` contribute to the loss.  This prevents low-quality
    pseudo-labels from poisoning the student.

    Teacher outputs are always stop-grad'd inside this loss.

    Parameters
    ----------
    temperature : float
        Softmax temperature T.  Higher T produces softer distributions.
        Typical range: 1.0–4.0.
    threshold : float
        Minimum teacher confidence required for a sample to contribute.
        Typical range: 0.70–0.95.
    """

    def __init__(self, temperature: float = 2.0, threshold: float = 0.85):
        super().__init__()
        self.temperature = temperature
        self.threshold = threshold

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        student_logits : Tensor  [B, C]
        teacher_logits : Tensor  [B, C]  (will be stop-grad'd internally)

        Returns
        -------
        Scalar loss.  Returns 0 if no sample passes the confidence threshold.
        """
        teacher_logits = teacher_logits.detach()
        T = self.temperature

        # Use untempered probabilities for confidence gating. Temperature is
        # meant to soften the distillation target, but using the softened
        # distribution for thresholding can accidentally filter out almost every
        # sample when T > 1.
        teacher_base_probs = F.softmax(teacher_logits, dim=1)
        confidence = teacher_base_probs.max(dim=1).values     # [B]
        mask = confidence > self.threshold

        if mask.sum() == 0:
            return student_logits.sum() * 0.0  # keeps gradient graph alive

        teacher_probs = F.softmax(teacher_logits / T, dim=1)
        student_log_probs = F.log_softmax(student_logits[mask] / T, dim=1)
        kl = F.kl_div(student_log_probs, teacher_probs[mask], reduction='batchmean')
        # Scale by T^2 to preserve gradient magnitude as in Hinton et al. 2015
        return kl * (T ** 2)
