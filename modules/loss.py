import torch
import torch.nn as nn
import torch.nn.functional as F


class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Computes the language model loss using a masked cross entropy approach.
        
        Args:
            input (torch.Tensor): The logits output from the model (shape: [batch_size, seq_len, vocab_size]).
            target (torch.Tensor): The target indices for language modeling (shape: [batch_size, seq_len]).
            mask (torch.Tensor): A mask to ignore padding tokens (shape: [batch_size, seq_len]).

        Returns:
            torch.Tensor: The average loss over the non-padded tokens.
        """
        # Truncate target and mask to the same length as the input sequence
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        # Gather the predictions corresponding to the target token indices
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask

        # Calculate the average loss, ignoring padding tokens
        output = torch.sum(output) / torch.sum(mask)

        return output


class SmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        """
        Initializes the smoothed cross-entropy loss with label smoothing.
        
        Args:
            smoothing (float): The smoothing factor for the loss. Default is 0.1.
        """
        super(SmoothedCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Computes the smoothed cross-entropy loss.
        
        Args:
            pred (torch.Tensor): The predicted logits (shape: [batch_size, seq_len, vocab_size]).
            target (torch.Tensor): The target labels (shape: [batch_size, seq_len]).

        Returns:
            torch.Tensor: The computed loss value.
        """
        # Compute log probabilities from the predictions
        log_prob = F.log_softmax(pred, dim=-1)

        # Compute negative log likelihood loss
        nll_loss = -log_prob.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)

        # Compute the smooth loss
        smooth_loss = -log_prob.mean(dim=-1)

        # Combine both losses with label smoothing
        loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

        return loss.mean()


def compute_loss(output: torch.Tensor, reports_ids: torch.Tensor, reports_masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the loss for a given output, report ids, and masks using language model criterion.
    
    Args:
        output (torch.Tensor): The predicted logits (shape: [batch_size, seq_len, vocab_size]).
        reports_ids (torch.Tensor): The target token ids (shape: [batch_size, seq_len]).
        reports_masks (torch.Tensor): The mask to ignore padding tokens (shape: [batch_size, seq_len]).

    Returns:
        torch.Tensor: The loss computed by the LanguageModelCriterion.
    """
    criterion = LanguageModelCriterion()
    loss = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    return loss
