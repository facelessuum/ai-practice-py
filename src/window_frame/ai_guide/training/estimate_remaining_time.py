"""Estimate the planned remaining run, including validation (not early stopping)."""
from collections import deque


class RemainingTime:
    def __init__(self, epochs, batches_per_epoch):
        self.epochs = epochs
        self.batches_per_epoch = batches_per_epoch
        self.recent_epochs = deque(maxlen=3)

    def estimate(self, epoch, elapsed, completed_batches):
        if self.recent_epochs:
            duration = sum(self.recent_epochs) / len(self.recent_epochs)
        elif completed_batches:
            # First-epoch approximation treats train/validation batches equally.
            # Completed epochs subsequently include measured validation/checkpoint time.
            duration = elapsed / completed_batches * self.batches_per_epoch
        else:
            return None
        return max(0, duration - elapsed) + max(0, self.epochs - epoch - 1) * duration

    def finish_epoch(self, seconds):
        self.recent_epochs.append(seconds)
        return sum(self.recent_epochs) / len(self.recent_epochs)
