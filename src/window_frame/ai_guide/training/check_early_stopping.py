"""Stop on a validation-loss plateau, never during the preparatory stages."""
import math


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0001, start_epoch=0):
        if not isinstance(patience, int) or isinstance(patience, bool) or patience < 0:
            raise ValueError('early_stopping_patience must be an integer >=0 (0 disables it)')
        if not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('early_stopping_min_delta must be finite and >=0')
        self.patience, self.min_delta, self.start_epoch = patience, min_delta, start_epoch
        self.best = None
        self.bad_epochs = 0

    def update(self, epoch, validation_loss):
        if not math.isfinite(validation_loss):
            raise ValueError('Early stopping requires finite validation loss')
        if not self.patience or epoch < self.start_epoch:
            return False
        if self.best is None or validation_loss < self.best - self.min_delta:
            self.best = validation_loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state_dict(self):
        return dict(patience=self.patience, min_delta=self.min_delta, start_epoch=self.start_epoch,
                    best=self.best, bad_epochs=self.bad_epochs)

    def restore(self, saved):
        """Only restore counters if the stopping policy hasn't changed."""
        if not saved or any(saved.get(key) != getattr(self, key) for key in ('patience', 'min_delta', 'start_epoch')):
            return False
        self.best, self.bad_epochs = saved['best'], saved['bad_epochs']
        return True
