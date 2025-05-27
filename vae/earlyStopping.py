from config import cfg

import torch
import os

# Early stopping mechanism to stop training when validation loss doesn't improve
class EarlyStopping:
    def __init__(self, patience=cfg.early_stopping_patience, min_delta=0, checkpoint_path=None):
        self.patience = patience                          # Number of allowed non-improving epochs
        self.min_delta = min_delta                        # Minimum change to consider as improvement
        self.counter = 0                                  # Count epochs without improvement
        self.best_loss = None                             # Store the best validation loss
        self.checkpoint_path = checkpoint_path            # File path to save the best model

    def check(self, validation_loss, model):
        # If it's the first epoch or validation loss improved
        if self.best_loss is None or validation_loss < self.best_loss - self.min_delta:
            self.best_loss = validation_loss
            self.counter = 0
            if self.checkpoint_path:
                torch.save(model.state_dict(), self.checkpoint_path)  # Save current best model
                print(f"Model saved with validation loss: {validation_loss:.4f}\n")
        else:
            self.counter += 1  # Increase counter if no improvement

        # Return True if stopping criteria met
        return self.counter >= self.patience
