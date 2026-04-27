"""Deep learning models with sklearn-compatible wrappers.

Provides MLP and LSTM classifiers that work seamlessly with
sklearn's clone(), fit(), predict(), predict_proba() interface,
enabling integration with Purged K-Fold CV and Optuna tuning.
"""

import numpy as np
import torch
from torch import nn
from skorch import NeuralNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin


# ---------------------------------------------------------------------------
# PyTorch modules
# ---------------------------------------------------------------------------

class MLPModule(nn.Module):
    """Feedforward network: [Linear → BatchNorm → ReLU → Dropout] × N."""

    def __init__(self, input_dim=10, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        layers = []
        in_d = input_dim
        for _ in range(n_layers):
            layers.extend([
                nn.Linear(in_d, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_d = hidden_dim
        layers.append(nn.Linear(hidden_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)


class LSTMModule(nn.Module):
    """LSTM network: processes sequences and classifies from last hidden state."""

    def __init__(self, input_dim=10, hidden_dim=32, n_lstm_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, X):
        # X: (batch, seq_len, features)
        _, (h_n, _) = self.lstm(X)
        return self.classifier(h_n[-1])


# ---------------------------------------------------------------------------
# Sklearn-compatible wrappers
# ---------------------------------------------------------------------------

class SklearnMLPClassifier(BaseEstimator, ClassifierMixin):
    """MLP classifier with full sklearn compatibility (clone, fit, predict)."""

    def __init__(self, input_dim=10, hidden_dim=64, n_layers=2,
                 dropout=0.3, lr=1e-3, max_epochs=100,
                 batch_size=32, random_state=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.net_ = NeuralNetClassifier(
            module=MLPModule,
            module__input_dim=self.input_dim,
            module__hidden_dim=self.hidden_dim,
            module__n_layers=self.n_layers,
            module__dropout=self.dropout,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            train_split=None,
            verbose=0,
            device="cpu",
        )
        X_np = np.asarray(X, dtype=np.float32)
        y_np = np.asarray(y, dtype=np.int64)
        self.net_.fit(X_np, y_np)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return self.net_.predict(np.asarray(X, dtype=np.float32))

    def predict_proba(self, X):
        return self.net_.predict_proba(np.asarray(X, dtype=np.float32))


class SklearnLSTMClassifier(BaseEstimator, ClassifierMixin):
    """LSTM classifier that internally creates sliding-window sequences."""

    def __init__(self, input_dim=10, hidden_dim=32, n_lstm_layers=1,
                 seq_len=5, dropout=0.2, lr=1e-3, max_epochs=100,
                 batch_size=32, random_state=42):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_lstm_layers = n_lstm_layers
        self.seq_len = seq_len
        self.dropout = dropout
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state

    def _make_sequences(self, X):
        """Convert (n, features) → (n, seq_len, features) with zero-padding."""
        X_np = np.asarray(X, dtype=np.float32)
        n, d = X_np.shape
        X_seq = np.zeros((n, self.seq_len, d), dtype=np.float32)
        for i in range(n):
            start = max(0, i - self.seq_len + 1)
            window = X_np[start:i + 1]
            X_seq[i, self.seq_len - len(window):] = window
        return X_seq

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        self.net_ = NeuralNetClassifier(
            module=LSTMModule,
            module__input_dim=self.input_dim,
            module__hidden_dim=self.hidden_dim,
            module__n_lstm_layers=self.n_lstm_layers,
            module__dropout=self.dropout,
            criterion=nn.CrossEntropyLoss,
            optimizer=torch.optim.Adam,
            lr=self.lr,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            train_split=None,
            verbose=0,
            device="cpu",
        )
        X_seq = self._make_sequences(X)
        y_np = np.asarray(y, dtype=np.int64)
        self.net_.fit(X_seq, y_np)
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return self.net_.predict(self._make_sequences(X))

    def predict_proba(self, X):
        return self.net_.predict_proba(self._make_sequences(X))
