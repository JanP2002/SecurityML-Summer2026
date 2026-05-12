"""
lib.py — Shared data, model, training, and evaluation utilities.
================================================================

Both step1.py and step2.py import from this module.  It contains:

    Data loaders
        prepare_clean_data   — download MNIST / Fashion-MNIST as clean class
        load_pythia_data     — load hidden Pythia dataset from PNG folders
        split_train_test     — random 80/20 split helper

    Visualisation
        visualize_samples    — render a row of labelled image samples

    DataLoader
        make_dataloader      — create a DataLoader with GPU-optimised settings

    Model
        AnomalyCNN           — 3-block CNN binary anomaly classifier

    Training
        train_model          — training loop with early stopping

    Evaluation
        evaluate_model       — compute Acc, Prec, Rec, F1, AUC-ROC

    Result I/O
        parse_results        — convert metric tuple to dict
        save_results         — serialise results dict to JSON

    Pythia utilities
        check_pythia_available — verify dataset directory exists; abort with instructions if not
"""

from __future__ import annotations

import copy
import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Pythia dataset source
# ---------------------------------------------------------------------------
PYTHIA_URL = "https://cs.pwr.edu.pl/talalaj/#mlsec-ch2"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — never opens a window
import matplotlib.pyplot as plt


# ===========================================================================
# DATA LOADING
# ===========================================================================


def prepare_clean_data(
    dataset_name: str = "mnist",
    data_dir: Path | str = Path("./data"),
) -> tuple[TensorDataset, TensorDataset, torch.Tensor, torch.Tensor]:
    """Download and prepare MNIST or Fashion-MNIST as the 'clean' (label 0) class.

    Pixel values are extracted from torchvision's ``.data`` attribute
    (uint8, range [0, 255]) and normalised to float32 [0.0, 1.0] by
    dividing by 255.  A channel dimension is inserted so the shape
    becomes (N, 1, 28, 28) as required by the CNN.

    Parameters
    ----------
    dataset_name : str
        ``'mnist'`` or ``'fashion_mnist'``.
    data_dir : Path | str
        Root path for torchvision download cache (default ``Path('./data')``).

    Returns
    -------
    clean_train_dataset : TensorDataset
        Training images with label = 0.  Shape per image: (1, 28, 28).
    clean_test_dataset : TensorDataset
        Test images with label = 0.
    train_images : torch.Tensor
        Raw normalised training images (N, 1, 28, 28) — returned so
        callers can apply contamination transforms without re-downloading.
    test_images : torch.Tensor
        Raw normalised test images (M, 1, 28, 28).

    Raises
    ------
    ValueError
        If ``dataset_name`` is not ``'mnist'`` or ``'fashion_mnist'``.
    """
    print(f"Rozpoczynam pobieranie i ładowanie zbioru {dataset_name.upper()}...")

    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_name.lower() == "mnist":
        dataset_class = datasets.MNIST
    elif dataset_name.lower() == "fashion_mnist":
        dataset_class = datasets.FashionMNIST
    else:
        raise ValueError(
            "Nieobsługiwany zbiór danych. Wybierz 'mnist' lub 'fashion_mnist'."
        )

    original_train = dataset_class(root=data_dir, train=True, download=True, transform=transform)
    original_test = dataset_class(root=data_dir, train=False, download=True, transform=transform)

    # Normalise uint8 [0,255] → float32 [0,1] and add channel dim
    train_images = original_train.data.float() / 255.0
    train_images = train_images.unsqueeze(1)                       # (N, 1, 28, 28)
    train_labels = torch.zeros(len(train_images), dtype=torch.long)

    test_images = original_test.data.float() / 255.0
    test_images = test_images.unsqueeze(1)                         # (M, 1, 28, 28)
    test_labels = torch.zeros(len(test_images), dtype=torch.long)

    clean_train_dataset = TensorDataset(train_images, train_labels)
    clean_test_dataset = TensorDataset(test_images, test_labels)

    print(
        f"Załadowano {len(clean_train_dataset)} obrazów treningowych "
        f"oraz {len(clean_test_dataset)} testowych.\n"
    )
    return clean_train_dataset, clean_test_dataset, train_images, test_images


def load_pythia_data(
    data_dir: Path | str = Path("./pythia"),
    partition: str = "clean",
) -> TensorDataset:
    """Load a single partition of the hidden Pythia dataset.

    Pythia images are 70×70 px grayscale PNGs stored in sub-folders named
    by partition (``clean``, ``attack_a`` … ``attack_h``).  Files must be
    named numerically (``0.png``, ``1.png``, …).

    ``transforms.ToTensor()`` applied to a PIL 'L'-mode image divides by
    255 automatically, producing float32 (1, 70, 70) tensors in [0, 1].

    Labelling:  ``'clean'`` → 0,  any other partition name → 1 (attack).

    If the folder is missing, 1 000 synthetic noise samples are returned
    so that the rest of the pipeline remains runnable without the archive.

    Parameters
    ----------
    data_dir : Path | str
        Root directory containing partition sub-folders.
    partition : str
        Sub-folder name, e.g. ``'clean'`` or ``'attack_c'``.

    Returns
    -------
    TensorDataset
        (image_tensor, label) pairs.  Image shape: (1, 70, 70).
    """
    folder_path = Path(data_dir) / partition

    if not folder_path.is_dir():
        raise FileNotFoundError(
            f"Pythia partition folder not found: '{folder_path}'\n"
            f"The Pythia dataset must be provided by the user.\n"
            f"Obtain it from: {PYTHIA_URL}\n"
            f"Call check_pythia_available('{data_dir}') before loading any partition."
        )

    print(f"  Ładowanie obrazów z folderu: {folder_path}...")
    image_tensors = []

    # Sort numerically (0.png, 1.png, ...) for deterministic load order
    transform = transforms.ToTensor()

    for img_path in sorted(folder_path.glob("*.png"), key=lambda p: int(p.stem)):
        img = Image.open(img_path).convert("L")   # force grayscale
        image_tensors.append(transform(img))       # (1, 70, 70)

    images = torch.stack(image_tensors)            # (N, 1, 70, 70)
    label_val = 0 if partition == "clean" else 1
    labels = torch.full((len(images),), label_val, dtype=torch.long)

    return TensorDataset(images, labels)


def check_pythia_available(data_dir: Path | str = Path("./pythia")) -> None:
    """Verify the Pythia dataset directory exists; abort with clear instructions if not.

    The Pythia dataset is *not* bundled with this repository.  It must be
    downloaded separately by the user from the course page and extracted so
    that the directory layout matches::

        <data_dir>/clean/0.png
        <data_dir>/attack_a/0.png
        <data_dir>/attack_b/0.png
        ...

    Parameters
    ----------
    data_dir : Path | str
        Root path that should contain the Pythia partition sub-folders
        (default ``Path('./pythia')``).

    Raises
    ------
    SystemExit
        If ``data_dir`` does not exist on disk, prints a prominent error
        message with the download URL and exits with code 1.
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        border = "=" * 65
        print(
            f"\n{border}\n"
            f"  ERROR: Pythia dataset not found at '{data_dir}'\n"
            f"\n"
            f"  The Pythia dataset must be provided by the user.\n"
            f"  Download it from:\n"
            f"    {PYTHIA_URL}\n"
            f"\n"
            f"  After downloading, extract the archive so that the\n"
            f"  directory structure matches:\n"
            f"    {data_dir / 'clean' / '0.png'}  ...\n"
            f"    {data_dir / 'attack_a' / '0.png'}  ...\n"
            f"    {data_dir / 'attack_b' / '0.png'}  ...\n"
            f"    (attack_a … attack_h for Step 2)\n"
            f"{border}\n"
        )
        sys.exit(1)


def make_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = False,
) -> DataLoader:
    """Create a DataLoader with GPU-optimised settings.

    When CUDA is available, enables ``pin_memory`` (allows async DMA
    transfers to GPU) and ``non_blocking`` transfers.  Uses
    ``num_workers=4`` for background prefetching and
    ``persistent_workers=True`` to avoid re-spawning between epochs.
    Falls back to ``num_workers=0`` when CUDA is not present (CPU-only
    inference is rarely bottlenecked by data loading).

    Parameters
    ----------
    dataset : Dataset
        Any PyTorch Dataset.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        Whether to shuffle on each epoch (default ``False``).

    Returns
    -------
    DataLoader
        Optimised DataLoader ready for training or evaluation.
    """
    cuda = torch.cuda.is_available()
    # On Windows (spawn start method) num_workers > 0 causes worker crashes
    # with complex in-memory dataset chains. Data is already in RAM so there
    # is no I/O bottleneck; the GPU compute path benefits from pin_memory alone.
    import os as _os
    num_workers = 0 if _os.name == "nt" else 4
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=cuda,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def split_train_test(
    dataset,
    train_ratio: float = 0.8,
):
    """Randomly split a dataset into training and test subsets.

    Parameters
    ----------
    dataset : Dataset
        Any PyTorch Dataset.
    train_ratio : float
        Fraction for training (default 0.8).  Remainder is the test split.

    Returns
    -------
    train_subset, test_subset
        Two Subsets from ``torch.utils.data.random_split``.

    Notes
    -----
    No seed is set here; for reproducible splits call
    ``torch.manual_seed(42)`` before invoking this function.
    """
    train_len = int(len(dataset) * train_ratio)
    test_len = len(dataset) - train_len
    return random_split(dataset, [train_len, test_len])


# ===========================================================================
# VISUALISATION
# ===========================================================================


def visualize_samples(
    dataset: TensorDataset,
    save_path: Path | str,
    num_samples: int = 5,
    title_prefix: str = "",
) -> None:
    """Save a horizontal strip of sample images (with labels) to disk.

    Images are rendered at 150 DPI in PNG format and written to
    ``save_path``.  Parent directories are created automatically.
    No window is ever opened (Agg backend).

    Parameters
    ----------
    dataset : TensorDataset
        Dataset whose first element is an image tensor (C, H, W) and
        second element is an integer label scalar.
    save_path : Path | str
        Full file path for the output PNG, e.g.
        ``Path('plots/step1/mnist_clean.png')``.
    num_samples : int
        Number of images to include in the strip (default 5).
    title_prefix : str
        String prepended to each subplot title, e.g. ``"MNIST Clean, "``.

    Notes
    -----
    ``image.squeeze()`` removes the channel dimension (C=1) so that
    matplotlib receives a 2-D (H, W) array for grayscale display.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
    for i in range(num_samples):
        image, label = dataset[i]
        axes[i].imshow(image.squeeze(), cmap="gray")
        axes[i].set_title(f"{title_prefix}Label: {label.item()}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] Saved → {save_path}")


# ===========================================================================
# MODEL
# ===========================================================================


class AnomalyCNN(nn.Module):
    """Binary anomaly detection CNN with dynamic input-size support.

    Architecture
    ------------
    Three convolutional blocks, each::

        Conv2d(C_in → C_out, 3×3, padding=1) → ReLU → MaxPool2d(2, 2)

    followed by a two-layer classification head::

        Linear(flatten_size → 128) → ReLU → Linear(128 → 1)  [raw logit]

    The model outputs a **raw logit** :math:`z \in \mathbb{R}`.  Pass it to
    ``torch.nn.BCEWithLogitsLoss`` during training and apply
    ``torch.sigmoid`` in evaluation to convert to a probability in
    :math:`(0, 1)`.  This avoids numerical instability and is safe for
    AMP mixed-precision training on Tensor Core hardware.

    Flatten size is inferred automatically at construction time via a
    dummy forward pass, so the model handles any square input:

    +-----------+------------------+-------------------+
    | H (input) | H' (after 3×pool)| flatten_size      |
    +===========+==================+===================+
    | 28 (MNIST)| 3                | 3 × 3 × 64 = 576  |
    +-----------+------------------+-------------------+
    | 70 (Pythia| 8                | 8 × 8 × 64 = 4096 |
    +-----------+------------------+-------------------+

    Parameters
    ----------
    input_size : int
        Spatial dimension H of the square input image.
    """

    def __init__(self, input_size: int = 28):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten_size = self._get_flatten_size(input_size)

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def _get_flatten_size(self, input_size: int) -> int:
        """Infer flattened feature count by running a dummy tensor through
        the convolutional stack.

        Parameters
        ----------
        input_size : int
            Spatial side length of the square input.

        Returns
        -------
        int
            Total number of scalar features after the last pool layer.
        """
        dummy = torch.zeros(1, 1, input_size, input_size)
        x = self.pool(F.relu(self.conv1(dummy)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Mini-batch of images, shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Anomaly probabilities, shape (B, 1), values in (0, 1).
        """
        x = self.pool(F.relu(self.conv1(x)))   # (B, 16, H/2, W/2)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 32, H/4, W/4)
        x = self.pool(F.relu(self.conv3(x)))   # (B, 64, H/8, W/8)
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        return self.fc2(x)                     # (B, 1)  raw logit


# ===========================================================================
# TRAINING
# ===========================================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 20,
    patience: int = 3,
) -> nn.Module:
    """Train a binary classifier with early stopping on validation BCE loss.

    Algorithm
    ---------
    For each epoch:
      1. **Train phase** — forward pass, BCELoss, back-prop, Adam update.
      2. **Validation phase** — forward only (no_grad), record val loss.
      3. **Early stopping** — if val loss did not improve for ``patience``
         consecutive epochs, restore best weights and halt.

    Best weights are saved via ``copy.deepcopy(state_dict())`` so that
    subsequent epochs cannot corrupt the checkpoint.

    Parameters
    ----------
    model : nn.Module
        Model to train (moved to CUDA automatically if available).
    train_loader : DataLoader
        Batched training data.
    val_loader : DataLoader
        Batched validation data.
    criterion : nn.Module
        Loss function — expected ``nn.BCELoss()``.
    optimizer : Optimizer
        Update rule — expected ``Adam(lr=0.001)``.
    num_epochs : int
        Maximum epochs (default 20).
    patience : int
        Epochs without val-loss improvement before early stopping (default 3).

    Returns
    -------
    nn.Module
        Model with weights restored to the best-val-loss checkpoint.

    Notes
    -----
    Labels are reshaped via ``.view(-1, 1).float()`` so their shape
    (B, 1) matches the model's raw logit output for ``BCEWithLogitsLoss``.
    """
    best_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_no_improve = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    print(f"  [Device: {device} | AMP: {use_amp}]")

    for epoch in range(num_epochs):
        # ------------------------------------------------------------------
        # Training phase
        # ------------------------------------------------------------------
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            # BCELoss: targets must be float with shape (B, 1) matching outputs
            labels = labels.to(device, non_blocking=True).view(-1, 1).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # ------------------------------------------------------------------
        # Validation phase
        # ------------------------------------------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).view(-1, 1).float()
                with torch.amp.autocast(device.type, enabled=use_amp):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoka {epoch + 1:02d}/{num_epochs:02d} | "
            f"Strata (Train): {epoch_train_loss:.4f} | "
            f"Strata (Val):   {epoch_val_loss:.4f}"
        )

        # ------------------------------------------------------------------
        # Early stopping check
        # ------------------------------------------------------------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f" -> Early Stopping! Brak poprawy od {patience} epok. "
                    "Przywracam najlepsze wagi."
                )
                break

    print(f"Najlepsza strata walidacyjna: {best_val_loss:.4f}")
    model.load_state_dict(best_wts)
    return model


# ===========================================================================
# EVALUATION
# ===========================================================================


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float = 0.5,
) -> tuple[float, float, float, float, float]:
    """Evaluate a trained binary classifier and print all metrics.

    Metrics computed
    ----------------
    .. math::
        \\text{Accuracy}  &= \\frac{TP + TN}{TP + TN + FP + FN} \\\\
        \\text{Precision} &= \\frac{TP}{TP + FP} \\\\
        \\text{Recall}    &= \\frac{TP}{TP + FN} \\\\
        \\text{F1}        &= \\frac{2 \\cdot P \\cdot R}{P + R} \\\\
        \\text{AUC-ROC}   &= \\int_0^1 TPR(FPR^{-1}(t))\\, dt

    AUC-ROC is threshold-independent (uses raw probabilities).

    Parameters
    ----------
    model : nn.Module
        Trained binary classifier.
    test_loader : DataLoader
        Labelled test data.
    threshold : float
        Decision threshold τ: predict anomaly if :math:`\\hat{y} \\geq \\tau`
        (default 0.5).

    Returns
    -------
    acc, prec, rec, f1, auc : float
        Metric values.  ``auc`` is ``float('nan')`` when the test set
        contains only one class (degenerate edge case).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_labels: list = []
    all_probs: list = []
    use_amp = device.type == "cuda"

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()

            with torch.amp.autocast(device.type, enabled=use_amp):
                logits = model(inputs).squeeze()

            # Guard: if batch_size == 1, squeeze collapses to scalar
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)

            # Convert raw logits → probabilities in FP32 for metric computation
            probs = torch.sigmoid(logits.float())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= threshold).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = float("nan")

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")

    return acc, prec, rec, f1, auc


# ===========================================================================
# RESULT I/O
# ===========================================================================


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy scalars, arrays, and NaN/Inf."""

    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            # JSON does not support NaN/Inf natively
            return None if (np.isnan(v) or np.isinf(v)) else v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

    def iterencode(self, obj, _one_shot=False):
        # Also handle plain Python float NaN/Inf
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            yield "null"
        else:
            yield from super().iterencode(obj, _one_shot)


def parse_results(res: tuple) -> dict:
    """Convert evaluate_model's return tuple to a named dict.

    Handles both the current 5-element tuple (Acc, Prec, Rec, F1, AUC)
    and legacy 4-element tuples for backwards compatibility.

    Parameters
    ----------
    res : tuple
        Return value of :func:`evaluate_model`.

    Returns
    -------
    dict
        Keys: ``Accuracy``, ``Precision``, ``Recall``, ``F1_Score``,
        ``AUC_ROC``.
    """
    if len(res) == 5:
        return {
            "Accuracy": res[0],
            "Precision": res[1],
            "Recall": res[2],
            "F1_Score": res[3],
            "AUC_ROC": res[4],
        }
    return {
        "Accuracy": res[0],
        "Precision": res[1],
        "Recall": res[2],
        "F1_Score": "Brak danych",
        "AUC_ROC": res[3],
    }


def save_results(
    results_summary: dict,
    output_filename: Path | str = "results.json",
) -> None:
    """Serialise the experiment results dict to a JSON file.

    NumPy scalars, arrays, and NaN/Inf values are handled by the custom
    ``_NumpyEncoder``.  NaN and Inf are written as JSON ``null``.

    Parameters
    ----------
    results_summary : dict
        Nested dict of metric values (may contain numpy scalars).
    output_filename : Path | str
        Target file path.
    """
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4, ensure_ascii=False, cls=_NumpyEncoder)
    print(f"Wyniki zapisano do: {output_filename}")
