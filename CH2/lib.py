"""
lib.py — Shared data, model, training, and evaluation utilities.
================================================================

step1.py, step2.py and step3.py all import from this module.  It contains:

    Data loaders
        prepare_clean_data   — download MNIST / Fashion-MNIST as clean class
        load_pythia_data     — load hidden Pythia dataset from PNG folders
        split_train_test     — random 80/20 split helper

    Visualisation
        visualize_samples    — render a row of labelled image samples

    DataLoader
        make_dataloader      — create a DataLoader with GPU-optimised settings

    Models
        AnomalyCNN           — 3-block CNN binary anomaly classifier (Step 1/2)
        ProfessorCNN         — professor-suggested configurable CNN (Pythia, Steps 1/2/3)
        ConvAutoencoder      — "3+2" convolutional autoencoder (Step 3)
        get_professor_cnn_best — load ProfessorCNN with best searched configuration

    Training
        train_model          — supervised training loop with early stopping
        train_autoencoder    — unsupervised reconstruction training loop

    Evaluation
        evaluate_model       — compute Acc, Prec, Rec, F1, AUC-ROC (classifier)
        reconstruction_scores — per-sample reconstruction-error anomaly scores
        select_threshold     — pick anomaly threshold from clean-only scores
        evaluate_autoencoder — compute Acc, Prec, Rec, F1, AUC-ROC (autoencoder)

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, random_split
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as TF
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

    If the folder is missing, ``FileNotFoundError`` is raised with a message
    pointing to the download URL.  Call ``check_pythia_available()`` first
    to get a clean diagnostic before any partition is loaded.

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
    transfers to GPU).  Uses ``num_workers=4`` with
    ``persistent_workers=True`` on Linux/macOS for background prefetching.
    Always uses ``num_workers=0`` on Windows (``os.name == 'nt'``) because
    the spawn-based multiprocessing start method crashes with in-memory
    dataset chains (``ConcatDataset``, ``Subset``, etc.).

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
# AUGMENTATION
# ===========================================================================


class AugmentedDataset(torch.utils.data.Dataset):
    """Thin wrapper that applies random on-the-fly augmentation to a dataset.

    Only augments the training split.  Pass ``augment=False`` for val/test
    to get deterministic evaluation.

    Augmentations applied when ``augment=True``:
    - Random horizontal flip (p = 0.5)
    - Random vertical flip   (p = 0.5)

    Parameters
    ----------
    base_dataset : Dataset
        Any dataset whose ``__getitem__`` returns ``(image_tensor, label)``.
    augment : bool
        Whether to apply random transforms.  Default False.
    """

    def __init__(
        self,
        base_dataset: torch.utils.data.Dataset,
        augment: bool = False,
    ) -> None:
        self.ds = base_dataset
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, lbl = self.ds[idx]
        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = TF.hflip(img)
            if torch.rand(1).item() < 0.5:
                img = TF.vflip(img)
        return img, lbl


# ===========================================================================
# MODEL
# ===========================================================================


# ===========================================================================
# PYTHIA DETECTOR  (transfer-learning, ResNet18-based)
# ===========================================================================


class PythiaResNet(nn.Module):
    """Transfer-learning binary anomaly detector for small 1-channel datasets.

    Wraps pretrained ResNet18.  Key design choices for the Pythia 70×70
    grayscale dataset:

    1. **1-channel adaptation**: The first Conv2d is replaced with a
       single-channel version whose weights are the channel-wise average of
       the pretrained 3-channel filters.  This preserves the learned edge /
       texture detectors without the scale-doubling bias of channel replication.

    2. **Frozen shallow layers, fine-tuned deep layers**: ``layer1``–``layer3``
       are frozen (pretrained low/mid-level features rarely need adjustment).
       ``layer4`` and the classification head are fine-tuned with a small LR
       (``2e-5`` for the backbone, ``2e-4`` for the head).

    3. **Lightweight head**: ``Flatten → Dropout(0.3) → Linear(512, 1)``.
       A single linear layer on top of ResNet18’s global-average-pooled
       features avoids adding new parameters that would overfit on ~1 000
       training samples.

    Parameters
    ----------
    freeze_shallow : bool
        If True (default), freeze layer1–layer3 and only fine-tune layer4
        plus the head.  Set False to fine-tune the whole network.
    """

    def __init__(self, freeze_shallow: bool = True) -> None:
        super().__init__()

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # --- Adapt first conv to single-channel input -------------------
        # Average the 3 pretrained input filters along the channel axis so
        # the model starts with valid, meaningful edge/texture detectors.
        old_w = base.conv1.weight.data          # (64, 3, 7, 7)
        new_conv = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        new_conv.weight.data = old_w.mean(dim=1, keepdim=True)
        base.conv1 = new_conv

        # --- Build backbone (everything except the final FC layer) -------
        # children(): conv1, bn1, relu, maxpool, layer1–layer4, avgpool, fc
        backbone_layers = list(base.children())[:-1]   # drop the FC
        self.backbone = nn.Sequential(*backbone_layers)  # → (B, 512, 1, 1)

        # --- Freeze shallow layers if requested -------------------------
        if freeze_shallow:
            # layer1 = index 4, layer2 = 5, layer3 = 6 in the sequential
            freeze_up_to = 7  # freeze conv1..layer3 (indices 0–6)
            for i, child in enumerate(self.backbone.children()):
                if i < freeze_up_to:
                    for p in child.parameters():
                        p.requires_grad_(False)

        # --- Classification head -----------------------------------------
        self.head = nn.Sequential(
            nn.Flatten(),      # (B, 512)
            nn.Dropout(p=0.3),
            nn.Linear(512, 1), # raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Grayscale image batch, shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Raw logit, shape (B, 1).
        """
        return self.head(self.backbone(x))        # (B, 1)

    def trainable_param_groups(
        self,
        backbone_lr: float = 2e-5,
        head_lr: float = 2e-4,
    ) -> list[dict]:
        """Return optimizer parameter groups with differential learning rates.

        Parameters
        ----------
        backbone_lr : float
            LR for the unfrozen backbone layers (layer4 by default).
        head_lr : float
            LR for the classification head.

        Returns
        -------
        list[dict]
            Pass directly to ``torch.optim.Adam(param_groups)``.
        """
        backbone_params = [
            p for p in self.backbone.parameters() if p.requires_grad
        ]
        head_params = list(self.head.parameters())
        return [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params,     "lr": head_lr},
        ]


# ===========================================================================
# MODEL  (AnomalyCNN — matches notebook ch2_step1_v2.ipynb exactly)
# ===========================================================================


class AnomalyCNN(nn.Module):
    """Binary anomaly detection CNN with dynamic input-size support.

    Exactly matches the architecture from the reference notebook:

    Three convolutional blocks, each::

        Conv2d → ReLU → MaxPool2d(2, 2)

    Followed by Flatten and two Dense layers::

        Linear(flatten_size → 128) → ReLU → Linear(128 → 1) → Sigmoid

    The model outputs a **probability** in [0, 1] — use ``BCELoss`` for
    training (not BCEWithLogitsLoss, as sigmoid is already applied).

    Parameters
    ----------
    input_size : int
        Spatial side-length H of the square input (28 for MNIST, 70 for Pythia).
    """

    def __init__(self, input_size: int = 28):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1,  out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dynamically infer the flattened feature size so the same class
        # works for both 28×28 (MNIST) and 70×70 (Pythia) without manual maths.
        self.flatten_size = self._get_flatten_size(input_size)

        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 1)

    def _get_flatten_size(self, input_size: int) -> int:
        """Pass a dummy tensor through the conv stack to infer flatten size."""
        dummy = torch.zeros(1, 1, input_size, input_size)
        x = self.pool(F.relu(self.conv1(dummy)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.flatten_size)      # flatten
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))         # probability in [0, 1]
        return x


# ===========================================================================
# PROFESSOR CNN  (Pythia — professor-suggested configurable architecture)
# ===========================================================================


def _get_activation(name: str) -> nn.Module:
    """Return a fresh activation-function module by name.

    Parameters
    ----------
    name : str
        One of ``'leaky_relu'``, ``'swish'``, ``'gelu'``.

    Returns
    -------
    nn.Module
        A newly constructed activation layer.
    """
    name = name.lower()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01, inplace=True)
    elif name in ("swish", "silu"):
        return nn.SiLU()          # SiLU(x) = x \u00b7 \u03c3(x)  \u2261  Swish
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(
            f"Unknown activation '{name}'.  "
            "Choose: 'leaky_relu', 'swish', 'gelu'."
        )


class ProfessorCNN(nn.Module):
    """Professor-suggested CNN binary classifier for the Pythia 70\u00d770 dataset.

    Architecture (whiteboard sketch)::

        Input
          \u2193
        3 \u00d7 [Conv2d \u2192 Activation \u2192 Dropout2d \u2192 MaxPool(2,2)]
          \u2193
        GlobalAveragePooling  |  Flatten
          \u2193
        Linear(\u2192 dense[0]) \u2192 Activation \u2192 Dropout
          \u2193
        Linear(\u2192 dense[1]) \u2192 Activation \u2192 Dropout
          \u2193
        Linear(\u2192 1) \u2192 Sigmoid \u2192 probability \u2208 (0, 1)

    All convolutions use ``padding='same'`` so the spatial dimensions are
    preserved through each block; the only downsampling is the
    ``MaxPool2d(2, 2)`` at the end of each block.

    Two candidate kernel sizes are tested (as per the ambiguous whiteboard
    sketch): ``(4, 5)`` and ``(5, 5)``.  Three activation functions are
    supported: ``'leaky_relu'``, ``'swish'``, ``'gelu'``.  Dropout is applied
    after every activation: ``Dropout2d`` (channel-wise) in the convolutional
    blocks, regular ``Dropout`` in the dense head.

    The model outputs a **probability** in (0, 1) — sigmoid is applied inside
    ``forward()``.  Use ``nn.BCELoss`` for training and :func:`evaluate_model`
    for evaluation, consistent with :class:`AnomalyCNN`.

    To find and load the best configuration from the staged search, use
    :func:`get_professor_cnn_best`.

    Parameters
    ----------
    input_size : int
        Spatial side-length H of the square input (70 for Pythia).
    kernel_size : tuple[int, int] or int
        Conv kernel size.  Use ``(5, 5)`` or ``(4, 5)`` as suggested.
        ``padding='same'`` is always applied so any size is valid.
    activation : str
        Activation function: ``'leaky_relu'``, ``'swish'``, or ``'gelu'``.
    dropout : float
        Dropout probability in both conv and dense parts (default 0.2).
    dense_head : list[int]
        Sizes of the two hidden dense layers (default ``[128, 64]``).
    pooling : str
        ``'global_average'`` (AdaptiveAvgPool2d \u2192 64-dim vector) or
        ``'flatten'`` (Flatten \u2192 C \u00d7 H \u00d7 W vector).
    conv_channels : list[int]
        Output channels for each of the three conv blocks
        (default ``[16, 32, 64]``).
    """

    def __init__(
        self,
        input_size: int = 70,
        kernel_size: tuple | int = (5, 5),
        activation: str = "gelu",
        dropout: float = 0.2,
        dense_head: list[int] | None = None,
        pooling: str = "global_average",
        conv_channels: list[int] | None = None,
    ) -> None:
        super().__init__()
        if dense_head is None:
            dense_head = [128, 64]
        if conv_channels is None:
            conv_channels = [16, 32, 64]

        self._pooling = pooling

        def act() -> nn.Module:
            return _get_activation(activation)

        # --- 3 convolutional blocks: Conv \u2192 Activation \u2192 Dropout2d \u2192 MaxPool ---
        self.conv_blocks = nn.Sequential(
            # Block 1
            nn.Conv2d(1,                conv_channels[0], kernel_size=kernel_size, padding="same"),
            act(), nn.Dropout2d(p=dropout), nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(conv_channels[0], conv_channels[1], kernel_size=kernel_size, padding="same"),
            act(), nn.Dropout2d(p=dropout), nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(conv_channels[1], conv_channels[2], kernel_size=kernel_size, padding="same"),
            act(), nn.Dropout2d(p=dropout), nn.MaxPool2d(2, 2),
        )

        # --- Compute flat size for the dense head ----------------------
        if pooling == "global_average":
            # AdaptiveAvgPool2d(1) squeezes spatial dims to 1\u00d71.
            flat_size = conv_channels[2]
        else:
            # padding='same' keeps H\u00d7W unchanged through each conv;
            # three MaxPool(2,2) each floor-divide the spatial side by 2.
            h = input_size
            for _ in range(3):
                h = h // 2
            flat_size = conv_channels[2] * h * h

        # --- Dense head: 2 hidden layers + binary output ---------------
        self.head = nn.Sequential(
            nn.Linear(flat_size, dense_head[0]), act(), nn.Dropout(p=dropout),
            nn.Linear(dense_head[0], dense_head[1]), act(), nn.Dropout(p=dropout),
            nn.Linear(dense_head[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Grayscale image batch, shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Probability of anomaly, shape (B, 1).
        """
        x = self.conv_blocks(x)
        if self._pooling == "global_average":
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, C)
        else:
            x = x.flatten(1)                             # (B, C*H*W)
        return torch.sigmoid(self.head(x))               # probability in (0, 1)


# ---------------------------------------------------------------------------
# ProfessorCNN configuration registry
# ---------------------------------------------------------------------------

# Path to the best-config JSON written by step_professor_search.py.
_PROFESSOR_SEARCH_JSON = Path(__file__).parent / "faza_professor_cnn_search.json"

# Fallback when the search has not been run yet.
_PROFESSOR_CNN_DEFAULT_CONFIG: dict = {
    "kernel_size": [5, 5],
    "activation": "gelu",
    "dropout": 0.2,
    "dense_head": [128, 64],
    "pooling": "global_average",
}


def get_professor_cnn_best(input_size: int = 70) -> "ProfessorCNN":
    """Return a :class:`ProfessorCNN` initialised with the best searched config.

    Reads ``faza_professor_cnn_search.json`` (produced by
    ``step_professor_search.py``).  Falls back to a built-in sensible default
    and prints a reminder if the file does not exist.

    Parameters
    ----------
    input_size : int
        Spatial side-length H of the square input (70 for Pythia).

    Returns
    -------
    ProfessorCNN
        A freshly initialised (untrained) model ready for training.
    """
    cfg = dict(_PROFESSOR_CNN_DEFAULT_CONFIG)
    if _PROFESSOR_SEARCH_JSON.exists():
        try:
            with open(_PROFESSOR_SEARCH_JSON, "r", encoding="utf-8") as _f:
                _data = json.load(_f)
            cfg = _data.get("best_config", cfg)
        except (json.JSONDecodeError, KeyError):
            print(
                "[ProfessorCNN] Warning: could not parse best config from "
                f"'{_PROFESSOR_SEARCH_JSON}'. Using defaults."
            )
    else:
        print(
            "[ProfessorCNN] Search results not found at "
            f"'{_PROFESSOR_SEARCH_JSON}'.\n"
            "  Run 'python step_professor_search.py' first to select the best "
            "variant.\n  Proceeding with default config."
        )

    ks = cfg.get("kernel_size", [5, 5])
    if isinstance(ks, list):
        ks = tuple(ks)
    return ProfessorCNN(
        input_size=input_size,
        kernel_size=ks,
        activation=cfg.get("activation", "gelu"),
        dropout=float(cfg.get("dropout", 0.2)),
        dense_head=cfg.get("dense_head", [128, 64]),
        pooling=cfg.get("pooling", "global_average"),
    )


# ===========================================================================
# TRAINING  (matches notebook ch2_step1_v2.ipynb exactly)
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
    """Train a binary classifier with early stopping on validation loss.

    Exactly matches the training loop from the reference notebook.
    Model is moved to GPU automatically if available.
    Best weights (lowest val loss) are restored before returning.

    Parameters
    ----------
    model : nn.Module
        Model to train.
    train_loader, val_loader : DataLoader
        Batched training and validation data.
    criterion : nn.Module
        Loss function — ``nn.BCELoss()`` (model must output probabilities).
    optimizer : Optimizer
        ``torch.optim.Adam``.
    num_epochs : int
        Maximum training epochs.
    patience : int
        Epochs without val-loss improvement before early stopping.

    Returns
    -------
    nn.Module
        Model with best-val-loss weights restored.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_no_improve = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        # ------------------------------------------------------------------
        # Training phase
        # ------------------------------------------------------------------
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).view(-1, 1).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # ------------------------------------------------------------------
        # Validation phase
        # ------------------------------------------------------------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).view(-1, 1).float()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoka {epoch + 1:02d}/{num_epochs:02d} | "
            f"Strata (Train): {epoch_train_loss:.4f} | "
            f"Strata (Val): {epoch_val_loss:.4f}"
        )

        # ------------------------------------------------------------------
        # Early stopping
        # ------------------------------------------------------------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f" -> Early Stopping! Brak poprawy od {patience} epok. "
                    "Przerywam trening."
                )
                break

    print(f"Najlepsza strata walidacyjna (Val Loss): {best_val_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model


# ===========================================================================
# EVALUATION  (matches notebook ch2_step1_v2.ipynb exactly)
# ===========================================================================


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float = 0.5,
) -> tuple[float, float, float, float, float]:
    """Evaluate a trained binary classifier and print all metrics.

    Expects the model to output probabilities (sigmoid already applied).
    Matches the evaluate_model function from the reference notebook.

    Returns
    -------
    acc, prec, rec, f1, auc : float
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_labels: list = []
    all_preds_probs: list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()

            outputs = model(inputs).squeeze()

            # Guard: batch of exactly 1 collapses squeeze to scalar
            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)

            all_preds_probs.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds_probs = np.array(all_preds_probs)
    all_preds_classes = (all_preds_probs >= threshold).astype(int)

    acc  = accuracy_score(all_labels, all_preds_classes)
    prec = precision_score(all_labels, all_preds_classes, zero_division=0)
    rec  = recall_score(all_labels, all_preds_classes, zero_division=0)
    f1   = f1_score(all_labels, all_preds_classes, zero_division=0)

    try:
        auc = roc_auc_score(all_labels, all_preds_probs)
    except ValueError:
        auc = float("nan")

    print(f"  Accuracy:  {acc:.4f} (Dokładność ogólna)")
    print(f"  Precision: {prec:.4f} (Jak często wytypowana anomalia faktycznie nią była)")
    print(f"  Recall:    {rec:.4f} (Jaką część wszystkich prawdziwych anomalii udało się wykryć)")
    print(f"  F1-Score:  {f1:.4f} (Średnia harmoniczna precyzji i czułości)")
    print(f"  AUC-ROC:   {auc:.4f} (Zdolność modelu do rozróżniania klas)")

    return acc, prec, rec, f1, auc


# ===========================================================================
# AUTOENCODER  (Step 3 — unsupervised anomaly detection)
# ===========================================================================


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for unsupervised anomaly detection.

    The encoder follows the instructor-suggested **"3+2" topology** — three
    convolutional blocks followed by two fully-connected layers — and keeps
    the same channel progression as :class:`AnomalyCNN` (1->16->32->64, then
    a 128-unit hidden layer), so the trained encoder can be reused as a
    feature extractor by the Step 4 hybrid system.

    **Batch normalisation + LeakyReLU.** Each convolutional and hidden
    fully-connected layer is followed by batch normalisation, and every
    activation is a LeakyReLU.  This is a deliberate, necessary departure
    from a plain ReLU stack: without normalisation the decoder collapses on
    sparse, low-intensity data such as MNIST (mean pixel value ~0.13, ~87%
    of pixels black).  In that regime the output sigmoids saturate at zero,
    the gradient vanishes, and the network freezes while emitting an all-
    black image for every input — a reconstruction loss stuck at the data's
    mean square ``E[x^2]`` (~0.112 for MNIST).  Batch normalisation keeps
    activations in a healthy range and LeakyReLU keeps a gradient flowing
    through otherwise-dead units, which together prevent the collapse.
    Adding these layers does not affect Step 4 reuse — the encoder is simply
    reused together with its normalisation layers.

    Encoder ``g``  (3 conv blocks + 2 FC)::

        Conv2d(1->16,  k3, p1) -> BatchNorm -> LeakyReLU -> MaxPool(2)
        Conv2d(16->32, k3, p1) -> BatchNorm -> LeakyReLU -> MaxPool(2)
        Conv2d(32->64, k3, p1) -> BatchNorm -> LeakyReLU -> MaxPool(2)
        Flatten
        Linear(flatten -> 128) -> BatchNorm -> LeakyReLU
        Linear(128 -> latent_dim)              # bottleneck (latent code)

    Decoder ``f``  (2 FC + 3 up-conv blocks — mirror of the encoder)::

        Linear(latent_dim -> 128) -> BatchNorm -> LeakyReLU
        Linear(128 -> flatten)    -> BatchNorm -> LeakyReLU
        reshape -> (64, h, w)
        Upsample x2 -> Conv2d(64->32, k3, p1) -> BatchNorm -> LeakyReLU
        Upsample x2 -> Conv2d(32->16, k3, p1) -> BatchNorm -> LeakyReLU
        Upsample x2 -> Conv2d(16->8,  k3, p1) -> BatchNorm -> LeakyReLU
        interpolate -> (input_size, input_size)
        Conv2d(8->1, k3, p1) -> Sigmoid        # reconstruction in [0, 1]

    The decoder ends with an explicit ``F.interpolate`` to the exact input
    resolution because three 2x up-samplings of the encoder's spatial size
    do not in general land back on the original H (e.g. 3->6->12->24 != 28
    for MNIST, 8->16->32->64 != 70 for Pythia).  The final interpolation
    makes the same class work for any square input size.

    Trained with ``nn.MSELoss`` to minimise the reconstruction error on
    **clean images only**.  At inference the per-sample reconstruction error
    ``||x - f(g(x))||^2`` is the anomaly score (see
    :func:`reconstruction_scores`): the model reconstructs the clean manifold
    accurately but reconstructs anomalies poorly, so a high error flags an
    anomaly.

    Parameters
    ----------
    input_size : int
        Spatial side-length H of the square input (28 for MNIST,
        70 for Pythia).
    latent_dim : int
        Dimensionality of the bottleneck code (default 32).  A tight
        bottleneck is intentional — it forces the model to learn a compact
        representation of normality and prevents it from trivially copying
        arbitrary (anomalous) inputs through to the output.

    Notes
    -----
    Because the model contains batch-normalisation layers it must be run with
    a batch size of at least 2 while in training mode.  :func:`train_autoencoder`
    enforces this by skipping any final batch of size 1.  Inference utilities
    (:func:`reconstruction_scores`, :func:`evaluate_autoencoder`) call
    ``model.eval()``, where batch normalisation uses its stored running
    statistics and any batch size is safe.
    """

    # Negative slope for every LeakyReLU activation in the network.
    LEAK = 0.01

    def __init__(self, input_size: int = 28, latent_dim: int = 32) -> None:
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim

        # ---- Encoder convolutional stack -------------------------------
        self.enc_conv1 = nn.Conv2d(1,  16, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Infer spatial size after the 3 pooling layers via a dummy forward.
        # Only conv + pool are used here — batch normalisation does not
        # change spatial dimensions, so it can be safely ignored for this
        # shape calculation (and a dummy batch of 1 cannot pass through a
        # BatchNorm layer in training mode anyway).
        self.enc_h, self.enc_w = self._encoder_spatial(input_size)
        self.enc_channels = 64
        self.flatten_size = self.enc_channels * self.enc_h * self.enc_w

        # ---- Encoder batch-norm + fully-connected head -----------------
        self.enc_bn1 = nn.BatchNorm2d(16)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)
        self.enc_fc1 = nn.Linear(self.flatten_size, 128)
        self.enc_bn_fc1 = nn.BatchNorm1d(128)
        self.enc_fc2 = nn.Linear(128, latent_dim)        # latent: no BN

        # ---- Decoder fully-connected head ------------------------------
        self.dec_fc1 = nn.Linear(latent_dim, 128)
        self.dec_bn_fc1 = nn.BatchNorm1d(128)
        self.dec_fc2 = nn.Linear(128, self.flatten_size)
        self.dec_bn_fc2 = nn.BatchNorm1d(self.flatten_size)

        # ---- Decoder up-convolution stack ------------------------------
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec_conv1 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.dec_bn2 = nn.BatchNorm2d(16)
        self.dec_conv3 = nn.Conv2d(16,  8, kernel_size=3, padding=1)
        self.dec_bn3 = nn.BatchNorm2d(8)
        self.dec_conv_out = nn.Conv2d(8, 1, kernel_size=3, padding=1)  # output: no BN

    def _encoder_spatial(self, input_size: int) -> tuple[int, int]:
        """Return (H, W) of the feature map after the 3 conv/pool blocks.

        Uses only the convolution and pooling layers; activations and batch
        normalisation leave the spatial dimensions unchanged and are omitted.
        """
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_size, input_size)
            x = self.pool(self.enc_conv1(dummy))
            x = self.pool(self.enc_conv2(x))
            x = self.pool(self.enc_conv3(x))
        return x.shape[2], x.shape[3]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map an image batch to its latent code.

        Exposed separately so the trained encoder can be reused as a frozen
        feature extractor by the Step 4 hybrid system.

        Parameters
        ----------
        x : torch.Tensor
            Image batch, shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Latent codes, shape (B, latent_dim).
        """
        x = self.pool(F.leaky_relu(self.enc_bn1(self.enc_conv1(x)), self.LEAK))
        x = self.pool(F.leaky_relu(self.enc_bn2(self.enc_conv2(x)), self.LEAK))
        x = self.pool(F.leaky_relu(self.enc_bn3(self.enc_conv3(x)), self.LEAK))
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.enc_bn_fc1(self.enc_fc1(x)), self.LEAK)
        return self.enc_fc2(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct an image batch from latent codes.

        Parameters
        ----------
        z : torch.Tensor
            Latent codes, shape (B, latent_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed images, shape (B, 1, input_size, input_size),
            values in [0, 1].
        """
        x = F.leaky_relu(self.dec_bn_fc1(self.dec_fc1(z)), self.LEAK)
        x = F.leaky_relu(self.dec_bn_fc2(self.dec_fc2(x)), self.LEAK)
        x = x.view(x.size(0), self.enc_channels, self.enc_h, self.enc_w)
        x = F.leaky_relu(self.dec_bn1(self.dec_conv1(self.up(x))), self.LEAK)
        x = F.leaky_relu(self.dec_bn2(self.dec_conv2(self.up(x))), self.LEAK)
        x = F.leaky_relu(self.dec_bn3(self.dec_conv3(self.up(x))), self.LEAK)
        x = F.interpolate(
            x, size=(self.input_size, self.input_size),
            mode="bilinear", align_corners=False,
        )
        return torch.sigmoid(self.dec_conv_out(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode -> decode pass.

        Parameters
        ----------
        x : torch.Tensor
            Image batch, shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Reconstructed batch, same shape as ``x``.
        """
        return self.decode(self.encode(x))


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 30,
    patience: int = 5,
) -> nn.Module:
    """Train an autoencoder with early stopping on validation reconstruction loss.

    **Unsupervised:** the class labels yielded by the loaders are ignored
    entirely.  For every batch the reconstruction target is the input image
    itself, so when the caller passes clean-only loaders (as Step 3 does) the
    model only ever sees normal data and never learns anything attack-specific.

    Mirrors :func:`train_model` (device handling, early stopping, best-weight
    restoration) so the two training utilities behave consistently.

    Parameters
    ----------
    model : nn.Module
        A :class:`ConvAutoencoder` (or any module mapping image -> image).
    train_loader, val_loader : DataLoader
        Batched data.  Each batch is ``(image, label)``; the label is unused.
    criterion : nn.Module
        Reconstruction loss, typically ``nn.MSELoss()``.
    optimizer : Optimizer
        e.g. ``torch.optim.Adam``.
    num_epochs : int
        Maximum number of training epochs (default 30).
    patience : int
        Epochs without validation-loss improvement before stopping (default 5).

    Returns
    -------
    nn.Module
        The model with best-validation-loss weights restored.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    epochs_no_improve = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(num_epochs):
        # ---- Training phase --------------------------------------------
        model.train()
        running_loss = 0.0
        seen = 0
        for inputs, _ in train_loader:          # labels deliberately ignored
            # Batch normalisation cannot process a single-sample batch in
            # training mode; skip a stray final batch of size 1 (harmless,
            # at most one image per epoch).
            if inputs.size(0) < 2:
                continue
            inputs = inputs.to(device)
            seen += inputs.size(0)
            optimizer.zero_grad()
            reconstruction = model(inputs)
            loss = criterion(reconstruction, inputs)   # target = input
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_loss = running_loss / max(seen, 1)

        # ---- Validation phase ------------------------------------------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                reconstruction = model(inputs)
                loss = criterion(reconstruction, inputs)
                val_loss += loss.item() * inputs.size(0)
        epoch_val_loss = val_loss / len(val_loader.dataset)

        print(
            f"Epoka {epoch + 1:02d}/{num_epochs:02d} | "
            f"Strata rekonstrukcji (Train): {epoch_train_loss:.6f} | "
            f"Strata rekonstrukcji (Val): {epoch_val_loss:.6f}"
        )

        # ---- Early stopping --------------------------------------------
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f" -> Early Stopping! Brak poprawy od {patience} epok. "
                    "Przerywam trening."
                )
                break

    print(f"Najlepsza strata walidacyjna (Val Loss): {best_val_loss:.6f}")
    model.load_state_dict(best_model_wts)
    return model


def reconstruction_scores(
    model: nn.Module,
    loader: DataLoader,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-sample reconstruction-error anomaly scores.

    The anomaly score for an image ``x`` is the reconstruction error

    .. math::
        s(x) = \\tfrac{1}{C H W} \\sum \\bigl(x - f(g(x))\\bigr)^2

    i.e. the **mean** squared error over all pixels.  This is a strictly
    monotonic rescaling of the textbook score ``||x - f(g(x))||^2`` (they
    differ only by the constant factor C*H*W), so every ranking-based metric
    — AUC-ROC in particular — is identical for the two forms.  The mean form
    is used because it is resolution-independent (directly comparable between
    28x28 MNIST and 70x70 Pythia) and numerically better behaved.

    A higher score means a worse reconstruction, i.e. a more anomalous image.

    Parameters
    ----------
    model : nn.Module
        A trained autoencoder.
    loader : DataLoader
        Yields ``(image, label)`` batches.

    Returns
    -------
    scores : np.ndarray, shape (N,)
        Per-sample reconstruction error.
    labels : np.ndarray, shape (N,)
        Ground-truth labels (0 = clean, 1 = attack), passed through unchanged
        for convenience when computing metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    all_scores: list = []
    all_labels: list = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            reconstruction = model(inputs)
            err = (reconstruction - inputs) ** 2
            # mean over channel + spatial dims -> one score per image
            err = err.flatten(start_dim=1).mean(dim=1)
            all_scores.extend(err.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_scores), np.array(all_labels)


def select_threshold(
    clean_scores: np.ndarray,
    percentile: float = 95.0,
) -> float:
    """Pick an anomaly-score threshold from clean reconstruction errors.

    The threshold is the given percentile of the reconstruction errors
    measured on a **clean-only** validation split.  An image whose score
    exceeds the threshold is flagged as an anomaly.

    Choosing the 95th percentile means the detector accepts a ~5 % false-
    positive rate on clean data by construction.  Crucially the threshold is
    derived **without ever looking at attack samples**, so the method remains
    fully unsupervised with respect to the anomaly class.

    Parameters
    ----------
    clean_scores : np.ndarray
        Reconstruction errors on clean validation images.
    percentile : float
        Percentile in (0, 100) used as the cut-off (default 95.0).

    Returns
    -------
    float
        The anomaly-score threshold.
    """
    return float(np.percentile(clean_scores, percentile))


def evaluate_autoencoder(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float,
) -> tuple[float, float, float, float, float]:
    """Evaluate an autoencoder anomaly detector and print all metrics.

    An image is predicted "attack" when its reconstruction error is greater
    than or equal to ``threshold``.  AUC-ROC is computed from the raw
    (continuous) reconstruction errors and is therefore threshold-free, which
    makes it the fairest metric for comparing the autoencoder against the
    supervised classifier of Step 1.

    The return signature matches :func:`evaluate_model` and
    :func:`evaluate_linear_detector` exactly, so the result tuple can be fed
    straight into :func:`parse_results`.

    Parameters
    ----------
    model : nn.Module
        A trained autoencoder.
    test_loader : DataLoader
        Yields ``(image, label)`` batches; labels 0 = clean, 1 = attack.
    threshold : float
        Anomaly-score cut-off, typically from :func:`select_threshold`.

    Returns
    -------
    acc, prec, rec, f1, auc : float
    """
    scores, labels = reconstruction_scores(model, test_loader)
    preds = (scores >= threshold).astype(int)

    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = float("nan")

    print(f"  Threshold: {threshold:.6f} (próg błędu rekonstrukcji)")
    print(f"  Accuracy:  {acc:.4f} (Dokładność ogólna)")
    print(f"  Precision: {prec:.4f} (Jak często wytypowana anomalia faktycznie nią była)")
    print(f"  Recall:    {rec:.4f} (Jaką część prawdziwych anomalii udało się wykryć)")
    print(f"  F1-Score:  {f1:.4f} (Średnia harmoniczna precyzji i czułości)")
    print(f"  AUC-ROC:   {auc:.4f} (Zdolność rozróżniania klas, niezależna od progu)")

    return acc, prec, rec, f1, auc


# ===========================================================================
# PYTHIA LINEAR DETECTOR  (Cohen's d pixel selection + Logistic Regression)
# ===========================================================================


class PythiaLinearDetector:
    """Pixel-selection linear detector for binary anomaly classification.

    Design rationale
    ----------------
    Neural networks trained from scratch on only ~1 000 labelled samples
    often fail to discover subtle, spatially-localised attack signals because
    they search the whole 4 900-pixel space simultaneously.  This detector
    takes a different approach:

    1. **Effect-size ranking**: compute Cohen's |d| for every pixel using
       ONLY the training split.  Large |d| means the attack consistently
       shifts that pixel's distribution relative to clean images.
    2. **Top-K feature selection**: retain the K pixels with the highest
       discriminative power (K=500 by default).
    3. **Regularised logistic regression**: train a linear classifier on the
       selected features with strong L2 regularisation (C=0.01) to prevent
       overfitting on the small dataset.

    All three steps use only training data, so the evaluation on the val and
    test sets is fully unbiased.

    Parameters
    ----------
    top_k : int
        Number of most discriminative pixels to keep (default 500).
    C : float
        Inverse regularisation strength for LogisticRegression (smaller →
        stronger regularisation; default 0.01).
    """

    def __init__(self, top_k: int = 500, C: float = 0.01) -> None:
        self.top_k = top_k
        self.C = C
        self._pixel_mask: np.ndarray | None = None   # selected flat indices
        self._scaler:     StandardScaler    | None = None
        self._clf:        LogisticRegression | None = None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dataset_to_arrays(
        dataset,
        batch_size: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a torch Dataset into flat NumPy arrays (X, y).

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Must yield (image_tensor, label_tensor) pairs.
        batch_size : int
            Batch size for the temporary DataLoader.

        Returns
        -------
        X : ndarray of shape (N, H*W)
            Flattened pixel features.
        y : ndarray of shape (N,)
            Integer class labels (0 = clean, 1 = attack).
        """
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        X_parts, y_parts = [], []
        for imgs, labels in loader:
            X_parts.append(imgs.numpy().reshape(len(imgs), -1))
            y_parts.append(labels.numpy().ravel().astype(int))
        return np.vstack(X_parts), np.concatenate(y_parts)

    @staticmethod
    def _cohens_d(X_clean: np.ndarray, X_attack: np.ndarray) -> np.ndarray:
        """Compute per-column Cohen's |d| between two groups.

        Parameters
        ----------
        X_clean, X_attack : ndarray of shape (N_c, F), (N_a, F)
            Feature matrices for clean and attack samples.

        Returns
        -------
        ndarray of shape (F,)
            Absolute Cohen's d effect size for each feature.
        """
        mu_c, mu_a = X_clean.mean(0), X_attack.mean(0)
        var_pooled  = (X_clean.var(0) + X_attack.var(0)) / 2 + 1e-8
        return np.abs(mu_a - mu_c) / np.sqrt(var_pooled)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, train_dataset) -> "PythiaLinearDetector":
        """Fit the detector on labelled training data.

        Steps:
        1. Separate clean (y=0) and attack (y=1) samples.
        2. Compute Cohen's |d| per pixel.
        3. Select top-K pixels by effect size.
        4. Fit StandardScaler + LogisticRegression on selected pixels.

        Parameters
        ----------
        train_dataset : torch.utils.data.Dataset
            Balanced training dataset yielding (image, label) pairs.

        Returns
        -------
        self
        """
        X, y = self._dataset_to_arrays(train_dataset)
        X_clean  = X[y == 0]
        X_attack = X[y == 1]

        # Step 1: effect-size ranking (training data only — no leakage)
        d = self._cohens_d(X_clean, X_attack)
        self._pixel_mask = np.argsort(d)[-self.top_k:]

        # Step 2: scale & fit
        X_sel = X[:, self._pixel_mask]
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_sel)

        self._clf = LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")
        self._clf.fit(X_scaled, y)

        # Report fitted statistics
        d_selected = d[self._pixel_mask]
        print(
            f"[PythiaLinearDetector] Top-{self.top_k} pixels selected | "
            f"max |d|={d_selected.max():.3f}  "
            f"mean |d|={d_selected.mean():.3f}"
        )
        train_proba = self._clf.predict_proba(X_scaled)[:, 1]
        train_auc   = roc_auc_score(y, train_proba)
        print(f"[PythiaLinearDetector] Train AUC: {train_auc:.4f}")

        return self

    def predict_proba(self, dataset) -> np.ndarray:
        """Return probability of the attack class (label=1).

        Parameters
        ----------
        dataset : torch.utils.data.Dataset
            Dataset yielding (image, label) pairs.

        Returns
        -------
        ndarray of shape (N,)
            Predicted probability of attack (sigmoid output).
        """
        if self._pixel_mask is None or self._clf is None:
            raise RuntimeError("Call .fit() before .predict_proba()")
        X, _ = self._dataset_to_arrays(dataset)
        X_sel    = X[:, self._pixel_mask]
        X_scaled = self._scaler.transform(X_sel)
        return self._clf.predict_proba(X_scaled)[:, 1]


def evaluate_linear_detector(
    detector: PythiaLinearDetector,
    test_dataset,
    label: str = "",
) -> tuple[float, float, float, float, float]:
    """Evaluate a fitted PythiaLinearDetector on a labelled test dataset.

    Parameters
    ----------
    detector : PythiaLinearDetector
        A fitted detector.
    test_dataset : torch.utils.data.Dataset
        Dataset yielding (image, label) pairs.
    label : str
        Optional name printed in the summary line.

    Returns
    -------
    tuple
        ``(accuracy, precision, recall, f1, auc_roc)`` — same layout as
        :func:`evaluate_model` for drop-in compatibility with
        :func:`parse_results`.
    """
    _, y_true = PythiaLinearDetector._dataset_to_arrays(test_dataset)
    proba     = detector.predict_proba(test_dataset)
    y_pred    = (proba >= 0.5).astype(int)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc  = roc_auc_score(y_true, proba)

    tag = f" [{label}]" if label else ""
    print(f"Ewaluacja{tag}:")
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