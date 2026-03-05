"""
Centralized seed management for full reproducibility.

Ensures identical results across multiple runs by controlling
all sources of randomness: Python, NumPy, PyTorch, and CUDA.

Critical for research paper results.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


class SeedManager:
    """
    Ensures complete reproducibility across all random operations.

    Usage:
        seed_mgr = SeedManager(seed=42)
        seed_mgr.set_seed()
        generator = seed_mgr.get_generator()
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def set_seed(self) -> None:
        """Set all random seeds and enable deterministic mode."""
        # Python built-in random
        random.seed(self.seed)

        # NumPy
        np.random.seed(self.seed)

        # PyTorch CPU
        torch.manual_seed(self.seed)

        # PyTorch CUDA (all GPUs)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # CuDNN deterministic mode
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Python hash seed (dictionary ordering)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        # CUBLAS workspace config for deterministic CUDA operations
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Enable PyTorch deterministic algorithms
        torch.use_deterministic_algorithms(True, warn_only=True)

        print(f"[SeedManager] All random seeds set to {self.seed}")
        print(f"[SeedManager] Deterministic mode: ENABLED")

    def get_generator(self) -> torch.Generator:
        """Return a seeded torch.Generator for DataLoader."""
        g = torch.Generator()
        g.manual_seed(self.seed)
        return g

    def worker_init_fn(self, worker_id: int) -> None:
        """
        Seed function for DataLoader workers.

        Pass as worker_init_fn to torch.utils.data.DataLoader.
        Each worker gets a deterministic but unique seed.
        """
        worker_seed = self.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def get_device(self) -> torch.device:
        """Return the appropriate device (GPU if available, else CPU)."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[SeedManager] Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("[SeedManager] Using CPU")
        return device
