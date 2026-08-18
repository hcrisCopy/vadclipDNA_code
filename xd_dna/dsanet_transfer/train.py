"""Run the unchanged DNA-VadCLIP trainer with a DSANet transfer contract.

The parent trainer already preserves VadCLIP's XD optimizer, losses, AP2 model
selection, tqdm progress, atomic checkpoints and ``--resume`` behaviour.  This
thin entry point deliberately avoids forking that verified training logic.
"""
from __future__ import annotations

from ..train import main


if __name__ == "__main__":
    main()
