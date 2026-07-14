"""GraspCNN: deep learning for grasp inference from visual and gaze data.

The package is organised as:

- ``graspcnn.models``   — CNN/GNN model definitions (MANO-pose regression).
- ``graspcnn.losses``   — custom loss functions.
- ``graspcnn.features`` — engineered geometric feature extraction.
- ``graspcnn.data``     — image and mesh datasets.
- ``graspcnn.training`` — shared training infrastructure (``Trainer`` base).
"""

__version__ = "0.1.0"

# Register legacy module aliases so datasets pickled before the package
# restructure still load via ``torch.load``. See ``graspcnn._compat``.
from graspcnn import _compat as _compat  # noqa: E402

_compat.install()
