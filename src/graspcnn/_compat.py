"""Backward-compatibility aliases for unpickling pre-refactor artifacts.

Datasets cached before the ``src/graspcnn`` restructure were pickled while the
classes lived in top-level modules (``grasp_mesh_set``, ``grasp_image_set``,
``MODELS`` …). ``pickle`` records the defining module path, so ``torch.load``
on those files would fail with ``ModuleNotFoundError`` under the new layout.

Registering the old module names as aliases of their new homes in
``sys.modules`` lets the old files keep loading. Saved ``.pth`` checkpoints are
plain ``state_dict``\\ s and never needed this.
"""

import sys

from graspcnn import features, losses, models
from graspcnn.data import image_set, mesh_set

# old top-level module name → new module object
_ALIASES = {
    'grasp_mesh_set': mesh_set,
    'grasp_image_set': image_set,
    'MODELS': models,
    'custom_loss': losses,
    'engineered_features': features,
}


def install() -> None:
    """Register legacy module names so old pickles resolve. Idempotent."""
    for old_name, module in _ALIASES.items():
        sys.modules.setdefault(old_name, module)
