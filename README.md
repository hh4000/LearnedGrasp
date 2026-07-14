# Deep learning for grasp inference of the HOT3D dataset
 <img src="img.png">

In the present repository, I have attempted to learn the grasp from just visual and gaze data. 
All attempted models can be found in [`src/graspcnn/models.py`](src/graspcnn/models.py).
You will be able to see that I eventually strayed away from CNN models in favor of GNN models. 
This is rather a change in object representation from RGB image to 3D mesh.
Ultimately the models I made were dubious, with lackluster (but statistically significant) performance.
I want to note that the models here are made to regress the MANO pose (PCA-decomposed joint angles), which substantially raised the difficulty; this was to avoid the unclear classification boundaries that are natural with discrete grasp classification.

## Repository layout
```
src/graspcnn/          Installable library package
  models.py            CNN / GNN model definitions (MANO-pose regression)
  losses.py            Custom loss functions (FocalLoss, GaussianNLLLoss)
  features.py          Engineered geometric feature extraction
  data/                Datasets: image_set.py, mesh_set.py
  training/            Shared training infrastructure
                         base.py — Trainer, DataLoaderGetter, R2Accumulator
                         ui.py   — NiceGUI chart/config helpers
scripts/               Runnable entrypoints
  train_ui.py            NiceGUI: image classification
  train_regression_ui.py NiceGUI: image → MANO-pose regression
  train_gnn_ui.py        NiceGUI: mesh (GNN) regression
  train_gnn_batch.py     Headless batched GNN trial runner
  encoding_test.py       Contrastive mesh-embedding experiment
analysis/              Plotting, statistics, and visualisation scripts
```

## Setup & running
Install the package (editable) into your environment, then run any entrypoint
**from the repository root** (scripts resolve `models/`, `params/`, `results/`,
`train_log/` and the config JSONs relative to the working directory):

```bash
uv pip install -e .          # or: pip install -e .   (alt: PYTHONPATH=src)
python scripts/train_gnn_ui.py
python scripts/train_gnn_batch.py
```

Datasets cached before this restructure still load: legacy pickle module paths
(`grasp_mesh_set`, `MODELS`, …) are aliased to the new package modules in
`graspcnn._compat`.

## Latest developments
Before hand-in of my master thesis, I rushed out a script to attempt contrastive learning [`scripts/encoding_test.py`](scripts/encoding_test.py), where I tried to see if the GNN could learn object affordance from object geometry alone.
This was achieved by simply picking an anchor, and giving 3 positive (similar) examples and 2 negative (dissimilar) examples.
These are:
- <b>anchor</b> (original point)
- <b>near embedding</b> (<i>positive</i>; node on the same graph that is close to the anchor)
- <b>augmented embedding</b> (<i>positive</i>; same node as anchor but with slight value jitter)
- <b>rotated embedding</b> (<i>positive</i>; Same node as anchor but rotated at random)
- <b>distant embedding</b> (<i>negative</i>; random node on the same graph, with a minimum hop distance from initial point)
- <b>different object embedding</b> (<i>negative</i>; random node on different graph)

These were permuted into 6 different triplet pairs (anchor + positive + negative), which were fed into the model.
I found here that the model was able to learn different geometric structures fairly easily with this method, which led me to believe that <b>a pretrained mesh embedding network could substantially help in the classification process</b>.
An important note I would like to make here <b>this worked particularly well with edge-convolutions</b>.

### TL:DR
I believe pretraining an embedding network to understand object geometry is the way to proceed.



# Dev note
The codebase was restructured into an installable `graspcnn` package with the
runnable scripts split into `scripts/` (training) and `analysis/`. The training
front-ends now share a common `Trainer` base and helpers under
`graspcnn.training`.

If you need any help feel free to contact me via: 
- <b>WhatsApp</b>: +45 4250 6200

or
- <b>Email</b>: hans.henrik.dalgaard@gmail.com
