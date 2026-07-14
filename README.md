# Deep learning for grasp inference of the HOT3D dataset
 <img src="img.png">
In the present repository, I have attempted to learn the grasp from just visual and gaze data. 
All attempted models can be found in [<b>./MODELS.py</b>].
You will be able to see that I eventually strayed away from CNN models in favor of GNN models. 
This is rather a change in object representation from RGB image to 3D mesh.
Ultimately the models I made were dubious, with lackluster (but statistically significant) performance.
I want to note that the models here are made to regress the MANO pose (PCA-decomposed joint angles), which substantially raised the difficulty; this was to avoid the unclear classification boundaries that are natural with discrete grasp classification.

## Latest developments
Before hand-in of my master thesis, I rushed out a script to attempt contrastive learning (<b>./encoding-test.py</b>), where I tried to see if the GNN could learn object affordance from object geometry alone.
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
I apologize here for the messy repo; frankly, I needed more time to make it, hence why it is not super well maintained. I intend to clean it up more, but I wanted it to be able to hand it over ASAP.

If you need any help feel free to contact me via: 
- <b>WhatsApp</b>: +45 4250 6200

or
- <b>Email</b>: hans.henrik.dalgaard@gmail.com