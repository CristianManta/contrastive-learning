### Contrastive losses

Models trained with contrastive losses work as follows.  E.g. SimCLR, Contrastive Learning of Visual Representation.

Find an embedding from data to feature space, which consists of unit vectors,

$f: \mathcal X \to F = S^d = \{ x \in \mathbb R^d \mid \|x\|_2 = 1\}$ 

The embedding has a natural similarity, $s(x,y) = x\cdot y$, along with the distance, 

$d^2(x,y) = \|x-y\|^2 = 1-x\cdot y = 1-s(x,y)$

which simplifies, since the vectors are unit norm.  So $d(x,y)^2 = 1 -s(x,y)$.

The embedding comes from using data augmentation.  Use a loss which (without labels) requires that multiple views of the same image are close, while views of a different image are far.  Done with softmax version of the distances.  

Views: Data augmentation (Gaussian smoothing, cropping, etc).  Augmented images should be similar.  Different images should have a low similarity. The data augmentation is designed to erase nuisance features, and keep features which are relevant to downstream applications. 



#### Classification

After training for a long time to get the embedding (using no label). Train a classifier (and allow for weight fine tuning)

#### Faster Training

Mike and Mido: train augmenting the loss with some label information.  Trains faster. 

1. Possible explanation of faster training: getting better gradients.  
   - Method 1: Sometimes you are telling the model to that *different*  images from the *same class*  are dissimilar.  This is inconstistent with the later classification.   
   - Method 2: using a small fraction of label information: this inconsistency is happening less often. 
   - Quantify: infrequent events which tell two similar images to be dissimilar: obtain a large gradient
   - Possible improvement: during training, use confidence to guess if images are similar.  Choose only the most likely to be dissimilar ones in the batch to put in the loss.  Avoid ambiguous ones.
2. Second possible explanation of faster training: more organized initial information 

1. Possible Improved Method: *Use a semantic label embedding* 

- Use a semantic label embedding (e.g. Word2Vec) for the labelled images.  Now we already have a semantic embedding of the point.  New loss will push labelled images towards the label vector.  Will also push similar images close to these.
- Could lead to a more "organized" embedding from the start.  Faster training initially as the classes separate themselves. 

#### Other ideas

#### Confidence

Classification is done by using a small number of labelled images, then defining a linear (please clarify defnition) classifier, $c(x) = n\cdot f(x)$. 
I interpret this to mean that for each class, there is a typical (or mean) example with features $\bar f_i$ and then the classifier strength is simply $c_i(x) =\bar f_i \cdot f(x)$, based on similarity.  Then we have a classifier for each class, and use the strongest one to classify.  Confidence should be measured by similiary: $c_i(x) = 1$ would be the highest possible confidence.

#### Adversarial perturbations

An adversarial perturbation is a perturbation $v$  of small norm so that $x+$v has a different classification.  As shown in previous work, the $\delta = \|v\|_2$  can be estimated by 

$ \delta \leq \frac{ c_i(x) - c_{\max}(x)}{G}$

The ratio of : the gap between the value of the correct classification and the next largest one, and a bound on the gradient of the map, which in this (since the classifier is normalize to have graident 1) should be a bound on $\|\nabla f(x)\|$

Currently, there is nothing in training to make this bound small - except the imbedding pushing things away should already have some good properties.

#### Robustness

- model is alread more robust than CNN
- could augment with AT.

#### Inference with multiple views

- Model is trained using multiple views.
- Why not do inference using multiple views as well?
- Should be more accurate, more robust, better confidence...