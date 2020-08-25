# ContrastiveTeam0

## A place to push new ideas for Contrastive Learning
### Description
All the research has been done using the ResNet50 model as the encoder for learning the representations.

The [cifar10](https://github.com/AOTeam2020/ContrastiveTeamO/tree/cristian/cifar10) directory contains all the
training scripts and utilities for the contrastive learning model without Tikhonov regularization on the CIFAR10 dataset.
[train.py](https://github.com/AOTeam2020/ContrastiveTeamO/blob/cristian/cifar10/train.py)
is the main script for training the contrastive model.
[logreg_train.py](https://github.com/AOTeam2020/ContrastiveTeamO/blob/cristian/cifar10/logreg_train.py)
is the script for the linear classifier used in the linear evaluation protocol in [this paper](https://arxiv.org/pdf/2002.05709.pdf).
[logreg_train_fine_tune.py](https://github.com/AOTeam2020/ContrastiveTeamO/blob/cristian/cifar10/logreg_train_fine_tune.py)
implements the fine-tuning of the whole model, **but** with the following important difference
compared to `logreg_train.py`:
 
```python
# logreg_train_fine_tune.py
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, use_softmax=False):
        super(LogisticRegression, self).__init__()
        linear_layer1 = nn.Linear(input_dim, input_dim)
        linear_layer2 = nn.Linear(input_dim, output_dim)
        self.l1 = linear_layer1
        self.l2 = linear_layer2
        self.use_softmax = use_softmax

    def forward(self, x):
        x = self.l1(x)
        x = F.relu(x)
        outputs = self.l2(x)
        if self.use_softmax:
            outputs = outputs.softmax(dim=-1)
        return outputs
```

```python
# logreg_train.py
class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim, use_softmax=False):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.use_softmax = use_softmax

    def forward(self, x):
        outputs = self.linear(x)
        if self.use_softmax:
            outputs = outputs.softmax(dim=-1)
        return outputs
```
Indeed, I found that having 2 linear layers (with `relu` in between) *slightly* increased the accuracy, which I found appropriate for fine-tuning. Also, the idea came from the fact that this is the structure of the projection head g in [this paper](https://arxiv.org/pdf/2002.05709.pdf).
Moreover, I found that initializing the weights to 0 (after applying `relu`) did not increase the accuracy.

[train_baseline.py](https://github.com/AOTeam2020/ContrastiveTeamO/blob/cristian/cifar10/train_baseline.py)
is a fancy baseline training script with Tikhonov regularization options. It can also train on a random subset of the labels.
 **Important:** The Tikhonov regularization implemented here uses the finite difference approximation from [this repository](https://github.com/cfinlay/tulip/tree/master/cifar10).
 
[train_baseline_exact.py](https://github.com/AOTeam2020/ContrastiveTeamO/blob/cristian/cifar10/train_baseline_exact.py)
does the same job as `train_baseline.py`, but I replaced the finite difference approximation by a double backpropagation for better accuracy with "less smooth" loss functions at the cost of scalability. I kept both versions because they will likely both be useful in different circumstances.

The `cifar10` directory also contains all the attack codes. I adopted the convention that, if a name is not followed by "_baseline", then it refers to the basic contrastive model. The "_500" and "_5000" appended to the names of some files or directories refer to the number of labels that the model had access to during training.

The [imagenet](https://github.com/AOTeam2020/ContrastiveTeamO/tree/cristian/imagenet)
directory has a similar goal to the `cifar10` one, but has been much less explored.

The [regularizaztion](https://github.com/AOTeam2020/ContrastiveTeamO/tree/cristian/regularization)
 directory contains the training scripts for the Tikhonov regularized *contrastive models*. The structure is similar to
 that of `cifar10` directory. The two main scripts are [classic_tikhonov.py](https://github.com/AOTeam2020/ContrastiveTeamO/blob/cristian/regularization/classic_tikhonov.py)
 and [classic_tikhonov_exact.py](https://github.com/AOTeam2020/ContrastiveTeamO/blob/cristian/regularization/classic_tikhonov_exact.py)
 for training a Tikhonov regularized contrastive model. The difference between them is the implementation of the Tikhonov penalty term (see `train_baseline.py` vs. `train_baseline_exact.py` above).
 
[layer_wise.py](https://github.com/AOTeam2020/ContrastiveTeamO/blob/cristian/regularization/layer_wise.py)
is an **experimental** script towards the layer-by-layer regularization.
The last thing that I did with it was to figure out how to isolate the major layers of the resnet model in order to call them separately, thus enabling the possibility to add a penalty term to each layer. **This script is not in a working state**.

All the directories that start with either "runs" or "logs" are (logging) directories that should contain all the necessary info in order to reproduce a given experiment, in particular: `args.yaml` + `.py` script used at that time. They also contain info about the outcomes of each experiment, like accuracies.
In addition, all the directories inside the attack directories are for logging the adversarial distances.


## Comparison between the baseline ResNet50 and our contrastive model on CIFAR-10

### Accuracy on 100% of the Test Set

|   pct. of labels available at train time   | Baseline   | Contrastive Linear Eval  | Contrastive Fine-Tuned |
|:-------:|:-----:|:-------:|:---:|
| 100%      | 93.93%   | 74.79%   | 77.05% |
| 10%       | 64.39% |   70.83% | 71.36% |
| 1%        | 30.92% |    67.56%| 66.94% |

### Attacks (doesn't include fine-tuned contrastive yet)

#### Median adversarial distances

| attack type | Baseline | Baseline + Tikhonov    | Contrastive   | Contrastive + Tikhonov  |
|:----:|:-------------:|:-------------:|:-----:|:---:|
pgd  | 0.121 | To do     | 0 (?!) | 0 (?!) |
plb | 0.125 | To do      | 0.12      |   0.185 |

#### Mean adversarial distances

| attack type | Baseline | Baseline + Tikhonov    | Contrastive   | Contrastive + Tikhonov  |
|:----:|:-------------:|:-------------:|:-----:|:---:|
pgd | 0.147 | To do     | 0.0355 | 0.0627 |

#### Max adversarial distances

| attack type |   Baseline| Baseline + Tikhonov    | Contrastive   | Contrastive + Tikhonov  |
|:----:|:-------------:|:-------------:|:-----:|:---:|
pgd | 0.5 | To do     | 0.5 | 0.5 |
plb | 0.755 |  To do    | 0.122      |   nan (?!) |

To reproduce the experiments, please refer to the appropriate logging directory to find the appropriate arguments.

### Accuracy Claims from [this paper](https://arxiv.org/pdf/2002.05709.pdf)
| Baseline (supervised ResNet50) | Contrastive |
|:--------:|:--------:|
| 93.6%    | 90.6%    |

### Status

#### Aug. 24
I replaced the finite difference gradient approximation from [this paper](https://arxiv.org/pdf/1905.11468.pdf) and [this repo](https://github.com/cfinlay/tulip) by an exact computation (at the cost of scalability, but the scripts are working fine on CIFAR10 at least).
I had to do this in order to fix the problem that the accuracy of the baseline model was below 20% suddenly when applying Tikhonov (with `penalty=0.1`), while it was 94% without it.
The loss landscape must be so non smooth in this case that the finite difference approximation was very bad.

~~Now the penalty parameter used for the training of the Tikhonov regularized baseline model (0.1) is apparently too small (I know that because the attack distances were smaller than for the unregularized model, so it's within the same margin of error).~~ I'm re-running many scripts in parallel with varying penalties during the night.

#### Aug. 25
The actual issue was that `grad` needed `create_graph=True` passed in as parameter in order to re-use it again for further differentiation. Otherwise any penalty would have no effect at all with this new implementation.

Now running again the baseline + Tikhonov script with `penalty=0.1`, as well as contrastive + (classic) Tikhonov with the same `penalty`.
So far, the training finally seems to work well (accuracy values make sense).



### TODO
- [x] Adapt Chris's baseline training script to our experiments. In particular, add option to train on a subset of the labels. Run it on 100%, 10% and 1% of the labels.
- [x] Add fine-tuning procedure of the contrastive model by initializing the linear classifier weights to 0 (not its biases).
- [x] Run the fine-tuning on 100%, 10% and 1% of the labels
- [x] Add Tikhonov Regularization to the baseline training script
- [ ] Wrap-up the project (document well the experiments + summarize research + organize/clean this repo) <-- **In progress**
- [ ] Train the Tikhonov regularized baseline model and attack the unregularized and regularized version. Compare. <-- **In progress**
- [ ] Add plots to compare the attack values
- [ ] Implement contrastive loss layer by layer and using class label information instead of purely positive/negative samples
- [ ] Implement layer-by-layer Tikhonov regularization
- [ ] Implement a nonlinear classifier that uses an "average class vector" and give confidence by measuring similarity between class vector and encoder output
- [ ] It would be nice to visualize the decision boundary of such a classifier







