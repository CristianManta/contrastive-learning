# ContrastiveTeam0

## A place to push new ideas for Contrastive Learning


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

### Accuracy Claims from [this paper](https://arxiv.org/pdf/2002.05709.pdf)
| Baseline (supervised ResNet50) | Contrastive |
|:--------:|:--------:|
| 93.6%    | 90.6%    |

### Status

The penalty parameter used for the training of the Tikhonov regularized baseline model (0.1) is apparently too small (I know that because the attack distances were smaller than for the unregularized model, so it's within the same margin of error). I'm re-running many scripts in parallel with varying penalties during the night.

I also replaced the finite difference gradient approximation from [this paper](https://arxiv.org/pdf/1905.11468.pdf) and [this repo](https://github.com/cfinlay/tulip) by an exact computation (at the cost of scalability, but the scripts are working fine on CIFAR10 at least). I had to do this in order to fix the problem that the accuracy of the baseline model was below 20% suddenly when applying Tikhonov, while it was 94% without it. The loss landscape must be so non smooth in this case that the finite difference approximation was very bad.

### TODO
- [x] Adapt Chris's baseline training script to our experiments. In particular, add option to train on a subset of the labels. Run it on 100%, 10% and 1% of the labels.
- [x] Add fine-tuning procedure of the contrastive model by initializing the linear classifier weights to 0 (not its biases).
- [x] Run the fine-tuning on 100%, 10% and 1% of the labels
- [x] Add Tikhonov Regularization to the baseline training script
- [ ] Train the Tikhonov regularized baseline model and attack the unregularized and regularized version. Compare. <-- **In progress**
- [ ] Add plots to compare the attack values
- [ ] Implement contrastive loss layer by layer and using class label information instead of purely positive/negative samples
- [ ] Implement layer-by-layer Tikhonov regularization
- [ ] Implement a nonlinear classifier that uses an "average class vector" and give confidence by measuring similarity between class vector and encoder output







