# ContrastiveTeam0

## A place to push new ideas for Contrastive Learning


## Comparison between the baseline ResNet50 and our contrastive model on CIFAR-10

### Accuracy on 100% of the Test Set

|   pct. of labels available at train time   | Baseline   | Contrastive Linear Eval  | Contrastive Fine-Tuned |
|:-------:|:-----:|:-------:|:---:|
| 100%      | 93.93%   | 74.79%   | Coming soon |
| 10%       | 64.39% |   70.83% | Coming soon |
| 1%        | 30.92% |    67.56%| Coming soon |

### Attacks (doesn't include fine-tuned contrastive yet)

#### Median adversarial distances

| attack type | Baseline | Baseline + Tikhonov    | Contrastive   | Contrastive + Tikhonov  |
|:----:|:-------------:|:-------------:|:-----:|:---:|
pgd  | Coming soon | To do     | 0 (?!) | 0 (?!) |
plb | Coming soon | To do      | 0.12      |   0.185 |

#### Mean adversarial distances

| attack type | Baseline | Baseline + Tikhonov    | Contrastive   | Contrastive + Tikhonov  |
|:----:|:-------------:|:-------------:|:-----:|:---:|
pgd | Coming soon | To do     | 0.0355 | 0.0627 |

#### Max adversarial distances

| attack type |   Baseline| Baseline + Tikhonov    | Contrastive   | Contrastive + Tikhonov  |
|:----:|:-------------:|:-------------:|:-----:|:---:|
pgd | Coming soon | To do     | 0.5 | 0.5 |
plb | Coming soon |  To do    | 0.122      |   nan (?!) |

### Accuracy Claims from [this paper](https://arxiv.org/pdf/2002.05709.pdf)
| Baseline (supervised ResNet50) | Contrastive |
|:--------:|:--------:|
| 93.6%    | 90.6%    |

### TODO
- [x] Adapt Chris's baseline training script to our experiments. In particular, add option to train on a subset of the labels. Run it on 100%, 10% and 1% of the labels.
- [x] Add fine-tuning procedure of the contrastive model by initializing the linear classifier weights to 0 (not its biases).
- [ ] Run the fine-tuning on 100%, 10% and 1% of the labels **<-- In progress**
- [ ] Add Tikhonov Regularization to the baseline training script
- [ ] Train the Tikhonov regularized baseline model and attack the unregularized and regularized version. Compare.
- [ ] Add plots to compare the attack values
- [ ] Implement contrastive loss layer by layer and using class label information instead of purely positive/negative samples
- [ ] Implement layer-by-layer Tikhonov regularization
- [ ] Implement a nonlinear classifier that uses an "average class vector" and give confidence by measuring similarity between class vector and encoder output
2. Another item
⋅⋅* Unordered sub-list. 
1. Actual numbers don't matter, just that it's a number
⋅⋅1. Ordered sub-list
4. And another item.

⋅⋅⋅You can have properly indented paragraphs within list items. Notice the blank line above, and the leading spaces (at least one, but we'll use three here to also align the raw Markdown).

⋅⋅⋅To have a line break without a paragraph, you will need to use two trailing spaces.⋅⋅
⋅⋅⋅Note that this line is separate, but within the same paragraph.⋅⋅
⋅⋅⋅(This is contrary to the typical GFM line break behaviour, where trailing spaces are not required.)

* Unordered list can use asterisks
- Or minuses
+ Or pluses







