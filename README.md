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

| attack type |   Baseline    | Unregularized contrastive   | Regularized contrastive  |
|:----:|:-------------:|:-------------:|:-----:|
pgd  | To do     | 0 (?!) | 0 (?!) |
plb | To do      | 0.12      |   0.185 |

#### Mean adversarial distances

| attack type |   Baseline    | Unregularized contrastive   | Regularized contrastive  |
|:----:|:-------------:|:-------------:|:-----:|
pgd | To do      | 0.0355 | 0.0627 |

#### Max adversarial distances

| attack type |   Baseline    | Unregularized contrastive   | Regularized contrastive  |
|:----:|:-------------:|:-------------:|:-----:|
pgd | To do      | 0.5 | 0.5 |
plb | To do      | 0.122      |   nan (?!) |

### Accuracy Claims from [this paper](https://arxiv.org/pdf/2002.05709.pdf)
| Baseline (supervised ResNet50) | Contrastive |
|:--------:|:--------:|
| 93.6%    | 90.6%    |







