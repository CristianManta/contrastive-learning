# ContrastiveTeam0

## A place to push new ideas for Contrastive Learning


## Comparisons between the baseline ResNet50 and our contrastive model on CIFAR-10

## Accuracy 

|   pct. of labels available    | Baseline   | Contrastive  |
| ------------- |:-------------:| -----:|
| 100%      | 80% | 74.79% |
| 10%      | 55.3%      |   70.83% |
| 1% | 32.1%      |    67.56% |

## Attacks

### Median adversarial distances

| attack type |   Baseline    | Unregularized contrastive   | Regularized contrastive  |
---- | ------------- |:-------------:| -----:|
pgd  | 0 (?!)     | 0 (?!) | 0 (?!) |
plb | 0.118      | 0.12      |   0.185 |

### Mean adversarial distances

| attack type |   Baseline    | Unregularized contrastive   | Regularized contrastive  |
---- | ------------- |:-------------:| -----:|
pgd | 0.0868      | 0.0355 | 0.0627 |

### Max adversarial distances

| attack type |   Baseline    | Unregularized contrastive   | Regularized contrastive  |
---- | ------------- |:-------------:| -----:|
pgd | 0.5      | 0.5 | 0.5 |
plb | 0.699      | 0.122      |   nan (?!) |

### Accuracy Claims from [a Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/pdf/2002.05709.pdf)
| Baseline | Contrastive |
| -------- | -------- |
| 93.6%    | 90.6%    |







