## List of Pre-trained Models

The structure of the listings are as follows:

`<SEED>-<ID>: (hyper-parameters)`

### mini-Imagenet

- 100-0: --hidden-size 1
- 100-1: (default)
- 100-2: --hidden-size 1 --epoch 20
- 100-4: --meta-learner full-lstm --hidden-size 1
- 100-6: --meta-learner gru --hidden-size 1
- 100-8: --hidden-size 4
- 100-11: --n-shot 1 --hidden-size 1
- 100-13: --hidden-size 1 (trained without data augmentation)
- 100-36: --meta-learner sgd (hidden size does not mean anything)

### Omniglot

- 150-0: (default)
- 150-1: --hidden-size 1

### CUB

- 75-4: --hidden-size 1
- 75-5: (default)
