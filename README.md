## Meta-Learner LSTM for few Shot Learning

TensorFlow implementation of [Optimization as a Model for Few-shot Learning](https://openreview.net/forum?id=rJY0-Kcll) together with extended experiments.

## Getting Started

### Requirements

- Python3
- Python packages in requirements.txt

### Data preparation

- Mini-Imagenet data
  - You can download it from [here](https://drive.google.com/file/d/1rV3aj_hgfNTfCakffpPm7Vhpr1in87CR/view?usp=sharing) (~2.7GB, google drive link)

The data should placed and split properly as follows:
  ```
  - data/
    - miniImagenet/
      - train/
      - val/
      - test/
  - main.py
  - ...
  ```
- Check out the configuration in `main.py` to make sure `--data-root` is properly set

## Running with Default Settings

### Training a Model

main.py is configured to train a default model on mini-ImageNet data set using a random seed.

For training with default settings run:
```bash
python main.py
```

### Testing a Pre-Trained Model

This project also includes pre-trained models that can be used for testing.

For testing a model that has been trained on miniImagenet using the default settings:
```bash
python main.py --mode test --seed 100 --ID 1
```

## Changing the hyper-parameters

In order to change the hyper-parameters you can pass the following command line arguments together with desired values.

- **--n-shot** : How many examples per class for episode-training (k)
- **--n-eval** : How many examples per class for episode-evaluation
- **--n-class** : How many classes (N)
- **--hidden-size** : Hidden size for the meta-learner
- **--lr** : Learning rate
- **--episode** : Episodes for meta-train
- **--episode-test** : Episodes for meta-test
- **--epoch** : Epoch to train for an episode
- **--meta-learner** : Type of the meta-learner to be used (choices=['lstm', 'full-lstm', 'gru', 'sgd'])

This is not an exhaustive list of the all possible arguments. For further information, please check `main.py`.

### Testing with Other Pre-trained Models 

The pre-trained model library also includes models trained with different hyper-parameters.

For example, in order to test the model trained with "hidden-size = 1", you can run

```
python main.py --mode test --seed 100 --ID 0 --hidden-size 1
```

You can check `models.md` for a list of all pre-trained models together with their arguments.

You can check `run.ipynb` for a notebook consisting of the experiments in the paper.

## Other data sets

Directory for other data sets should be also set up in the same way as miniImagenet.

In order to train/test with Omniglot/CUB data, you can run `main_omniglot.py` or `main_CUB.py` in the same way descibed for `main.py`.

### Cross-Domain Scenario

For these tests you do not require the Omniglot/CUB data.

For testing a model trained with CUB data on miniImagenet data set:

```
python main.py --mode test --seed 75 --ID 4 --hidden-size 1
```

For testing a model trained with Omniglot data on miniImagenet data set:

```
python main.py --mode test --seed 150 --ID 1 --hidden-size 1
```

## Authors

* **Stanislas Furrer** 
* **Yiğit Efe Erginbaş** 
* **Mert Kayaalp** 
