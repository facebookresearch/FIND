# Description

This repository contains code for reproducing the main experiments of _What they do when in doubt: a study of inductive biases in seq2seq learners_, Eugene Kharitonov<sup>+</sup> and Rahma Chaabouni<sup>+</sup>. ICLR 2021. [[openreview]](https://openreview.net/forum?id=YmA86Zo-P_t) [[arxiv]](https://arxiv.org/abs/2006.14953).

Armed with the poverty of stimulus principle, we examine inductive biases of standard seq2seq learners. Most notably, we find that LSTM-based learners have strong inductive biases towards arithmetic operations, such as counting, addition, and multiplication by a constant. Similarly, Transformer-based learners are biased for hierarchical reasoning. 

Before running the code, you need to (1) install the required packages, and (2) generate data. After that, you can either tinker with individual learners or train multiple learners in parallel (locally).

<sup>+</sup> indicates equal contribution.

# Table of Contents
- [Installing requirements](#installing-requirements)
- [Generating data](#generating-data)
- [Example usage](#example-usage)
- [How the repo is organized](#how-the-repo-is-organized)
- [Parallel training tool](#parallel-training-tool)
- [Experiments with FPA](#experiments-with-fpa)
- [Experiments with description length](#experiments-with-description-length)
- [Reproducibility note](#reproducibility-note)
- [How to cite our work](#how-to-cite-our-work)


# Installing requirements
To run the code, you will need to have pytorch (v 1.20+) and fairseq (v0.9.0) installed.
You can build a fresh environment with conda by running these commands in your shell:
```bash
conda create -n inductive python=3.8 -y
conda activate inductive
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch # you might need to change this line, depending on your cuda driver version
pip install 'fairseq==0.9.0' # note that the master branch is incompatible due to interface changes
```

# Generating data
To algorithmically generate the datasets and prepare them for usage by fairseq, run the following command:
```bash
cd tasks && bash generate_data.sh && cd ..
```

# Example usage
After installing requirements and generating data we are all set for running experiments. In this section, we show some example runs before going into further details.

Create a directory to store resulting models:
```bash
mkdir tmp
```
(if it already exists, make sure that it doesn't contain earlier trained models, otherwise fairseq would try to re-use them.)

Now you can train a single model, representing an attention-enabled LSTM seq2seq learner that uses hyperparameters from the main text of paper:
```
rm tmp/*
python mdl.py tasks/add-or-mul/20/fpa/data-bin/ \
  --arch=lstm --disable-validation --save-dir=./tmp \
  --decoder-attention=1 --encoder-embed-dim=16 --encoder-hidden-size=512 \
  --encoder-layers=1 --decoder-embed-dim=16 --decoder-hidden-size=512 \
  --decoder-layers=1 --lr=1e-3 --dropout=0.5  \
  --lr-scheduler=inverse_sqrt --warmup-init-lr=1e-5  \
  --warmup-updates=1000 --optimizer=adam --seed=1 \
  --mdl-batches-per-epoch=3000  --mdl-train-examples=1
```
After some training, you will get the resulting model in `./tmp/0.pt`. We can use it to generate sequences to inputs in 
`tasks/add-or-mul/20/fpa/data/test*`:

```bash
python generate.py tasks/add-or-mul/20/fpa/data-bin/ --path=tmp/0.pt
```
It will output something like this:
```json
{"src": "a a a a a a a a a a a a a a a a a a a a a a a", "pred": "b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b", "src_len": 23, "pred_len": 46}
{"src": "a a a a a a a a a a a a a a a a a a a a a a", "pred": "b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b", "src_len": 22, "pred_len": 44}
{"src": "a a a a a a a a a a a a a a a a a a a a a", "pred": "b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b", "src_len": 21, "pred_len": 42}
....
```
which indicates that the model follows the multiplicative generalization rule -- the output length is twice the input length.

The parameters we use to specify architectures are similar to those of fairseq; we only add a few new parameters:
* `--mdl-batches-per-epoch` sets the number of batches (updates) during training;
* `--mdl-train-examples` specifies the size of the initial training set T: the training data files contain the concatenation of T and H, with first `mdl-train-examples` examples forming T;
* `--mdl-block-size` sets the size of the block when calculating description length;
* `--mdl-batch-size` if specified, the batches are formed by sampling with replacement from the training data. If not, all training data is used (that is, all transmitted until the current step, see the paper for details). Only specified in SCAN experiments.

You can try training a CNN-s2s model:
```bash
rm tmp/*
python mdl.py tasks/add-or-mul/20/fpa/data-bin/ \
  --arch=fconv --disable-validation --save-dir=./tmp \
  --decoder-attention="[True]" --encoder-embed-dim=16 --encoder-layers="[(512,3)]" \
  --decoder-embed-dim=16 --decoder-layers="[(512,3)]" \
  --lr=1e-3 --dropout=0.5  \
  --lr-scheduler=inverse_sqrt --warmup-init-lr=1e-5  \
  --warmup-updates=1000 --optimizer=adam --seed=1 \
  --mdl-batches-per-epoch=3000  --mdl-train-examples=1
```
or a Transformer:
```bash
rm tmp/*
python mdl.py tasks/add-or-mul/20/fpa/data-bin/ \
  --arch=transformer --disable-validation --save-dir=./tmp \
  --decoder-attention-heads=8 --encoder-attention-heads=8 \
  --encoder-layers=1 --decoder-layers=1 \
  --decoder-embed-dim=16 --encoder-embed-dim=16 \
  --encoder-ffn-embed-dim=512 --decoder-ffn-embed-dim=512 \
  --lr=1e-3 --dropout=0.5  \
  --lr-scheduler=inverse_sqrt --warmup-init-lr=1e-5  \
  --warmup-updates=1000 --optimizer=adam --seed=2 \
  --mdl-batches-per-epoch=3000  --mdl-train-examples=1
 ```

The outputs on hold-out inputs can be generated by the same command as before:
```bash
python generate.py tasks/add-or-mul/20/fpa/data-bin/ --path=tmp/0.pt
```
We can check that these models memorize the training output:
```json
...
{"src": "a a a a a a a a a a a a a a a a a a", "pred": "b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b", "src_len": 18, "pred_len": 40}
{"src": "a a a a a a a a a a a a a a a a a", "pred": "b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b b", "src_len": 17, "pred_len": 40}
```


# How the repo is organized
In the following, we describe 
  (1) how the code/data/hyperparameters are organized, 
  (2) how we get estimates of description length,
  (3) how you can run trainings with multiple random seeds.

A high-level outline of the directories is the following:
```
 - hyperparams/  # contains hyperparameter grids that we used in our experiments
 - tasks/ # contains data for Count-or-Memorization, Add-or-Multiply, Hierarchical-or-Linear
     add-or-mul/
     hierar-or-linear/
     comp-or-mem/
     count-or-mem/ # all related for Count-or-Memorization
        10/ # different instances of the task, depending on the example length
        20/
        ...
        40/
          count/ # a dataset that contains training data + extension according to `count` rule
          mem/ # a dataset that contains training data + extension according to `mem` rule
          fpa/ # version used for calculating FPA
            data/ # raw (textual) form of the dataset
              train.src # input for the training example 
              train.dst # output for the training example
              test.src # inputs used for probing after training 
              test.dst # dummy empty sequences
              valid.src # not used
              valid.dst # not used
            data-bin/ # data preprocessed for fairseq

 - mdl.py # code for training models and calculating description length
 - generate.py # code for generating sequences from a trained model; used for calculating FPA
 - local_grid.py # code for local (single machine) parallel training
```
## How datasets are organized
Each task have a few datasets associated with it. First, we vary the length of the training example(s), i.e. count-or-mem/10 and count-or-mem/20 contain data where the training example has length of 10 and 20 respectively.

Further, for each task we have several versions of the dataset, used for (a) calculating FPA, (b) calculating description length for various candidate extensions. The datasets that are used for calculating description length have training data extended by the data generated by a candidate hypothesis and have the original training data as the first example(s). In contrast, the `fpa` dataset only has the initial training data.

  
## Hyperparameters organization
The directory `hyperparams/` contains hyperparameters, run configurations, and random seeds specified in a form of json files (e.g. `hyperparams/default/lstm_attention.json`).

It has two sub-folders, `default` and `hierar-or-linear`, with the first one used for Count-or-Memorize and Add-or-Multiply, and the latter used for Hierarchical-or-Linear. There are no differences in the architectures, only in the values of the parameters `mdl-train-examples` and `mdl-block-size`: Hierarchical-or-Linear takes examples in blocks of size 4.

In the experiments reported in the paper, we used 100 random seeds. Those might be too heavy for a quick experimentation, so we also provide grids with 4 random seeds. Files with those have `_small` suffix (e.g. `hyperparams/default/lstm_attention_small.json` is a four-seed version of `hyperparams/default/lstm_attention.json`).

# Parallel training tool

In the paper, we average the learners' behavior over multiple random seeds. To simplify that, we provide 
a small tool for parallel training on a local machine, `local_grid.py`.

To run training with a small grid file (four random seeds), you can launch the following:
```bash
python local_grid.py --sweep=hyperparams/default/lstm_attention_small.json --task=tasks/add-or-mul/20/fpa/ --n_workers=4
```
The tool would utilize all GPUs and assign them uniformly over `n_workers` (since the models are small-ish, you can have more 
workers than GPU devices). `--task` specifies the task to train for, `--sweep` defines hyperparameter file.

The stdout/stderr, parameters of the training, and resulting models would be saved in
`./results/tasks/add-or-mul/20/fpa/<date and time>/{1,2,3,4}/`.
You can look into results of training and generating e.g. by running:
```bash
less results/tasks/add-or-mul/20/fpa/2020_06_07_06_05_34/3/stdout
```
*NB*: running the full grid (not \_small) would take considerable time, unless you have many workers and GPUs. Most likely, you will need to adapt our scripts to run on multiple nodes in parallel. We do not include such scripts, as they are specific for our environment.


# Experiments with FPA
As we descsribed above, in 'Example Usage', you can train individual learner instances and analyze the generated models. Here we provide commands to train multiple instances in parallel that can be used to get results similar to those reported in the paper. 

Note that: (a) to run trainings across 100 random seeds (as done in the paper) you will need to remove `_small` suffix from the json grids, (b) you might need to change the number of workers. We also provide only commands for one learner per task; they can be changed by changing the sweep file 
(`lstm_attention_small.json`, `lstm_noattention_small.json`, `cnn_small.json`, `transformer.json`). The length of the training example can be changed, too, by changing the task (e.g. `tasks/count-or-mem/40/fpa/` to `tasks/count-or-mem/20/fpa/`).

The generated sequences are reported in the stdout logs of each run.


### Count-or-Mem
```bash
python local_grid.py --sweep=hyperparams/default/lstm_attention_small.json --task=tasks/count-or-mem/40/fpa/ --n_workers=4
```

### Add-or-Multiply
```bash
python local_grid.py --sweep=hyperparams/default/lstm_attention_small.json --task=tasks/add-or-mul/20/fpa/ --n_workers=4
```

### Hierarchical-or-Linear
Transformer:
```bash
python local_grid.py --sweep=hyperparams/hierar-or-linear/transformer_small.json --task=tasks/hierar-or-linear/4/fpa/ --n_workers=4
```
CNN:
```bash
python local_grid.py --sweep=hyperparams/hierar-or-linear/cnn_small.json --task=tasks/hierar-or-linear/4/fpa/ --n_workers=4
```

# Experiments with description length

Generally, it works the same way, with the task being not `fpa`, but some candidate hypothesis. Training/evaluation takes longer, as we re-train from scratch after adding each hold-out example (or a block, see the main text).

You can, as before, train single learners, e.g.
```bash
rm tmp/*
python mdl.py tasks/add-or-mul/20/mem/data-bin/ \
  --arch=fconv --disable-validation --save-dir=./tmp \
  --decoder-attention="[True]" --encoder-embed-dim=16 --encoder-layers="[(512,3)]" \
  --decoder-embed-dim=16 --decoder-layers="[(512,3)]" \
  --lr=1e-3 --dropout=0.5  \
  --lr-scheduler=inverse_sqrt --warmup-init-lr=1e-5  \
  --warmup-updates=1000 --optimizer=adam --seed=1 \
  --mdl-batches-per-epoch=3000  --mdl-train-examples=1
```
(note that `tasks/add-or-mul/20/mem/data-bin/` corresponds to the `mem` candidate).

Again, the results will be output to the console, in a form of json:
```json
{"description_length": 9.631028448317283e-06}
```

We provide below commands for parallel trainings across multiple seeds. This way, the results would be in the stdout log.

### Count-or-Memorize
Mem candidate:
```bash
python local_grid.py --sweep=hyperparams/default/lstm_attention_small.json --task=tasks/count-or-mem/40/mem/ --n_workers=4
```
Count candidate:
```bash
python local_grid.py --sweep=hyperparams/default/lstm_attention_small.json --task=tasks/count-or-mem/40/count/ --n_workers=4
```

### Add-or-Multiply
Mem candidate:
```bash
python local_grid.py --sweep=hyperparams/default/lstm_attention_small.json --task=tasks/add-or-mul/20/mem/ --n_workers=4
```
Add candidate:
```bash
python local_grid.py --sweep=hyperparams/default/lstm_attention_small.json --task=tasks/add-or-mul/20/add/ --n_workers=4
```
Mul candidate:
```bash
python local_grid.py --sweep=hyperparams/default/lstm_attention_small.json --task=tasks/add-or-mul/20/mul/ --n_workers=4
```

### Hierarchical-or-Linear
Linear candidate
```bash
python local_grid.py --sweep=hyperparams/hierar-or-linear/transformer_small.json --task=tasks/hierar-or-linear/4/linear/ --n_workers=4
```

Hierarchical candidate
```bash
python local_grid.py --sweep=hyperparams/hierar-or-linear/transformer_small.json --task=tasks/hierar-or-linear/4/hierar/ --n_workers=4
```

#### Reproducibility note
* when preparing this code and data, we have changed the order of the examples in the training sets, which can result in non-significant changes w.r.t. the numbers reported in the text, 

That's it - thanks for reading!


# License
FIND is CC-BY-NC licensed, as found in the LICENSE file.

# How to cite our work

```
@inproceedings{
kharitonov2021what,
title={What they do when in doubt: a study of inductive biases in seq2seq learners},
author={Eugene Kharitonov and Rahma Chaabouni},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=YmA86Zo-P_t}
}
```
