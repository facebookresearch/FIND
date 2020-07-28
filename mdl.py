# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adopted from fairseq https://github.com/pytorch/fairseq/

import sys
import collections
import random
import pathlib
import json

import numpy as np
import torch

from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.criterions import CRITERION_REGISTRY
from fairseq.meters import AverageMeter

def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats

def main(args, init_distributed=False):
    utils.import_user_module(args)

    # Initialize CUDA and distributed training
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup task, (should be default, translation)
    task = tasks.setup_task(args)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    initial_state_checkpoint = str(pathlib.Path(args.save_dir) / 'initial.pt')
    trainer.save_checkpoint(initial_state_checkpoint, {'epoch': 0})

    batches_per_epoch = args.mdl_batches_per_epoch
    batch_size = args.mdl_batch_size
    block_size = args.mdl_block_size
    
    epoch_itr = trainer.get_train_iterator(epoch=0, load_dataset=True)

    examples = list(range(len(epoch_itr.dataset)))

    train_examples = examples[:args.mdl_train_examples]
    test_examples = examples[args.mdl_train_examples:]

    random.shuffle(test_examples)
    blocks =  [train_examples]
    blocks += [test_examples[i:i + block_size] for i in range(0, len(test_examples), block_size)]

    allowed_examples = []
    steps = len(blocks)
    block_cross_entropys = []

    for step in range(steps):
        trainer.load_checkpoint(initial_state_checkpoint, reset_optimizer=True, reset_lr_scheduler=True)

        epoch_itr = trainer.get_train_iterator(epoch=step, load_dataset=False)

        allowed_examples += blocks[step]

        # if mdl-batch-size is set, we sample batches with replacement,
        # otherwise, each batch contains all allowed_examples
        if batch_size:
            batches = tuple([random.choices(allowed_examples, k=batch_size) for _ in range(batches_per_epoch)])
        else:
            batches = tuple([allowed_examples for _ in range(batches_per_epoch)])

        epoch_itr.frozen_batches = batches

        train(args, trainer, task, epoch_itr)

        stashed_criterion = trainer.criterion
        train.criterion = CRITERION_REGISTRY['cross_entropy'](args, task)
        
        if step < steps - 1:
            stashed_criterion = trainer.criterion
            train.criterion = CRITERION_REGISTRY['cross_entropy'](args, task)
            next_block = (blocks[step + 1], )
            next_block_cross_entropy = validate(args, trainer, task, epoch_itr, subsets=['train'], \
                allowed_batches=next_block)
            train.criterion = stashed_criterion
            block_cross_entropys.append(next_block_cross_entropy)

        trainer.set_num_updates(0) #reset the num_update as not systematically updated in load_checkpoint
        state_checkpoint = str(pathlib.Path(args.save_dir) / f'{step}.pt')
        trainer.save_checkpoint(state_checkpoint, {'epoch': step})


    examples_seen = [len(b) for b in blocks]
    cross_entropy_sum = sum(n_examples * mean_cross_entropy for n_examples, mean_cross_entropy in zip(examples_seen[1:], block_cross_entropys))
    stats = dict(online_cross_entropy=block_cross_entropys,
                description_length=cross_entropy_sum,
                examples_seen=examples_seen)
    print(json.dumps(stats))
    
    state_checkpoint = str(pathlib.Path(args.save_dir) / 'last.pt')
    trainer.save_checkpoint(state_checkpoint, {'epoch': step})


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    # Update parameters every N batches
    update_freq = 1

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
        fix_batches_to_gpus=args.fix_batches_to_gpus,
        shuffle=False, # TODO: changed
    )

    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch, no_progress_bar='simple',
    )

    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        progress.log(stats, tag='train', step=stats['num_updates'])

    stats = get_training_stats(trainer)
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in ['train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def validate(args, trainer, task, epoch_itr, subsets, allowed_batches):
    """Evaluate the model on the validation set(s) and return the losses."""

    assert len(subsets) == 1

    valid_epoch_itr = task.get_batch_iterator(
        dataset=task.dataset(subsets[0]),
        max_tokens=args.max_tokens_valid,
        max_sentences=args.max_sentences_valid,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            trainer.get_model().max_positions(),
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        seed=args.seed,
        num_shards=args.distributed_world_size,
        shard_id=args.distributed_rank,
        num_workers=args.num_workers,
    )

    valid_epoch_itr.frozen_batches = allowed_batches

    itr = valid_epoch_itr.next_epoch_itr(shuffle=False)
    progress = progress_bar.build_progress_bar(
        args, itr, epoch_itr.epoch,
        prefix='next block validation',
        no_progress_bar='simple'
    )

    # reset validation loss meters
    for k in ['valid_loss', 'valid_nll_loss', 'loss']:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()

    for sample in progress:
        log_output = trainer.valid_step(sample)

    # log validation stats
    valid_loss = trainer.get_meter('valid_loss').avg

    return valid_loss


def cli_main(args):
    parser = options.get_training_parser()
    parser.add_argument("--mdl-block-size", type=int, default=1, 
        help="Size of the transmitted block. Used when calculating description length")
    parser.add_argument("--mdl-batches-per-epoch", type=int, default=3000, help="Number of updates in per training")
    parser.add_argument("--mdl-batch-size", type=int, default=None, help="If set, specifies the number of examples sampled (with replacement) "
                "for each update of the learner. If not specified, all examples available at the step are used.")
    parser.add_argument("--mdl-train-examples", type=int, default=None, required=True, 
            help="First `mdl-train-examples`  lines in the training dataset are considered as initial training data (see README).")
    args = options.parse_args_and_arch(parser, input_args=args)

    assert torch.cuda.is_available()
    assert args.mdl_train_examples

    if not args.sentence_avg:
        print('Overriding --sentence-avg', file=sys.stderr)
        args.sentence_avg = True

    # override multi-gpu logic
    args.distributed_world_size = 1
    main(args)

if __name__ == '__main__':
    cli_main(sys.argv[1:])