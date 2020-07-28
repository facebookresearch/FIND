# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adopted from fairseq https://github.com/pytorch/fairseq/

import torch
from fairseq import checkpoint_utils, options, progress_bar, tasks, utils
import json

def main(args):
    assert args.path is not None, '--path required for generation!'
    args.beam = args.nbest = 1
    args.max_tokens = int(1e4)

    utils.import_user_module(args)


    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    src_dict = getattr(task, 'source_dictionary', None)
    tgt_dict = task.target_dictionary

    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=args.beam,
            need_attn=False
        )
        model.cuda()

    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    generator = task.build_generator(args)

    with progress_bar.build_progress_bar(args, itr) as t:
        for sample in t:
            sample = utils.move_to_cuda(sample)
            if 'net_input' not in sample:
                continue

            prefix_tokens = None

            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)

            for i, sample_id in enumerate(sample['id'].tolist()):
                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())

                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                else:
                    src_str = ""

                # Process top predictions
                hypo = hypos[i][0]
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int().cpu(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=None,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                )

                result = dict(src=src_str, pred=hypo_str, src_len=len(src_str.split()), pred_len=len(hypo_str.split()))
                result_line = json.dumps(result)
                print(result_line)



def cli_main(args):
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser, input_args=args)
    main(args)


if __name__ == '__main__':
    import sys
    cli_main(sys.argv[1:])
