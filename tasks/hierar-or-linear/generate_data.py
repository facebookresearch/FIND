# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pathlib
import shutil
import subprocess
import argparse
import os
import itertools

def put_train_fpa(root, train_depth):
    with open(f'{root}/train.src', 'w') as train_src, open(f'{root}/train.dst', 'w') as train_tgt:
        for central_symbol, wrapper in itertools.product(['a', 'b'], ['a', 'b']):
            train_input = [wrapper] * train_depth + [central_symbol] + [wrapper] * train_depth
            train_input = ' '.join(train_input)
            print(train_input, file=train_src)

            train_output = central_symbol
            print(train_output, file=train_tgt)

def put_train_mdl(root, train_depth, test_from, test_to, rule):
    with open(f'{root}/train.src', 'w') as train_src, open(f'{root}/train.dst', 'w') as train_tgt:
        depths = [train_depth] + [d for d in range(test_from, test_to) if d != train_depth]

        for d in depths:
            for central_symbol, wrapper in itertools.product(['a', 'b'], ['a', 'b']):
                train_input = [wrapper] * d + [central_symbol] + [wrapper] * d
                print(' '.join(train_input), file=train_src)
                print(rule(train_input), file=train_tgt)

def put_test(root, depth_from, depth_to):
    with open(f'{root}/test.src', 'w') as test_src, open(f'{root}/test.dst', 'w') as test_tgt:
        for d in range(depth_from, depth_to):
            for central_symbol, wrapper in itertools.product(['a', 'b'], ['a', 'b']):
                train_input = [wrapper] * d + [central_symbol] + [wrapper] * d
                print(' '.join(train_input), file=test_src)
                print('', file=test_tgt) # not used
                
    shutil.copy(f'{root}/test.src', f'{root}/valid.src')
    shutil.copy(f'{root}/test.dst', f'{root}/valid.dst')

def main(train_depth, test_span):
    try:
        shutil.rmtree(str(train_depth))
    except:
        pass

    pathlib.Path(str(train_depth)).mkdir()

    generate_fpa(root=f'{train_depth}/fpa/', train_depth=train_depth, test_span=test_span)

    linear = lambda seq: seq[train_depth]
    hierar = lambda seq: seq[(len(seq) - 1) // 2]

    generate_mdl(root=f'{train_depth}/linear/', train_depth=train_depth, test_span=test_span, rule=linear)
    generate_mdl(root=f'{train_depth}/hierar/', train_depth=train_depth, test_span=test_span, rule=hierar)

def generate_fpa(root, train_depth, test_span):
    root_raw = f'{root}/data/'
    pathlib.Path(root_raw).mkdir(parents=True)

    put_train_fpa(root_raw, train_depth)
    test_from, test_to = train_depth - test_span, train_depth + test_span + 1 # not inclusive
    put_test(root_raw, test_from, test_to)

    root_bin = f'./{root}/data-bin/'
    pathlib.Path(root_bin).mkdir(parents=True)

    command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
               'dst', '--destdir', root_bin, '--trainpref', f'{root_raw}/train', '--testpref', f'{root_raw}/test']

    subprocess.check_call(command)

def generate_mdl(root, train_depth, test_span, rule):
    root_raw = f'{root}/data/'
    pathlib.Path(root_raw).mkdir(parents=True)


    test_from, test_to = train_depth - test_span, train_depth + test_span + 1 # not inclusive

    put_test(root_raw, test_from, test_to)
    put_train_mdl(root_raw, train_depth, test_from, test_to, rule)

    root_bin = f'./{root}/data-bin/'
    pathlib.Path(root_bin).mkdir(parents=True)

    command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
               'dst', '--destdir', root_bin, '--trainpref', f'{root_raw}/train', '--testpref', f'{root_raw}/test']

    subprocess.check_call(command)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_depth", type=int,
                        required=True, help="Length of the training example")
    parser.add_argument("--test_span", 
                        type=int,
                        default=2,
                        )
    args = parser.parse_args()

    assert args.train_depth > 0
    main(args.train_depth, args.test_span)
