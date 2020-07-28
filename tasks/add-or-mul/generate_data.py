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

def put_train_fpa(root, train_length):
    with open(f'{root}/train.src', 'w') as train_src, open(f'{root}/train.dst', 'w') as train_tgt:
        train_input = ' '.join(['a'] * train_length)
        print(train_input, file=train_src)

        train_output = ' '.join(['b'] * (2 * train_length))
        print(train_output, file=train_tgt)

def put_train_mdl(root, train_length, test_from, test_to, rule):
    with open(f'{root}/train.src', 'w') as train_src, open(f'{root}/train.dst', 'w') as train_tgt:
        train_input = ' '.join(['a'] * train_length)
        print(train_input, file=train_src)
        train_output = ' '.join(['b'] * (2 * train_length))
        print(train_output, file=train_tgt)


        for length in range(test_from, test_to, 1):
            if length == train_length: continue

            train_input = ' '.join(['a'] * length)
            print(train_input, file=train_src)
            train_output = ' '.join(['b'] * rule(length))
            print(train_output, file=train_tgt)

def put_test(root, length_from, length_to):
    with open(f'{root}/test.src', 'w') as test_src, open(f'{root}/test.dst', 'w') as test_dst:
        for length in range(length_from, length_to):
            test_input = ' '.join(['a'] * length)
            print(test_input, file=test_src)

            test_output = ''  # test output is never used
            print(test_output, file=test_dst)

    shutil.copy(f'{root}/test.src', f'{root}/valid.src')
    shutil.copy(f'{root}/test.dst', f'{root}/valid.dst')


def main(train_length, test_span):
    try:
        shutil.rmtree(str(train_length))
    except:
        pass

    pathlib.Path(str(train_length)).mkdir()

    generate_fpa(root=f'{train_length}/fpa/', train_length=train_length, test_span=test_span)

    additive = lambda x: x + train_length
    memorization = lambda _: 2 * train_length
    multiplicative = lambda x: 2 * x

    generate_mdl(root=f'{train_length}/mem/', train_length=train_length, test_span=test_span, rule=memorization)
    generate_mdl(root=f'{train_length}/add/', train_length=train_length, test_span=test_span, rule=additive)
    generate_mdl(root=f'{train_length}/mul/', train_length=train_length, test_span=test_span, rule=multiplicative)

def generate_fpa(root, train_length, test_span):
    root_raw = f'{root}/data/'
    pathlib.Path(root_raw).mkdir(parents=True)

    put_train_fpa(root_raw, train_length)
    test_from, test_to = train_length - test_span, train_length + test_span + 1 # not inclusive
    put_test(root_raw, test_from, test_to)

    root_bin = f'./{root}/data-bin/'
    pathlib.Path(root_bin).mkdir(parents=True)

    command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
               'dst', '--destdir', root_bin, 
               '--trainpref', f'{root_raw}/train', 
               '--validpref', f'{root_raw}/valid', 
               '--testpref', f'{root_raw}/test']

    subprocess.check_call(command)

def generate_mdl(root, train_length, test_span, rule):
    root_raw = f'{root}/data/'
    pathlib.Path(root_raw).mkdir(parents=True)


    test_from, test_to = train_length - test_span, train_length + test_span + 1 # not inclusive

    put_test(root_raw, test_from, test_to)
    put_train_mdl(root_raw, train_length, test_from, test_to, rule)

    root_bin = f'./{root}/data-bin/'
    pathlib.Path(root_bin).mkdir(parents=True)

    command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
               'dst', '--destdir', root_bin, 
               '--trainpref', f'{root_raw}/train', 
               '--validpref', f'{root_raw}/valid',
               '--testpref', f'{root_raw}/test']

    subprocess.check_call(command)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_length", type=int,
                        required=True, help="Length of the training example")
    parser.add_argument("--test_span", 
                        type=int,
                        default=3,
                        )
    args = parser.parse_args()

    assert args.train_length > 0
    main(args.train_length, args.test_span)
