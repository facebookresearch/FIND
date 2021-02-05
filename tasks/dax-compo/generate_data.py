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
import string 



def put_train_fpa(root, train_N):
    alphabet_input = string.ascii_lowercase[:train_N]
    alphabet_output = string.ascii_uppercase[:train_N] 

    with open(f'{root}/train.src', 'w') as train_src, open(f'{root}/train.dst', 'w') as train_tgt:
        # Primitives
        for train_input, train_output in zip(alphabet_input, alphabet_output):
            print(train_input, file=train_src)
            print(train_output, file=train_tgt)

        # Function
        train_input = 'F ' + alphabet_input[0]
        print(train_input, file=train_src)
        train_output = ' '.join([alphabet_output[0]] * 3)
        print(train_output, file=train_tgt)

def put_train_mdl(root, train_N, rule):

    alphabet_input = string.ascii_lowercase[:train_N]
    alphabet_output = string.ascii_uppercase[:train_N] 

    with open(f'{root}/train.src', 'w') as train_src, open(f'{root}/train.dst', 'w') as train_tgt:
        # Primitives
        for train_input, train_output in zip(alphabet_input, alphabet_output):
            print(train_input, file=train_src)
            print(train_output, file=train_tgt)

        # Function
        train_input = 'F ' + alphabet_input[0]
        print(train_input, file=train_src)
        train_output = ' '.join([alphabet_output[0]] * 3)
        print(train_output, file=train_tgt)

        for char_inp, char_out in zip(alphabet_input[1:], alphabet_output[1:]):
            train_input = 'F ' + char_inp
            print(train_input, file=train_src)
            train_output = rule(char_out)
            print(train_output, file=train_tgt)


def put_test(root, train_N):

    alphabet_input = string.ascii_lowercase[:train_N]
    alphabet_output = string.ascii_uppercase[:train_N] 

    with open(f'{root}/test.src', 'w') as test_src, open(f'{root}/test.dst', 'w') as test_dst:
        test_output = '' # test output is never used

        for char_inp in alphabet_input:
            test_input = char_inp
            print(test_input, file=test_src)
            print(test_output, file=test_dst)

            test_input = 'F '+ char_inp
            print(test_input, file=test_src)
            print(test_output, file=test_dst)

    shutil.copy(f'{root}/test.src', f'{root}/valid.src')
    shutil.copy(f'{root}/test.dst', f'{root}/valid.dst')

def main(train_N):
    try:
        shutil.rmtree(str(train_N))
    except:
        pass

    pathlib.Path(str(train_N)).mkdir()

    generate_fpa(root=f'{train_N}/fpa/', train_N=train_N)

    alphabet_input = string.ascii_lowercase[:train_N]
    alphabet_output = string.ascii_uppercase[:train_N] 

    mem_modifier = lambda _: ' '.join([alphabet_output[0]] * 3)
    mem_prim = lambda x : x
    compo = lambda x: ' '.join([x] * 3)

    generate_mdl(root=f'{train_N}/mem_func/', train_N=train_N, rule=mem_modifier)
    generate_mdl(root=f'{train_N}/mem_prim/', train_N=train_N, rule=mem_prim)
    generate_mdl(root=f'{train_N}/compo/', train_N=train_N, rule=compo)


def generate_fpa(root, train_N):
    root_raw = f'{root}/data/'
    pathlib.Path(root_raw).mkdir(parents=True)

    put_train_fpa(root_raw, train_N)
    put_test(root_raw, train_N)

    root_bin = f'./{root}/data-bin/'
    pathlib.Path(root_bin).mkdir(parents=True)

    command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
               'dst', '--destdir', root_bin, 
               '--trainpref', f'{root_raw}/train', 
               '--validpref', f'{root_raw}/valid', 
               '--testpref', f'{root_raw}/test']

    subprocess.check_call(command)

def generate_mdl(root, train_N, rule):
    root_raw = f'{root}/data/'
    pathlib.Path(root_raw).mkdir(parents=True)

    put_test(root_raw, train_N)
    put_train_mdl(root_raw, train_N, rule)

    root_bin = f'./{root}/data-bin/'
    pathlib.Path(root_bin).mkdir(parents=True)

    command = ['fairseq-preprocess', '--source-lang', 'src', '--target-lang',
               'dst', '--destdir', root_bin, 
               '--trainpref', f'{root_raw}/train', 
               '--testpref', f'{root_raw}/test',
               '--validpref', f'{root_raw}/valid', 
               ]

    subprocess.check_call(command)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_N", type=int,
                        required=True, help="Number of primitives")
    args = parser.parse_args()

    assert args.train_N > 0
    main(args.train_N)
