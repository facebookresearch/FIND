# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from mdl import cli_main as train_main
from generate import cli_main as generate_main
from concurrent.futures import ProcessPoolExecutor, wait
import itertools
import json
import time
import subprocess
import os
import torch

class ConcurrentWrapper:
    def __init__(self, runnable, log_dir, job_id):
        self.runnable = runnable
        self.args = None
        self.log_dir = log_dir
        self.job_id = job_id

    def __call__(self, args):
        stdout_path = pathlib.Path(self.log_dir) / 'stdout'
        self.stdout = open(stdout_path, 'w')

        stderr_path = pathlib.Path(self.log_dir) / 'stderr'
        self.stderr = open(stderr_path, 'w')

        sys.stdout = self.stdout
        sys.stderr = self.stderr
        cuda_id = -1
        n_devices = torch.cuda.device_count()
        if n_devices > 0:
            cuda_id = self.job_id % n_devices
        print(f'# {json.dumps(args)}', flush=True)

        with torch.cuda.device(cuda_id):
            self.runnable(args)

def parse_json_sweep(config):
    config = { k: v if type(v) is list else [v] for k, v in config.items() }
    perms = list(itertools.product(*config.values()))

    def to_arg(k, v):
        if type(v) in (int, float):
            return f"--{k}={v}"
        elif type(v) is bool:
            return f"--{k}" if v else ""
        elif type(v) is str:
            assert '"' not in v, f"Key {k} has string value {v} which contains forbidden quotes."
            return f'--{k}={v}'
        else:
            raise Exception(f"Key {k} has value {v} of unsupported type {type(v)}.")

    commands = []
    for p in perms:
        args = [to_arg(k, p[i]) for i, k in enumerate(config.keys())]
        commands.append(args)
    return commands


def sweep(fname):
    with open(fname, 'r') as config_file:
        config = json.loads(config_file.read())
    return parse_json_sweep(config)

def combined_run(params):
    train_main(params)

    checkpoint_path = "--path=" + params[1].split('=')[1] + "/0.pt"
    generate_params = [params[0].strip(), checkpoint_path, '--beam=1', 
                '--batch-size=128', '--gen-subset=test']
    generate_main(generate_params)

if __name__ == '__main__':
    import pathlib
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--sweep", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--n_workers", type=int, default=None)
    parser.add_argument("--task", type=str)

    args = parser.parse_args()

    if args.name is None: args.name = args.task
    assert args.sweep and args.name

    if not args.n_workers:
        device_count = torch.cuda.device_count()
        print(f'n_workers is not specified, using cuda.device_count instead ({device_count})')
        args.n_workers = device_count

    data_path = pathlib.Path(__file__).parent.absolute() / args.task / 'data-bin'
    print(data_path)
    assert data_path.exists()

    args.root_dir = pathlib.PosixPath('./results') / args.name / time.strftime("%Y_%m_%d_%H_%M_%S")
    args.root_dir.mkdir(parents=True)

    hyper_grid = sweep(args.sweep)

    jobs_array = []

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        for combo_id, combo in enumerate(hyper_grid):
            path = args.root_dir / str(combo_id)
            path.mkdir()

            train_params = [str(data_path), f'--save-dir={str(path)}', 
                            '--disable-validation', '--no-epoch-checkpoints', '--sentence-avg'] + combo

            with open(path / 'params', 'w') as f:
                json.dump(dict(train_params=train_params), f)

            runner = ConcurrentWrapper(runnable=combined_run,
                                        log_dir=path,
                                        job_id=combo_id)
            job = executor.submit(runner, train_params)
            print(' '.join(train_params))
            jobs_array.append(job)

    wait(jobs_array)
    print(f'Results are in {args.root_dir}')
    print(f'Check all of them: `cat {args.root_dir}/?/stdout | less`')
