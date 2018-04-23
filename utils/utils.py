import os


def print_arguments(args, log_fn):
    with open(log_fn, 'w') as f:
        for k, v in args.items():
            print(k, v, file=f)


def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)