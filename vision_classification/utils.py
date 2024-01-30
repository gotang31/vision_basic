import argparse
import sys
import json

def x(**kwargs):
    print(**kwargs)

sys_argv = sys.argv[1:]

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size',
    help='Batch size to use for training',
    default=64,
    type=int,)

parser.add_argument('--num-workers',
    help='Number of worker processes for background data loading',
    default=8,
    type=int,)

parser.add_argument('--epochs',
    help='Number of epochs to train for',
    default=10,
    type=int,)

parser.add_argument('--dataset',
    help="What to dataset to feed the model.",
    action='store',
    default='',)

parser.add_argument('--model',
    help="What to model class name to use.",
    action='store',
    default='',)

parser.add_argument('--num_classes',
    help="What to model class name to use.",
    default = 80,
    type = int)

parser.add_argument('--freeze_depth',
    help="What to model class name to use.",
    default = 1,
    type = int)

parser.add_argument('--val-datasets',
    help="Directory where you made the validation datasets of dataframe format by default",
    default = 'val2017.csv',
    type = str,)

cli = parser.parse_args(sys_argv)

print(cli.batch_size)
print(cli.num_workers)
print(cli.val_datasets)
print(vars(cli))

with open('prct.json', 'w') as fp:
    json.dump(vars(cli), fp)