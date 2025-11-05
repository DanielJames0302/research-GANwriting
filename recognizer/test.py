from main_torch_latest import all_data_loader, test
import argparse
import sys

parser = argparse.ArgumentParser(description='test')
parser.add_argument('epoch', type=int, help='epoch that you want to evaluate')
parser.add_argument('--dataset', type=str, default='iam', choices=['iam', 'captcha'],
                    help='Dataset to use: iam or captcha')
parser.add_argument('--captcha_dir', type=str, default='test/correct_test',
                    help='Directory containing captcha images (only used when --dataset=captcha)')
test_args = parser.parse_args()

# Pass arguments to main_torch_latest by setting sys.argv
# This is needed because main_torch_latest parses its own args
original_argv = sys.argv
sys.argv = ['test.py', str(test_args.epoch), '--dataset', test_args.dataset, '--captcha_dir', test_args.captcha_dir]

_, _, test_loader = all_data_loader()
test(test_loader, test_args.epoch, showAttn=False)

# Restore original argv
sys.argv = original_argv
