import sys

import torch


if __name__ == '__main__':
    argv = sys.argv
    if len(argv) != 3:
        print(f'script.py in_file out_file')
        exit(-1)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    checkpoint = torch.load(in_path)
    new_ckpt = dict(
        model_state_dict=checkpoint['model_state_dict']
    )
    torch.save(new_ckpt, out_path)