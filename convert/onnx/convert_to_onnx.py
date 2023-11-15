import os.path
import os
import sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '../../'))
sys.path.append(os.path.join(CURRENT_DIR, '../../core'))

import argparse
import torch
from dlnr import DLNR, autocast
import matplotlib.pyplot as plt

DEVICE = 'cuda'

def GetArgs():
    parser = argparse.ArgumentParser(description='Script for inferencing DLNR')
    parser.add_argument("--restore_ckpt", type=str, default=None, help="model path")
    parser.add_argument("-o", "--output",
                        help='Path for saving the adapted model (only needed if performing MAD)',
                        default=None, required=False)
    parser.add_argument("--height", help='Model image input height resolution', type=int, default=480)
    parser.add_argument("--width", help='Model image input height resolution', type=int, default=640)
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")

    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    args = parser.parse_args()

    return args


def main():
    args = GetArgs()

    model = torch.nn.DataParallel(DLNR(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    height = args.height
    width = args.width

    # Initialise the model
    input_L = torch.randn(1, 3, height, width, device='cuda:0')
    input_R = torch.randn(1, 3, height, width, device='cuda:0')

    output_path = "DLNR.onnx" #if args.restore_ckpt is None else os.path.splitext(args.restore_ckpt)[0] + ".onnx"

    if args.output is not None:
        output_path = os.path.join(args.output, os.path.basename(output_path))

    input_names = ['L', 'R']
    output_names = ['disp']

    torch.onnx.export(
        model,
        (input_L,input_R),
        args.output,
        verbose=True,
        opset_version=12,
        input_names=input_names,
        output_names=output_names)

    print("export onnx to {}".format(output_path))


if __name__ == "__main__":
    main()
