import os
import argparse
import subprocess
from pathlib import Path
from typing import Optional

envs = os.environ.get('PATH').split(':')


def check_pnnx(envs: str) -> Optional[Path]:
    for env in envs:
        env = Path(env)
        if env.is_file():
            if env.stem == 'pnnx':
                return env
        elif env.is_dir():
            for i in env.iterdir():
                if i.name == 'pnnx':
                    return i
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('torchscript', help='TORCHSCRIPT path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640],
                        help='Image size of height and width')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--device', default='cpu', help='Device used for export')
    parser.add_argument('--fp16', action='store_true', help='FP16 model export')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    exe = check_pnnx()
    assert isinstance(exe, Path), 'NOT FOUND PNNX'
    assert os.path.exists(args.onnx), f'NOT FOUND {args.onnx}'
    input_shape = [args.batch_size, 3, *args.img_size]
    cmd = f'pnnx {args.torchscript} inputshape={str(input_shape)} device={args.device}'
    if args.fp16:
        cmd += ' fp16=1'
    subprocess.run(cmd.split(), check=True, env=os.environ)
    print(f'NCNN export success, save into ./')


if __name__ == '__main__':
    args = parse_args()
    args.envs = envs
    main(args)
