import os
import argparse
import subprocess
from pathlib import Path
from typing import Optional

envs = os.environ.get('PATH').split(':')


def check_onnx2ncnn() -> Optional[Path]:
    global envs
    for env in envs:
        env = Path(env)
        if env.is_file():
            if env.stem == 'onnx2ncnn':
                return env
        elif env.is_dir():
            for i in env.iterdir():
                if i.name == 'onnx2ncnn':
                    return i
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx', help='ONNX path')
    parser.add_argument('output', help='NCNN save dir')
    parser.add_argument('--fp16', action='store_true', help='FP16 model export')
    args = parser.parse_args()
    args.onnx = Path(args.onnx)
    args.output = Path(args.output)
    if args.fp16:
        print('Warning: DO NOT SUPPORT FP16 !!')
    return args


def main(args: argparse.Namespace):
    exe = check_onnx2ncnn()
    assert isinstance(exe, Path), 'NOT FOUND ONNX2NCNN'
    assert args.onnx.exists(), f'NOT FOUND {args.onnx}'
    param = args.output / args.onnx.stem + '.param'
    bin = args.output / args.onnx.stem + '.bin'
    cmd = f'{exe} {args.onnx} {param} {bin}'
    subprocess.run(cmd.split(), check=True, env=os.environ)
    print(f'NCNN export success, save into {param} and {bin}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
