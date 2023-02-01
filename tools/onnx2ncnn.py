import os
import argparse
import subprocess
from pathlib import Path
from typing import Optional


def check_onnx2ncnn() -> Optional[Path]:
    envs = os.environ.get('PATH', [])
    for env in envs:
        env = Path(env)
        if env.is_file():
            if env.stem == 'onnx2ncnn':
                return env
        elif env.is_dir():
            for i in env.iterdir():
                if i.stem == 'onnx2ncnn':
                    return env
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx', help='ONNX path')
    parser.add_argument('ncnnparam', help='NCNN param save path')
    parser.add_argument('ncnnbin', help='NCNN bin save path')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    exe = check_onnx2ncnn()
    assert isinstance(exe, Path), 'NOT FOUND ONNX2NCNN'
    assert os.path.exists(args.onnx), f'NOT FOUND {args.onnx}'
    cmd = f'onnx2ncnn {args.onnx} {args.ncnnparam} {args.ncnnbin}'
    subprocess.run(cmd.split(), check=True, env=os.environ)
    print(f'NCNN export success, save into {args.ncnnparam} and {args.ncnnbin}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
