import os
import argparse
import subprocess


def check_openvino():
    try:
        import openvino.inference_engine as ie
    except ImportError:
        return False
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx', help='ONNX path')
    parser.add_argument('output', help='OPENVINO save dir')
    parser.add_argument('--fp16', action='store_true', help='FP16 model export')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    status = check_openvino()
    assert status, 'NOT FOUND OPENVINO'
    assert os.path.exists(args.onnx), f'NOT FOUND {args.onnx}'
    cmd = f'mo --input_model {args.onnx} --output_dir {args.output} --data_type {"FP16" if args.fp16 else "FP32"}'
    subprocess.run(cmd.split(), check=True, env=os.environ)
    print(f'OPENVINO export success, save into {args.output}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
