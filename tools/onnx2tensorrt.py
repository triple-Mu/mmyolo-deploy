import os
import argparse
import subprocess
from pathlib import Path
from typing import Union

envs = os.environ.get('PATH').split(':')


def check_trtexec() -> Union[Path, None, type(os)]:
    global envs
    for env in envs:
        env = Path(env)
        if env.is_file():
            if env.stem == 'trtexec':
                return env
        elif env.is_dir():
            for i in env.iterdir():
                if i.name == 'trtexec':
                    return i
    try:
        import tensorrt as trt
    except ImportError:
        return None
    return trt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx', help='ONNX path')
    parser.add_argument('output', help='TENSORRT engine save path')
    parser.add_argument('--fp16', action='store_true', help='FP16 model export')
    args = parser.parse_args()
    return args


def main(args: argparse.Namespace):
    trt = check_trtexec()
    assert isinstance(trt, (Path, bool)), 'NOT FOUND TRTEXEC OR TENSORRT'
    assert os.path.exists(args.onnx), f'NOT FOUND {args.onnx}'
    if isinstance(trt, Path):
        cmd = f'trtexec --onnx={args.onnx} --saveEngine={args.output}'
        if args.fp16:
            cmd += ' --fp16'
        subprocess.run(cmd.split(), check=True, env=os.environ)
    elif isinstance(trt, type(os)):
        logger = trt.Logger(trt.Logger.INFO)
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = 4 << 30  # 4G
        flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(flag)
        parser = trt.OnnxParser(network, logger)
        assert parser.parse_from_file(args.onnx), f'failed to load ONNX file: {args.onnx}'
        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            print(f'input "{inp.name}" with shape {inp.shape} dtype: {inp.dtype}')
        for out in outputs:
            print(f'output "{out.name}" with shape {out.shape} dtype: {out.dtype}')
        if builder.platform_has_fast_fp16 and args.fp16:
            config.set_flag(trt.BuilderFlag.FP16)
        with builder.build_engine(network, config) as engine, \
                open(args.output, 'wb') as t:
            t.write(engine.serialize())
    else:
        print(f'TENSORRT export failed')
        exit()
    print(f'TENSORRT export success, save into {args.output}')


if __name__ == '__main__':
    args = parse_args()
    main(args)
