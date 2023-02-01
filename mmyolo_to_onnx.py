import argparse
import os
import json
import onnx
import torch
from torch.nn import Module
from io import BytesIO

from utils.model import WarpperModel
from mmdet.apis import init_detector
from mmengine.utils.path import mkdir_or_exist

from mmyolo.utils import register_all_modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--work-dir', default='./work_dir', help='Path to save export model')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='Image size of height and width')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--opset', type=int, default=11, help='ONNX opset version')
    parser.add_argument('--simplify', action='store_true', help='Simplify onnx model by onnx-sim')
    parser.add_argument('--device', default='cpu', help='Device used for export')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1
    return args


def build_model_from_cfg(config_path: str, checkpoint_path: str, device: str) -> Module:
    model = init_detector(config_path, checkpoint_path, device=device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    register_all_modules()

    mkdir_or_exist(args.work_dir)
    save_path = os.path.join(args.work_dir, 'exported.onnx')
    mean_std_path = os.path.join(args.work_dir, 'mean_std.json')

    baseModel = build_model_from_cfg(args.config, args.checkpoint, args.device)

    deploy_model = WarpperModel(baseModel=baseModel, backend='torchscript')
    deploy_model.eval()

    mean = deploy_model.mean.flatten().tolist()
    std = deploy_model.std.flatten().tolist()

    with open(mean_std_path, 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)

    fake_input = torch.randn(args.batch_size, 3, *args.img_size).to(args.device)
    # dry run
    outputs = deploy_model(fake_input)

    output_names = [
        f'{"bbox" if i % 2 else "score"}_{o.shape[1]}x{o.shape[2]}'
        for i, o in enumerate(outputs)]

    with BytesIO() as f:
        torch.onnx.export(
            deploy_model,
            fake_input,
            f,
            input_names=['images'],
            output_names=output_names,
            opset_version=args.opset)
        f.seek(0)
        onnx_model = onnx.load(f)
        onnx.checker.check_model(onnx_model)

    if args.simplify:
        try:
            import onnxsim
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            print(f'Simplify failure: {e}')
    onnx.save(onnx_model, save_path)
    print(f'ONNX export success, save into {save_path}')


if __name__ == '__main__':
    main()
