import os
import argparse
from os import path as osp

from PIL import Image

import paddle
import paddle.nn.functional as F
from paddle import inference
from paddle.inference import Config, create_predictor

from data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, create_transform


def str2bool(v):
    return v.lower() in ("true", "t", "1")


# general params
parser = argparse.ArgumentParser(description='Paddle ImageNet Inference model script')
parser.add_argument("--input_file", type=str, help="input file path")
parser.add_argument("--model_file", type=str)
parser.add_argument("--params_file", type=str)

# params for predict
parser.add_argument('--input_size', default=224, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument("-b", "--batch_size", type=int, default=1)
parser.add_argument("--use_gpu", type=str2bool, default=True)
parser.add_argument("--precision", type=str, default="fp32")
parser.add_argument("--ir_optim", type=str2bool, default=True)
parser.add_argument("--use_tensorrt", type=str2bool, default=False)
parser.add_argument("--gpu_mem", type=int, default=8000)
parser.add_argument("--enable_benchmark", type=str2bool, default=False)
parser.add_argument("--enable_mkldnn", type=str2bool, default=False)
parser.add_argument("--cpu_threads", type=int, default=None)
parser.add_argument("--crop_pct", default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument("--mean", type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument("--std", type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument("--interpolation", default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')


def create_paddle_predictor(args):
    config = Config(args.model_file, args.params_file)
    if args.use_gpu:
        config.enable_use_gpu(args.gpu_mem, 0)
    else:
        config.disable_gpu()
        if args.cpu_threads:
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        if args.enable_mkldnn:
            # cache 10 different shapes for mkldnn to avoid memory leak
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()

    # config.disable_glog_info()
    config.switch_ir_optim(args.ir_optim)  # default true
    if args.use_tensorrt:
        # choose precision
        if args.precision == "fp16":
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32

        # calculate real max batch size during inference when tenrotRT enabled
        num_seg = 1
        num_views = 1
        max_batch_size = args.batch_size * num_views * num_seg
        config.enable_tensorrt_engine(precision_mode=precision,
                                      max_batch_size=max_batch_size)

    config.enable_memory_optim()
    # use zero copy
    config.switch_use_feed_fetch_ops(False)
    predictor = create_predictor(config)

    return config, predictor


def parse_file_paths(input_path: str) -> list:
    if osp.isfile(input_path):
        files = [
            input_path,
        ]
    else:
        files = os.listdir(input_path)
        files = [
            file for file in files
            if (file.endswith(".jpg"))
        ]
        files = [osp.join(input_path, file) for file in files]
    return files


def main(args):
    inference_config, predictor = create_paddle_predictor(args)

    # get the absolute file path(s) to be processed
    files = parse_file_paths(args.input_file)

    if args.enable_benchmark:
        num_warmup = 0

        # instantiate auto log
        import auto_log
        pid = os.getpid()
        autolog = auto_log.AutoLogger(
            model_name='paddle',
            model_precision=args.precision,
            batch_size=args.batch_size,
            data_shape="dynamic",
            save_path="./output/auto_log.lpg",
            inference_config=inference_config,
            pids=pid,
            process_name=None,
            gpu_ids=0 if args.use_gpu else None,
            time_keys=['preprocess_time', 'inference_time', 'postprocess_time'],
            warmup=num_warmup)

    # eval transform
    interpolation = 'bicubic' \
        if args.interpolation is None or args.interpolation == 'random' else args.interpolation
    preprocess = create_transform(
        input_size=args.input_size,
        interpolation=interpolation,
        mean=args.mean or IMAGENET_DEFAULT_MEAN,
        std=args.std or IMAGENET_DEFAULT_STD,
        crop_pct=args.crop_pct)

    # Inferencing process
    batch_num = args.batch_size
    for st_idx in range(0, len(files), batch_num):
        ed_idx = min(st_idx + batch_num, len(files))

        # auto log start
        if args.enable_benchmark:
            autolog.times.start()

        # Pre process batched input
        batched_inputs = [files[st_idx:ed_idx]]
        imgs = []
        deal_imgs_name = []
        for inp in batched_inputs[0]:
            deal_imgs_name.append(inp)
            precess_im = preprocess(Image.open(inp).convert("RGB"))  # preprocess
            imgs.append(precess_im)
        imgs = paddle.stack(imgs, axis=0)
        batched_inputs = [imgs.cpu().numpy()]
        # get pre process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        # run inference
        input_names = predictor.get_input_names()
        for i, name in enumerate(input_names):
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(batched_inputs[i].shape)
            input_tensor.copy_from_cpu(batched_inputs[i].copy())

        # do the inference
        predictor.run()

        # get inference process time cost
        if args.enable_benchmark:
            autolog.times.stamp()

        # get out data from output tensor
        results = []
        # get out data from output tensor
        output_names = predictor.get_output_names()
        for i, name in enumerate(output_names):
            output_tensor = predictor.get_output_handle(name)
            output_data = output_tensor.copy_to_cpu()
            results.append(output_data)

        class_map = {}
        with open('demo/imagenet1k_label_list.txt', 'r') as f:
            for line in f.readlines():
                cat_id, *name = line.split('\n')[0].split(' ')
                class_map[int(cat_id)] = ' '.join(name)

        preds = []
        result = paddle.to_tensor(results[0])
        for file_name, scores, class_ids in zip(deal_imgs_name, *F.softmax(result).topk(5, 1)):
            preds.append({
                'class_ids': class_ids.tolist(),
                'scores': scores.tolist(),
                'file_name': file_name,
                'label_names': [class_map[i] for i in class_ids.tolist()]
            })
        print(preds)

        # get post process time cost
        if args.enable_benchmark:
            autolog.times.end(stamp=True)

    # report benchmark log if enabled
    if args.enable_benchmark:
        autolog.report()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
