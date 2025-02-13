"""Video demo application.

Use --scale=0.2 to reduce the input image size to 20%.
Use --json-output for headless processing.

Example commands:
    python3 -m pifpaf.video --source=0  # default webcam
    python3 -m pifpaf.video --source=1  # another webcam

    # streaming source
    python3 -m pifpaf.video --source=http://127.0.0.1:8080/video

    # file system source (any valid OpenCV source)
    python3 -m pifpaf.video --source=docs/coco/000000081988.jpg

Trouble shooting:
* MacOSX: try to prefix the command with "MPLBACKEND=MACOSX".
"""


import argparse
import json
import logging
import os
import time

import torch

from . import decoder, logger, network, show, transforms, visualizer, __version__
from .stream import Stream

LOG = logging.getLogger(__name__)


class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():  # pylint: disable=too-many-statements,too-many-branches
    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.video',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    network.Factory.cli(parser)
    decoder.cli(parser)
    logger.cli(parser)
    show.cli(parser)
    Stream.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('--source', default='0',
                        help=('OpenCV source url. Integer for webcams. '
                              'Or ipwebcam urls (rtsp/rtmp). '
                              'Use "screen" for screen grabs.'))
    parser.add_argument('--video-output', default=None, nargs='?', const=True,
                        help='video output file or "virtualcam"')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='json output file')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='long edge of input images')
    parser.add_argument('--separate-debug-ax', default=False, action='store_true')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    args = parser.parse_args()

    logger.configure(args, LOG)  # logger first

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
             args.device, torch.cuda.is_available(), torch.cuda.device_count())

    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    Stream.configure(args)
    visualizer.configure(args)

    # check whether source should be an int
    if len(args.source) == 1:
        args.source = int(args.source)

    # standard filenames
    if args.video_output is True:
        args.video_output = '{}.openpifpaf.mp4'.format(args.source)
        if os.path.exists(args.video_output):
            os.remove(args.video_output)
    assert args.video_output is None or not os.path.exists(args.video_output)
    if args.json_output is True:
        args.json_output = '{}.openpifpaf.json'.format(args.source)
        if os.path.exists(args.json_output):
            os.remove(args.json_output)
    assert args.json_output is None or not os.path.exists(args.json_output)

    return args


def main():
    args = cli()

    model, _ = network.Factory().factory()
    model = model.to(args.device)
    processor = decoder.factory(model.head_metas)

    # assemble preprocessing transforms
    rescale_t = None
    if args.long_edge is not None:
        rescale_t = transforms.RescaleAbsolute(args.long_edge, fast=True)
    preprocess = transforms.Compose([
        transforms.NormalizeAnnotations(),
        rescale_t,
        transforms.CenterPadTight(16),
        transforms.EVAL_TRANSFORM,
    ])
    capture = Stream(args.source, preprocess=preprocess)

    annotation_painter = show.AnnotationPainter()
    animation = show.AnimationFrame(
        video_output=args.video_output,
        second_visual=args.separate_debug_ax,
    )

    last_loop = time.perf_counter()
    for (ax, ax_second), (image, processed_image, _, meta) in zip(animation.iter(), capture):
        if ax is None:
            ax, ax_second = animation.frame_init(image)

        visualizer.Base.image(image, meta=meta)
        visualizer.Base.processed_image(processed_image)
        visualizer.Base.common_ax = ax_second if args.separate_debug_ax else ax
        preds = processor.batch(model, torch.unsqueeze(processed_image, 0), device=args.device)[0]

        start_post = time.perf_counter()
        preds = [ann.inverse_transform(meta) for ann in preds]

        if args.json_output:
            with open(args.json_output, 'a+') as f:
                json.dump({
                    'frame': meta['frame_i'],
                    'predictions': [ann.json_data() for ann in preds]
                }, f, separators=(',', ':'))
                f.write('\n')
        if (not args.json_output or args.video_output) \
           and (args.separate_debug_ax or not (args.debug or args.debug_indices)):
            ax.imshow(image)
            annotation_painter.annotations(ax, preds)
        postprocessing_time = time.perf_counter() - start_post

        LOG.info('frame %d, loop time = %.0fms (pre = %.1fms, post = %.1fms), FPS = %.1f',
                 meta['frame_i'],
                 (time.perf_counter() - last_loop) * 1000.0,
                 meta['preprocessing_s'] * 1000.0,
                 postprocessing_time * 1000.0,
                 1.0 / (time.perf_counter() - last_loop))
        last_loop = time.perf_counter()


if __name__ == '__main__':
    main()
