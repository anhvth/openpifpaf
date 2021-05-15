import argparse

import torch

import openpifpaf
from avcv import *

from .dataset import CocoDataset
from .constants import (
    PARKING_LINE_SKELETON,
    PARKING_KEYPOINTS,
    PARKING_LINE_SIGMAS,
    PARKING_POSE,
    # COCO_CATEGORIES,
    # PARKING_KEYPOINTS,
    # PARKING_PERSON_SKELETON,
    # PARKING_PERSON_SIGMAS,
    PARKING_CATEGORIES,
    PARKING_LINE_SCORE_WEIGHTS,
    # PARKING_POSE,
    # DENSER_COCO_PERSON_CONNECTIONS,
    HFLIP,
)

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass

ROOT_DIR = '/data/fisheye-parking/1k8_12Mar/'
class ParkingLineKp(openpifpaf.datasets.DataModule, openpifpaf.Configurable):
    train_annotations = osp.join(ROOT_DIR, 'train_keypoints.json')
    val_annotations = osp.join(ROOT_DIR, 'train_keypoints.json')
    eval_annotations = val_annotations
    train_image_dir = osp.join(ROOT_DIR, 'train', 'image')
    val_image_dir = osp.join(ROOT_DIR, 'train', 'image')
    eval_image_dir = val_image_dir

    square_edge = 385
    with_dense = False
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1
    bmin = 0.1

    eval_annotation_filter = True
    eval_long_edge = 641
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    skeleton = PARKING_LINE_SKELETON

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        cif = openpifpaf.headmeta.Cif('cif', 'parking_line_kp',
                                      keypoints=PARKING_KEYPOINTS,
                                      sigmas=PARKING_LINE_SIGMAS,
                                      pose=PARKING_POSE,
                                      draw_skeleton=self.skeleton,
                                      score_weights=PARKING_LINE_SCORE_WEIGHTS)
        caf = openpifpaf.headmeta.Caf('caf', 'parking_line_kp',
                                      keypoints=PARKING_KEYPOINTS,
                                      sigmas=PARKING_LINE_SIGMAS,
                                      pose=PARKING_POSE,
                                      skeleton=self.skeleton)
        # dcaf = openpifpaf.headmeta.Caf('caf25', 'parking_line_kp',
        #                                keypoints=PARKING_KEYPOINTS,
        #                                sigmas=PARKING_PERSON_SIGMAS,
        #                                pose=COCO_UPRIGHT_POSE,
        #                                skeleton=DENSER_COCO_PERSON_CONNECTIONS,
        #                                sparse_skeleton=self.skeleton,
        #                                only_in_field_of_view=True)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        # dcaf.upsample_stride = self.upsample_stride
        # self.head_metas = [cif, caf, dcaf] if self.with_dense else [cif, caf]
        self.head_metas = [cif, caf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module parking_line_kp')

        group.add_argument('--parking_line_kp-train-annotations', default=cls.train_annotations,
                           help='train annotations')
        group.add_argument('--parking_line_kp-val-annotations', default=cls.val_annotations,
                           help='val annotations')
        group.add_argument('--parking_line_kp-train-image-dir', default=cls.train_image_dir,
                           help='train image dir')
        group.add_argument('--parking_line_kp-val-image-dir', default=cls.val_image_dir,
                           help='val image dir')

        group.add_argument('--parking_line_kp-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert not cls.with_dense
        group.add_argument('--parking_line_kp-with-dense',
                           default=False, action='store_true',
                           help='train with dense connections')
        assert not cls.extended_scale
        group.add_argument('--parking_line_kp-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--parking_line_kp-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--parking_line_kp-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--parking_line_kp-no-augmentation',
                           dest='parking_line_kp_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--parking_line_kp-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--parking_line_kp-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--parking_line_kp-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')
        group.add_argument('--parking_line_kp-bmin',
                           default=cls.bmin, type=float,
                           help='bmin')

        # evaluation
        eval_set_group = group.add_mutually_exclusive_group()
        eval_set_group.add_argument('--parking_line_kp-eval-test2017', default=False, action='store_true')
        eval_set_group.add_argument('--parking_line_kp-eval-testdev2017', default=False, action='store_true')

        # assert cls.eval_annotation_filter
        # group.add_argument('--coco-no-eval-annotation-filter',
        #                    dest='coco_eval_annotation_filter',
        #                    default=True, action='store_false')
        # group.add_argument('--coco-eval-long-edge', default=cls.eval_long_edge, type=int,
        #                    help='set to zero to deactivate rescaling')
        # assert not cls.eval_extended_scale
        # group.add_argument('--coco-eval-extended-scale', default=False, action='store_true')
        # group.add_argument('--coco-eval-orientation-invariant',
        #                    default=cls.eval_orientation_invariant, type=float)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # parking_line_kp specific
        cls.train_annotations = args.parking_line_kp_train_annotations
        cls.val_annotations = args.parking_line_kp_val_annotations
        cls.train_image_dir = args.parking_line_kp_train_image_dir
        cls.val_image_dir = args.parking_line_kp_val_image_dir

        cls.square_edge = args.parking_line_kp_square_edge
        cls.with_dense = args.parking_line_kp_with_dense
        cls.extended_scale = args.parking_line_kp_extended_scale
        cls.orientation_invariant = args.parking_line_kp_orientation_invariant
        cls.blur = args.parking_line_kp_blur
        cls.augmentation = args.parking_line_kp_augmentation
        cls.rescale_images = args.parking_line_kp_rescale_images
        cls.upsample_stride = args.parking_line_kp_upsample
        cls.min_kp_anns = args.parking_line_kp_min_kp_anns
        cls.bmin = args.parking_line_kp_bmin

        # evaluation
        cls.eval_annotation_filter = args.coco_eval_annotation_filter
        if args.parking_line_kp_eval_test2017:
            cls.eval_image_dir = cls._test2017_image_dir
            cls.eval_annotations = cls._test2017_annotations
            cls.annotation_filter = False
        if args.parking_line_kp_eval_testdev2017:
            cls.eval_image_dir = cls._test2017_image_dir
            cls.eval_annotations = cls._testdev2017_annotations
            cls.annotation_filter = False
        cls.eval_long_edge = args.coco_eval_long_edge
        cls.eval_orientation_invariant = args.coco_eval_orientation_invariant
        cls.eval_extended_scale = args.coco_eval_extended_scale

        if (args.parking_line_kp_eval_test2017 or args.parking_line_kp_eval_testdev2017) \
                and not args.write_predictions and not args.debug:
            raise Exception('have to use --write-predictions for this dataset')

    def _preprocess(self):
        encoders = [openpifpaf.encoder.Cif(self.head_metas[0], bmin=self.bmin),
                    openpifpaf.encoder.Caf(self.head_metas[1], bmin=self.bmin)]
        if len(self.head_metas) > 2:
            encoders.append(openpifpaf.encoder.Caf(self.head_metas[2], bmin=self.bmin))

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.25 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.4 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(PARKING_KEYPOINTS, HFLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            openpifpaf.transforms.RandomChoice(
                [openpifpaf.transforms.RotateBy90(),
                 openpifpaf.transforms.RotateUniform(30.0)],
                [self.orientation_invariant, 0.4],
            ),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=True,
            min_kp_anns=self.min_kp_anns,
            category_ids=[1],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = openpifpaf.transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                None,
                openpifpaf.transforms.RotateBy90(fixed_angle=90),
                openpifpaf.transforms.RotateBy90(fixed_angle=180),
                openpifpaf.transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToKpAnnotations(
                    PARKING_CATEGORIES,
                    keypoints_by_category={1: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[1].skeleton},
                ),
                openpifpaf.transforms.ToCrowdAnnotations(PARKING_CATEGORIES),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = CocoDataset(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            min_kp_anns=self.min_kp_anns if self.eval_annotation_filter else 0,
            category_ids=[1] if self.eval_annotation_filter else [],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [openpifpaf.metric.Coco(
            pycocotools.coco.COCO(self.eval_annotations),
            max_per_image=20,
            category_ids=[1],
            iou_type='keypoints',
        )]
