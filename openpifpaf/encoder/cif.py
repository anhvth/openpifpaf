import dataclasses
import logging
from typing import ClassVar

import numpy as np
import torch

from .annrescaler import AnnRescaler
from .. import headmeta
from ..visualizer import Cif as CifVisualizer
from ..utils import create_sink, mask_valid_area
from avcv import get_name
LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class Cif:
    meta: headmeta.Cif
    rescaler: AnnRescaler = None
    v_threshold: int = 0
    bmin: float = 0.1  #: in pixels
    visualizer: CifVisualizer = None

    side_length: ClassVar[int] = 4
    padding: ClassVar[int] = 10
    
    def __call__(self, image, anns, meta):
        # init a cif-generator then call it
        cif = CifGenerator(self)
        return cif(image, anns, meta)


class CifGenerator():
    def __init__(self, config: Cif):
        self.config = config
        stride = config.meta.stride
        self.rescaler = config.rescaler or AnnRescaler(
            stride, config.meta.pose)
        self.visualizer = config.visualizer or CifVisualizer(config.meta)

        self.intensities = None
        self.fields_reg = None
        self.fields_bmin = None
        self.fields_scale = None
        self.fields_reg_l = None

        self.sink = create_sink(config.side_length)
        self.s_offset = (config.side_length - 1.0) / 2.0

    def __call__(self, image, anns, img_meta):
        width_height_original = image.shape[2:0:-1]

        keypoint_sets = self.rescaler.keypoint_sets(anns)
        bg_mask = self.rescaler.bg_mask(anns, width_height_original,
                                        crowd_margin=(self.config.side_length - 1) / 2)
        valid_area = self.rescaler.valid_area(img_meta)
        LOG.debug('valid area: %s, pif side length = %d', valid_area, self.config.side_length)

        n_fields = len(self.config.meta.keypoints)
        self.init_fields(n_fields, bg_mask)
        self.fill_keypoint_list(keypoint_sets)
        fields = self.fields(valid_area)

        self.visualizer.processed_image(image)
        img_basename = get_name(img_meta['file_name'])
        name_signature = f'cif-target-{img_basename}'
        self.visualizer.targets(fields, annotation_dicts=anns, name_signature=name_signature)
        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[1] + 2 * self.config.padding
        field_h = bg_mask.shape[0] + 2 * self.config.padding
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg = np.full((n_fields, 2, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_bmin = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_scale = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # bg_mask
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][:, bg_mask == 0] = 1.0
        self.intensities[:, p:-p, p:-p][:, bg_mask == 0] = np.nan

    def fill_keypoint_list(self, keypoint_sets):
        for instance_keypoints in keypoint_sets:
            self.fill_keypoint_single_instance(instance_keypoints)

    def fill_keypoint_single_instance(self, keypoints):
        scale = self.rescaler.scale(keypoints)
        for keypoint_id, xyv in enumerate(keypoints):
            if xyv[2] <= self.config.v_threshold:
                continue

            joint_scale = (
                scale
                if self.config.meta.sigmas is None
                else scale * self.config.meta.sigmas[keypoint_id]
            )

            self.fill_coordinate(keypoint_id, xyv, joint_scale)

    def fill_coordinate(self, keypoint_id, xyv, scale):
        ij = np.round(xyv[:2] - self.s_offset).astype(np.int) + self.config.padding
        minx, miny = int(ij[0]), int(ij[1])
        maxx, maxy = minx + self.config.side_length, miny + self.config.side_length
        if minx < 0 or maxx > self.intensities.shape[2] or \
           miny < 0 or maxy > self.intensities.shape[1]:
            return

        offset = xyv[:2] - (ij + self.s_offset - self.config.padding)
        offset = offset.reshape(2, 1, 1)

        # mask
        sink_reg = self.sink + offset
        sink_l = np.linalg.norm(sink_reg, axis=0)
        mask = sink_l < self.fields_reg_l[keypoint_id, miny:maxy, minx:maxx]
        mask_peak = np.logical_and(mask, sink_l < 0.7)
        self.fields_reg_l[keypoint_id, miny:maxy, minx:maxx][mask] = sink_l[mask]

        # update intensity
        self.intensities[keypoint_id, miny:maxy, minx:maxx][mask] = 1.0
        self.intensities[keypoint_id, miny:maxy, minx:maxx][mask_peak] = 1.0

        # update regression
        patch = self.fields_reg[keypoint_id, :, miny:maxy, minx:maxx]
        patch[:, mask] = sink_reg[:, mask]

        # update bmin
        bmin = self.config.bmin / self.config.meta.stride
        self.fields_bmin[keypoint_id, miny:maxy, minx:maxx][mask] = bmin

        # update scale
        assert np.isnan(scale) or 0.0 < scale < 100.0
        self.fields_scale[keypoint_id, miny:maxy, minx:maxx][mask] = scale

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg = self.fields_reg[:, :, p:-p, p:-p]
        fields_bmin = self.fields_bmin[:, p:-p, p:-p]
        fields_scale = self.fields_scale[:, p:-p, p:-p]

        mask_valid_area(intensities, valid_area)
        mask_valid_area(fields_reg[:, 0], valid_area, fill_value=np.nan)
        mask_valid_area(fields_reg[:, 1], valid_area, fill_value=np.nan)
        mask_valid_area(fields_bmin, valid_area, fill_value=np.nan)
        mask_valid_area(fields_scale, valid_area, fill_value=np.nan)

        return torch.from_numpy(np.concatenate([
            np.expand_dims(intensities, 1),
            fields_reg,
            np.expand_dims(fields_bmin, 1),
            np.expand_dims(fields_scale, 1),
        ], axis=1))
