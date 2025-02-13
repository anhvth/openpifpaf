import argparse
import cv2
from collections import defaultdict
import heapq
import logging
import time
from typing import List

import numpy as np

from .decoder import Decoder
from ..annotation import Annotation
from . import utils
from .. import headmeta, visualizer
import mmcv
# pylint: disable=import-error
from ..functional import caf_center_s, grow_connection_blend

LOG = logging.getLogger(__name__)


class DenseAdapter:
    def __init__(self, cif_meta, caf_meta, dense_caf_meta):
        self.cif_meta = cif_meta
        self.caf_meta = caf_meta
        self.dense_caf_meta = dense_caf_meta

        # overwrite confidence scale
        self.dense_caf_meta.confidence_scales = [
            CifCaf.dense_coupling for _ in self.dense_caf_meta.skeleton
        ]

        concatenated_caf_meta = headmeta.Caf.concatenate(
            [caf_meta, dense_caf_meta])
        self.cifcaf = CifCaf([cif_meta], [concatenated_caf_meta])

    @classmethod
    def factory(cls, head_metas):
        if len(head_metas) < 3:
            return []
        return [
            DenseAdapter(cif_meta, caf_meta, dense_meta)
            for cif_meta, caf_meta, dense_meta in zip(head_metas, head_metas[1:], head_metas[2:])
            if (isinstance(cif_meta, headmeta.Cif)
                and isinstance(caf_meta, headmeta.Caf)
                and isinstance(dense_meta, headmeta.Caf))
        ]

    def __call__(self, fields, initial_annotations=None):
        cifcaf_fields = [
            fields[self.cif_meta.head_index],
            np.concatenate([
                fields[self.caf_meta.head_index],
                fields[self.dense_caf_meta.head_index],
            ], axis=0)
        ]
        return self.cifcaf(cifcaf_fields)


class CifCaf(Decoder):
    """Generate CifCaf poses from fields.

    :param: nms: set to None to switch off non-maximum suppression.
    """
    connection_method = 'blend'
    occupancy_visualizer = visualizer.Occupancy()
    force_complete = False
    force_complete_caf_th = 0.001
    greedy = False
    keypoint_threshold = 0.15
    keypoint_threshold_rel = 0.5
    nms = utils.nms.Keypoints()
    nms_before_force_complete = False
    dense_coupling = 0.0

    reverse_match = True

    def __init__(self,
                 cif_metas: List[headmeta.Cif],
                 caf_metas: List[headmeta.Caf],
                 *,
                 cif_visualizers=None,
                 caf_visualizers=None):
        super().__init__()
        self.cif_metas = cif_metas
        self.caf_metas = caf_metas
        self.skeleton_m1 = np.asarray(self.caf_metas[0].skeleton) - 1
        self.keypoints = cif_metas[0].keypoints
        self.score_weights = cif_metas[0].score_weights
        self.out_skeleton = caf_metas[0].skeleton
        self.confidence_scales = caf_metas[0].decoder_confidence_scales

        self.cif_visualizers = cif_visualizers
        if self.cif_visualizers is None:
            self.cif_visualizers = [visualizer.Cif(meta) for meta in cif_metas]
        self.caf_visualizers = caf_visualizers
        if self.caf_visualizers is None:
            self.caf_visualizers = [visualizer.Caf(meta) for meta in caf_metas]

        # prefer decoders with more keypoints and associations
        self.priority += sum(m.n_fields for m in cif_metas) / 1000.0
        self.priority += sum(m.n_fields for m in caf_metas) / 1000.0

        self.timers = defaultdict(float)

        # init by_target and by_source
        self.by_target = defaultdict(dict)
        for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
            self.by_target[j2][j1] = (caf_i, True)
            self.by_target[j1][j2] = (caf_i, False)
        self.by_source = defaultdict(dict)
        for caf_i, (j1, j2) in enumerate(self.skeleton_m1):
            self.by_source[j1][j2] = (caf_i, True)
            self.by_source[j2][j1] = (caf_i, False)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""
        group = parser.add_argument_group('CifCaf decoder')
        assert not cls.force_complete
        group.add_argument('--force-complete-pose',
                           default=False, action='store_true')
        group.add_argument('--force-complete-caf-th', type=float,
                           default=cls.force_complete_caf_th,
                           help='CAF threshold for force complete. Set to -1 to deactivate.')
        assert not cls.nms_before_force_complete
        group.add_argument('--nms-before-force-complete', default=False, action='store_true',
                           help='run an additional NMS before completing poses')

        assert utils.nms.Keypoints.keypoint_threshold == cls.keypoint_threshold
        group.add_argument('--keypoint-threshold', type=float,
                           default=cls.keypoint_threshold,
                           help='filter keypoints by score')
        group.add_argument('--keypoint-threshold-rel', type=float,
                           default=cls.keypoint_threshold_rel,
                           help='filter keypoint connections by relative score')

        assert not cls.greedy
        group.add_argument('--greedy', default=False, action='store_true',
                           help='greedy decoding')
        group.add_argument('--connection-method',
                           default=cls.connection_method,
                           choices=('max', 'blend'),
                           help='connection method to use, max is faster')
        group.add_argument('--dense-connections', nargs='?', type=float,
                           default=0.0, const=1.0)

        assert cls.reverse_match
        group.add_argument('--no-reverse-match',
                           default=True, dest='reverse_match', action='store_false')
        group.add_argument('--ablation-cifseeds-nms',
                           default=False, action='store_true')
        group.add_argument('--ablation-cifseeds-no-rescore',
                           default=False, action='store_true')
        group.add_argument('--ablation-caf-no-rescore',
                           default=False, action='store_true')
        group.add_argument('--ablation-independent-kp',
                           default=False, action='store_true')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        # force complete
        keypoint_threshold_nms = args.keypoint_threshold
        if args.force_complete_pose:
            if not args.ablation_independent_kp:
                args.keypoint_threshold = 0.0
            args.keypoint_threshold_rel = 0.0
            keypoint_threshold_nms = 0.0
        # check consistency
        if args.seed_threshold < args.keypoint_threshold:
            LOG.warning(
                'consistency: decreasing keypoint threshold to seed threshold of %f',
                args.seed_threshold,
            )
            args.keypoint_threshold = args.seed_threshold

        cls.force_complete = args.force_complete_pose
        cls.force_complete_caf_th = args.force_complete_caf_th
        cls.nms_before_force_complete = args.nms_before_force_complete
        cls.keypoint_threshold = args.keypoint_threshold
        utils.nms.Keypoints.keypoint_threshold = keypoint_threshold_nms
        cls.keypoint_threshold_rel = args.keypoint_threshold_rel

        cls.greedy = args.greedy
        cls.connection_method = args.connection_method
        cls.dense_coupling = args.dense_connections

        cls.reverse_match = args.reverse_match
        utils.CifSeeds.ablation_nms = args.ablation_cifseeds_nms
        utils.CifSeeds.ablation_no_rescore = args.ablation_cifseeds_no_rescore
        utils.CafScored.ablation_no_rescore = args.ablation_caf_no_rescore
        if args.ablation_cifseeds_no_rescore and args.ablation_caf_no_rescore:
            utils.CifHr.ablation_skip = True

    @classmethod
    def factory(cls, head_metas):
        if cls.dense_coupling:
            return DenseAdapter.factory(head_metas)
        return [
            CifCaf([meta], [meta_next])
            for meta, meta_next in zip(head_metas[:-1], head_metas[1:])
            if (isinstance(meta, headmeta.Cif)
                and isinstance(meta_next, headmeta.Caf))
        ]

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        for vis, meta in zip(self.cif_visualizers, self.cif_metas):
            vis.predicted(fields[meta.head_index])
        for vis, meta in zip(self.caf_visualizers, self.caf_metas):
            vis.predicted(fields[meta.head_index])

        cifhr = utils.CifHr().fill(fields, self.cif_metas)
        seeds = utils.CifSeeds(cifhr.accumulated).fill(fields, self.cif_metas)
        caf_scored = utils.CafScored(cifhr.accumulated).fill(fields, self.caf_metas)

        occupied = utils.Occupancy(cifhr.accumulated.shape, 2, min_scale=4)
        predictions = []

        def mark_occupied(ann):
            list_joints = np.flatnonzero(ann.data[:, 2])
            for joint in list_joints:
                width = ann.joint_scales[joint]
                occupied.set(
                    joint,
                    ann.data[joint, 0],
                    ann.data[joint, 1],
                    width,  # width = 2 * sigma
                )

        for ann in initial_annotations:
            self._grow(ann, caf_scored)
            predictions.append(ann)
            mark_occupied(ann)

        # h,w = cifhr.accumulated.shape[1:]
        # i =0
        # lines = []

        for v, f, x, y, s in seeds.get():
            # _mask = np.zeros([h,w,3])
            if occupied.get(f, x, y):
                continue

            ann = Annotation(self.keypoints,# ['p1', 'p2']
                             self.out_skeleton, # [[1, 2], [2, 1]]
                             score_weights=self.score_weights # [3, 3]
                             ).add(f, (x, y, v))
            ann.joint_scales[f] = s
            self._grow(ann, caf_scored)
            predictions.append(ann)
            mark_occupied(ann)

            # Draw _mask
            # p1 = tuple(ann.data[0][:2].astype(int))
            # p2 = tuple(ann.data[1][:2].astype(int))
            # lines.append([p1,p2])


        # lens  = np.array(lines)
        # lens = ((lens[:,0]-lens[:,1])**2).sum(1)
        # lens = np.sqrt(lens)
        # for (p1, p2), l in zip(lines, lens):
        #     if abs(l/np.median(lens) - 1) < 0.4:
        #         print('Line:', p1, '->',p2)
        #         from avcv import put_text
        #         put_text(_mask, p1, f"{v:0.2f}")
        #         cv2.line(_mask, p1, p2, (255,0,0), 3)
        #         import matplotlib.pyplot as plt; plt.imshow(_mask); plt.savefig(f'test_{i}.jpg'); plt.close()
        #         i+=1


        # print('Num of prediction lines:', len(predictions))
        self.occupancy_visualizer.predicted(occupied)

        LOG.debug('annotations %d, %.3fs', len(predictions), time.perf_counter() - start)

        if self.force_complete:
            if self.nms_before_force_complete and self.nms is not None:
                assert self.nms.instance_threshold > 0.0, self.nms.instance_threshold
                predictions = self.nms.annotations(predictions)
            predictions = self.complete_annotations(cifhr, fields, predictions)

        if self.nms is not None:
            predictions = self.nms.annotations(predictions)

        LOG.info('%d annotations: %s', len(predictions),
                 [np.sum(ann.data[:, 2] > 0.1) for ann in predictions])
        return predictions

    def connection_value(self, ann, caf_scored, start_i, end_i, *, reverse_match=True):
        caf_i, forward = self.by_source[start_i][end_i]
        caf_forward, caf_backward = caf_scored.directed(caf_i, forward)
        xyv = ann.data[start_i]
        source_scale = max(0.0, ann.joint_scales[start_i])

        only_max = self.connection_method == 'max'

        new_xysv = grow_connection_blend(
            caf_forward, xyv[0], xyv[1], source_scale, only_max)
        if new_xysv[3] == 0.0:
            return 0.0, 0.0, 0.0, 0.0
        keypoint_score = np.sqrt(new_xysv[3] * xyv[2])  # geometric mean
        if keypoint_score < self.keypoint_threshold:
            return 0.0, 0.0, 0.0, 0.0
        if keypoint_score < xyv[2] * self.keypoint_threshold_rel:
            return 0.0, 0.0, 0.0, 0.0
        xys_target = max(0.0, new_xysv[2])

        # reverse match
        if self.reverse_match and reverse_match:
            reverse_xyv = grow_connection_blend(
                caf_backward, new_xysv[0], new_xysv[1], xys_target, only_max)
            if reverse_xyv[2] == 0.0:
                return 0.0, 0.0, 0.0, 0.0
            if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > source_scale:
                return 0.0, 0.0, 0.0, 0.0

        return (new_xysv[0], new_xysv[1], new_xysv[2], keypoint_score)

    @staticmethod
    def p2p_value(source_xyv, caf_scored, source_s, target_xysv, caf_i, forward):
        # TODO move to Cython (see grow_connection_blend)
        caf_f, _ = caf_scored.directed(caf_i, forward)
        xy_scale_s = max(0.0, source_s)

        # source value
        caf_field = caf_center_s(caf_f, source_xyv[0], source_xyv[1],
                                 sigma=2.0 * xy_scale_s)
        if caf_field.shape[1] == 0:
            return 0.0

        # distances
        d_source = np.linalg.norm(
            ((source_xyv[0],), (source_xyv[1],)) - caf_field[1:3], axis=0)
        d_target = np.linalg.norm(
            ((target_xysv[0],), (target_xysv[1],)) - caf_field[5:7], axis=0)

        # combined value and source distance
        xy_scale_t = max(0.0, target_xysv[2])
        sigma_s = 0.5 * xy_scale_s
        sigma_t = 0.5 * xy_scale_t
        scores = (
            np.exp(-0.5 * d_source**2 / sigma_s**2)
            * np.exp(-0.5 * d_target**2 / sigma_t**2)
            * caf_field[0]
        )
        return np.sqrt(source_xyv[2] * max(scores))

    def _grow(self, ann, caf_scored, *, reverse_match=True):
        frontier = []
        in_frontier = set()

        def add_to_frontier(start_i):
            for end_i, (caf_i, _) in self.by_source[start_i].items():
                if ann.data[end_i, 2] > 0.0:
                    continue
                if (start_i, end_i) in in_frontier:
                    continue

                max_possible_score = np.sqrt(ann.data[start_i, 2])
                if self.confidence_scales is not None:
                    max_possible_score *= self.confidence_scales[caf_i]
                heapq.heappush(frontier, (-max_possible_score, None, start_i, end_i))
                in_frontier.add((start_i, end_i))
                ann.frontier_order.append((start_i, end_i))

        def frontier_get():
            while frontier:
                entry = heapq.heappop(frontier)
                if entry[1] is not None:
                    return entry

                _, __, start_i, end_i = entry
                if ann.data[end_i, 2] > 0.0:
                    continue

                new_xysv = self.connection_value(
                    ann, caf_scored, start_i, end_i, reverse_match=reverse_match)
                if new_xysv[3] == 0.0:
                    continue
                score = new_xysv[3]
                if self.greedy:
                    return (-score, new_xysv, start_i, end_i)
                if self.confidence_scales is not None:
                    caf_i, _ = self.by_source[start_i][end_i]
                    score = score * self.confidence_scales[caf_i]
                heapq.heappush(frontier, (-score, new_xysv, start_i, end_i))

        # seeding the frontier
        for joint_i in np.flatnonzero(ann.data[:, 2]): # array([0])
            add_to_frontier(joint_i)

        while True:
            # entry contain joint_souce and joint_target, what how it gets joint_target
            entry = frontier_get()
            if entry is None:
                break

            _, new_xysv, joint_source, joint_target = entry
            if ann.data[joint_target, 2] > 0.0:
                continue

            ann.data[joint_target, :2] = new_xysv[:2] #xy
            ann.data[joint_target, 2] = new_xysv[3]#v
            ann.joint_scales[joint_target] = new_xysv[2]# scale update
            ann.decoding_order.append(
                (joint_source, joint_target, np.copy(ann.data[joint_source]), np.copy(ann.data[joint_target])))
            add_to_frontier(joint_target)

    def _flood_fill(self, ann):
        frontier = []

        def add_to_frontier(start_i):
            for end_i, (caf_i, _) in self.by_source[start_i].items():
                if ann.data[end_i, 2] > 0.0:
                    continue
                start_xyv = ann.data[start_i].tolist()
                score = start_xyv[2]
                if self.confidence_scales is not None:
                    score = score * self.confidence_scales[caf_i]
                heapq.heappush(frontier, (-score, end_i, start_xyv, ann.joint_scales[start_i]))

        for start_i in np.flatnonzero(ann.data[:, 2]):
            add_to_frontier(start_i)

        while frontier:
            _, end_i, xyv, s = heapq.heappop(frontier)
            if ann.data[end_i, 2] > 0.0:
                continue
            ann.data[end_i, :2] = xyv[:2]
            ann.data[end_i, 2] = 0.00001
            ann.joint_scales[end_i] = s
            add_to_frontier(end_i)

    def complete_annotations(self, cifhr, fields, annotations):
        start = time.perf_counter()

        if self.force_complete_caf_th >= 0.0:
            caf_scored = (utils
                          .CafScored(cifhr.accumulated, score_th=self.force_complete_caf_th)
                          .fill(fields, self.caf_metas))
            for ann in annotations:
                unfilled_mask = ann.data[:, 2] == 0.0
                self._grow(ann, caf_scored, reverse_match=False)
                now_filled_mask = ann.data[:, 2] > 0.0
                updated = np.logical_and(unfilled_mask, now_filled_mask)
                ann.data[updated, 2] = np.minimum(0.001, ann.data[updated, 2])

        # some joints might still be unfilled
        for ann in annotations:
            self._flood_fill(ann)

        LOG.debug('complete annotations %.3fs', time.perf_counter() - start)
        return annotations