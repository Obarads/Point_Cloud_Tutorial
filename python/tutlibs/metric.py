import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


class SemanticMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.tp = np.zeros(num_classes)
        self.tn = np.zeros(num_classes)
        self.fp = np.zeros(num_classes)
        self.fn = np.zeros(num_classes)

    def update(self, pred: np.ndarray, gt: np.ndarray):
        """
        Args:
            pred: [N1, N2, ...]
            gt: [N1, N2, ...]
        """
        for c in range(self.num_classes):
            self.tp[c] += np.sum((pred == c) & (gt == c))
            self.tn[c] += np.sum((pred != c) & (gt != c))
            self.fp[c] += np.sum((pred == c) & (gt != c))
            self.fn[c] += np.sum((pred != c) & (gt == c))

    def recall(self):
        """
        Returns:
            recall: [C]
        """
        return self.tp / (self.tp + self.fn)

    def precision(self):
        """
        Returns:
            precision: [C]
        """
        return self.tp / (self.tp + self.fp)

    def specificity(self):
        """
        Returns:
            specificity: [C]
        """
        return self.tn / (self.tn + self.fp)

    def f1(self):
        """
        Returns:
            f1: [C]
        """
        recall = self.recall()
        precision = self.precision()
        return 2 * recall * precision / (recall + precision)

    def accuracy(self):
        """
        Returns:
            accuracy: [C]
        """
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def iou(self):
        """
        Returns:
            IoU: [C]
        """
        return self.tp / (self.tp + self.fp + self.fn)


class CoverageMetrics:
    def __init__(self, num_calsses):
        self.num_classes = num_calsses

        self.sum_coverage = np.zeros(self.num_classes, dtype=np.float)
        self.num_gt = np.zeros(self.num_classes, dtype=np.int)
        self.weighted_sum_coverage = np.zeros(self.num_classes, dtype=np.float)
        self.num_gt_points = np.zeros(self.num_classes, dtype=np.int)

    def update(
        self,
        pred_sem: np.ndarray,
        pred_ins: np.ndarray,
        gt_sem: np.ndarray,
        gt_ins: np.ndarray,
    ):
        # instance
        un = np.unique(pred_ins)
        pts_in_pred = [[] for _ in range(self.num_classes)]
        for ig, g in enumerate(un):  # each object in prediction
            if g == -1:
                continue
            tmp = pred_ins == g
            sem_seg_i = int(stats.mode(pred_sem[tmp])[0])
            pts_in_pred[sem_seg_i] += [tmp]

        un = np.unique(gt_ins)
        pts_in_gt = [[] for _ in range(self.num_classes)]
        for ig, g in enumerate(un):
            tmp = gt_ins == g
            sem_seg_i = int(stats.mode(gt_sem[tmp])[0])
            pts_in_gt[sem_seg_i] += [tmp]

        # instance mucov & mwcov
        for i_sem in range(self.num_classes):
            for ig, ins_gt in enumerate(pts_in_gt[i_sem]):
                ovmax = 0.0
                for ip, ins_pred in enumerate(pts_in_pred[i_sem]):
                    union = ins_pred | ins_gt
                    intersect = ins_pred & ins_gt
                    iou = float(np.sum(intersect)) / np.sum(union)

                    if iou > ovmax:
                        ovmax = iou

                num_ins_gt_point = np.sum(ins_gt)
                self.sum_coverage[i_sem] += ovmax
                self.weighted_sum_coverage[i_sem] += ovmax * num_ins_gt_point
                self.num_gt_points[i_sem] += num_ins_gt_point

            self.num_gt[i_sem] += len(pts_in_gt[i_sem])

    def mucov(self):
        return self.sum_coverage / self.num_gt

    def mwcov(self):
        return self.weighted_sum_coverage / self.num_gt_points
