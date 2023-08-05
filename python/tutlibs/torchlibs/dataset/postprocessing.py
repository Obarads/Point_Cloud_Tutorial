import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


class BlockMerging:
    def __init__(self, gap=1e-3, max_range=1.0) -> None:
        self.gap = gap
        self.max_range = max_range

        volume_num = int(self.max_range / self.gap) + 1
        self.volume_num = volume_num
        self.volume = -1 * np.ones([volume_num, volume_num, volume_num]).astype(
            np.int32
        )
        self.volume_seg = -1 * np.ones(
            [volume_num, volume_num, volume_num]
        ).astype(np.int32)

    def assign(
        self, coords: np.ndarray, grouplabel: np.ndarray, sem_labels: np.ndarray
    ):
        groupseg: Dict[int, int] = {}
        num_clusters = np.max(grouplabel)

        for idx_cluster in range(num_clusters):
            tmp = grouplabel == idx_cluster
            if np.sum(tmp) != 0:  # add (for a cluster of zero element.)
                a = stats.mode(sem_labels[tmp])[0]
                estimated_seg = int(a)
                groupseg[idx_cluster] = estimated_seg

        gap = self.gap
        num_points = len(coords)

        overlapgroupcounts = np.zeros([100, 300])
        groupcounts = np.ones(100)
        x = (coords[:, 0] / gap).astype(np.int32)
        y = (coords[:, 1] / gap).astype(np.int32)
        z = (coords[:, 2] / gap).astype(np.int32)
        for i in range(num_points):
            xx = x[i]
            yy = y[i]
            zz = z[i]
            if grouplabel[i] != -1:
                if (
                    self.volume[xx, yy, zz] != -1
                    and self.volume_seg[xx, yy, zz] == groupseg[grouplabel[i]]
                ):
                    # overlapgroupcounts[grouplabel[i],volume[xx,yy,zz]] += 1
                    try:
                        overlapgroupcounts[
                            grouplabel[i], self.volume[xx, yy, zz]
                        ] += 1
                    except:
                        pass
            groupcounts[grouplabel[i]] += 1

        groupcate = np.argmax(overlapgroupcounts, axis=1)
        maxoverlapgroupcounts = np.max(overlapgroupcounts, axis=1)

        curr_max = np.max(self.volume)
        for i in range(groupcate.shape[0]):
            if maxoverlapgroupcounts[i] < 7 and groupcounts[i] > 30:
                curr_max += 1
                groupcate[i] = curr_max

        finalgrouplabel = -1 * np.ones(num_points)

        for i in range(num_points):
            if grouplabel[i] != -1 and self.volume[x[i], y[i], z[i]] == -1:
                self.volume[x[i], y[i], z[i]] = groupcate[grouplabel[i]]
                self.volume_seg[x[i], y[i], z[i]] = groupseg[grouplabel[i]]
                finalgrouplabel[i] = groupcate[grouplabel[i]]
        return finalgrouplabel
