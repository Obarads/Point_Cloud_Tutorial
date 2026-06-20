import heapq
import numpy as np
from typing import List, Tuple


class OctreeNode:
    """A node (cube cell) of an octree.

    Attributes:
        center: center coordinate of the cube, (3,)
        half_size: half of the cube edge length (distance from center to a face)
        depth: depth of this node (root is 0)
        children: list of 8 child OctreeNode (or None for an empty octant).
            This is None when the node is a leaf.
        point_indices: indices of points contained in this node.
            This is None when the node is an internal node.
    """

    def __init__(
        self, center: np.ndarray, half_size: float, depth: int
    ) -> None:
        self.center = center
        self.half_size = half_size
        self.depth = depth
        self.children: List["OctreeNode"] = None
        self.point_indices: np.ndarray = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None


class Octree:
    """Octree for a 3D point cloud.

    An octree recursively subdivides a cubic region of space into 8 octants
    (children). The subdivision stops when a node reaches ``max_depth`` or when
    the number of points in a node is equal to or less than ``min_points``.
    The resulting tree can be used to accelerate nearest neighbor search by
    pruning cells that cannot contain neighbors.
    """

    def __init__(
        self,
        coords: np.ndarray,
        max_depth: int = 10,
        min_points: int = 1,
    ) -> None:
        """Build an octree from a point cloud.

        Args:
            coords: coordinates of points, (N, 3)
            max_depth: maximum depth of the tree
            min_points: a node is a leaf if it has this number of points or less
        """
        self.coords = np.asarray(coords, dtype=np.float64)
        self.max_depth = max_depth
        self.min_points = min_points

        # compute the root cube so that it contains all points.
        min_coords = np.min(self.coords, axis=0)
        max_coords = np.max(self.coords, axis=0)
        center = (min_coords + max_coords) / 2
        # use the largest side as the edge length so the root is a cube.
        # add a small margin so that points on the boundary are included.
        half_size = np.max(max_coords - min_coords) / 2 + 1e-6

        indices = np.arange(len(self.coords))
        self.root = self._build(indices, center, half_size, 0)

    def _build(
        self,
        indices: np.ndarray,
        center: np.ndarray,
        half_size: float,
        depth: int,
    ) -> OctreeNode:
        """Recursively build octree nodes."""
        node = OctreeNode(center, half_size, depth)

        # stop subdivision and make a leaf.
        if depth >= self.max_depth or len(indices) <= self.min_points:
            node.point_indices = indices
            return node

        # decide the octant of each point with a 3 bit code.
        # bit 0: x, bit 1: y, bit 2: z (1 if larger than center, 0 otherwise)
        points = self.coords[indices]
        codes = (
            (points[:, 0] > center[0]).astype(np.int64)
            + (points[:, 1] > center[1]).astype(np.int64) * 2
            + (points[:, 2] > center[2]).astype(np.int64) * 4
        )

        child_half_size = half_size / 2
        children: List[OctreeNode] = []
        for code in range(8):
            mask = codes == code
            if not np.any(mask):
                # no point in this octant.
                children.append(None)
                continue
            # offset of the child center from the parent center.
            offset = (
                np.array(
                    [
                        1 if code & 1 else -1,
                        1 if code & 2 else -1,
                        1 if code & 4 else -1,
                    ]
                )
                * child_half_size
            )
            child = self._build(
                indices[mask], center + offset, child_half_size, depth + 1
            )
            children.append(child)

        node.children = children
        return node

    @staticmethod
    def _min_square_distance_to_cube(
        query: np.ndarray, node: OctreeNode
    ) -> float:
        """Square distance from a query to the closest point of a node cube.

        This returns 0 when the query is inside the cube.
        """
        diff = np.abs(query - node.center) - node.half_size
        diff = np.maximum(diff, 0)
        return float(np.sum(diff ** 2))

    def radius_search(
        self, query: np.ndarray, r: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find neighbors within a radius from a query.

        Args:
            query: a query coordinate, (3,)
            r: radius

        Returns:
            idxs: indices of neighbors sorted by distance
            dists: distance to each neighbor
        """
        query = np.asarray(query, dtype=np.float64)
        results: List[Tuple[float, int]] = []
        self._radius_search(self.root, query, r, results)
        results.sort(key=lambda x: x[0])
        idxs = np.array([i for _, i in results], dtype=np.int64)
        dists = np.array([d for d, _ in results], dtype=np.float64)
        return idxs, dists

    def _radius_search(
        self,
        node: OctreeNode,
        query: np.ndarray,
        r: float,
        results: List[Tuple[float, int]],
    ) -> None:
        if node is None:
            return
        # prune: skip a cube that does not overlap with the query sphere.
        if self._min_square_distance_to_cube(query, node) > r ** 2:
            return
        if node.is_leaf:
            points = self.coords[node.point_indices]
            dists = np.sqrt(np.sum((points - query) ** 2, axis=1))
            for index, dist in zip(node.point_indices, dists):
                if dist <= r:
                    results.append((float(dist), int(index)))
        else:
            for child in node.children:
                self._radius_search(child, query, r, results)

    def knn_search(
        self, query: np.ndarray, k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors of a query.

        Args:
            query: a query coordinate, (3,)
            k: number of nearest neighbors

        Returns:
            idxs: indices of k nearest neighbors sorted by distance
            dists: distance to each neighbor
        """
        query = np.asarray(query, dtype=np.float64)
        # max-heap of (-distance, index) so heap[0] keeps the farthest neighbor.
        heap: List[Tuple[float, int]] = []
        self._knn_search(self.root, query, k, heap)
        results = sorted(((-d, i) for d, i in heap), key=lambda x: x[0])
        idxs = np.array([i for _, i in results], dtype=np.int64)
        dists = np.array([d for d, _ in results], dtype=np.float64)
        return idxs, dists

    def _knn_search(
        self,
        node: OctreeNode,
        query: np.ndarray,
        k: int,
        heap: List[Tuple[float, int]],
    ) -> None:
        if node is None:
            return
        # prune: skip a cube whose closest point is farther than the current
        # k-th neighbor.
        min_square_dist = self._min_square_distance_to_cube(query, node)
        if len(heap) == k and min_square_dist > (-heap[0][0]) ** 2:
            return
        if node.is_leaf:
            points = self.coords[node.point_indices]
            dists = np.sqrt(np.sum((points - query) ** 2, axis=1))
            for index, dist in zip(node.point_indices, dists):
                if len(heap) < k:
                    heapq.heappush(heap, (-float(dist), int(index)))
                elif dist < -heap[0][0]:
                    heapq.heapreplace(heap, (-float(dist), int(index)))
        else:
            # visit children nearest first for more effective pruning.
            children = [c for c in node.children if c is not None]
            children.sort(
                key=lambda c: self._min_square_distance_to_cube(query, c)
            )
            for child in children:
                self._knn_search(child, query, k, heap)

    def leaf_boundaries(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the cube boundary of every leaf node (for visualization).

        Returns:
            min_corners: minimum corner of each leaf cube, (L, 3)
            max_corners: maximum corner of each leaf cube, (L, 3)
        """
        min_corners: List[np.ndarray] = []
        max_corners: List[np.ndarray] = []

        def _traverse(node: OctreeNode):
            if node is None:
                return
            if node.is_leaf:
                min_corners.append(node.center - node.half_size)
                max_corners.append(node.center + node.half_size)
            else:
                for child in node.children:
                    _traverse(child)

        _traverse(self.root)
        return np.array(min_corners), np.array(max_corners)


def boundaries_to_lines(
    min_corners: np.ndarray, max_corners: np.ndarray
) -> np.ndarray:
    """Convert cube boundaries to the 12 edges of each cube.

    Args:
        min_corners: minimum corner of each cube, (L, 3)
        max_corners: maximum corner of each cube, (L, 3)

    Returns:
        lines: start and end points of edges, (L * 12, 2, 3)
    """
    # 8 corners of a cube as offsets (0 or 1) along each axis.
    corner_bits = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
        ]
    )
    # 12 edges as pairs of corner indices.
    edges = np.array(
        [
            [0, 1], [0, 2], [1, 3], [2, 3],  # bottom face
            [4, 5], [4, 6], [5, 7], [6, 7],  # top face
            [0, 4], [1, 5], [2, 6], [3, 7],  # vertical edges
        ]
    )

    lines = []
    for min_corner, max_corner in zip(min_corners, max_corners):
        # (8, 3) corners of this cube.
        corners = np.where(corner_bits == 1, max_corner, min_corner)
        lines.append(corners[edges])  # (12, 2, 3)
    return np.concatenate(lines, axis=0)
