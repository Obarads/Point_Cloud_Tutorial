import numpy as np
from typing import List


class Node:
    def __init__(
        self,
        coords: np.ndarray,
        limit_resolution: float,
        min_coords: np.ndarray = None,
        max_coords: np.ndarray = None,
    ) -> None:
        if min_coords is not None and max_coords is not None:
            min_coords, max_coords = self.get_max_min_coords(coords)

        self.min_coords = min_coords
        self.max_coords = max_coords
        self.limit_resolution = limit_resolution

        corners, resolution = self.get_corner_from_boundary(
            self.max_coords, self.min_coords
        )

        if np.all(resolution >= limit_resolution):
            self.nodes = None
            self.coords = coords
        else:
            nodes: List[Node] = []
            for x in range(2):
                for y in range(2):
                    for z in range(2):
                        _min_coords = [corners[x, 0], corners[y, 1], corners[z, 2]]
                        more_coords = np.all(
                            coords >= _min_coords,
                            axis=1,
                        )
                        _max_coords = [
                            corners[x + 1, 0],
                            corners[y + 1, 1],
                            corners[z + 1, 2],
                        ]
                        less_coords = np.all(coords < _max_coords, axis=1)
                        mask = more_coords & less_coords

                        node = None
                        if np.sum(mask) >= 1:
                            node = Node(
                                coords[mask],
                                limit_resolution,
                                min_coords=_min_coords,
                                max_coords=_max_coords,
                            )

                        nodes.append(node)

            self.nodes = nodes
            self.coords = None

    @staticmethod
    def get_corner_from_boundary(
        max_coords: np.ndarray, min_coords: np.ndarray
    ) -> List[np.ndarray, np.ndarray]:
        center = (max_coords + min_coords) / 2
        corners = np.stack([min_coords, center, max_coords], axis=-1)
        resolution = center - min_coords
        return corners, resolution

    @staticmethod
    def get_max_min_coords(
        coords: np.ndarray,
    ) -> List[np.ndarray, np.ndarray]:
        max_coords = np.max(coords, axis=0)
        min_coords = np.min(coords, axis=0)
        return min_coords, max_coords

    def knn(self, coords:np.ndarray):
        pass

class Octree:
    def __init__(self, coords: np.ndarray, resolution: float) -> None:
        self.coords = coords
        self.resolution = resolution
        self.nodes = Node(self.coords, resolution)

    def knn(self, coords:np.ndarray) -> List[np.ndarray, np.ndarray]:
        return
