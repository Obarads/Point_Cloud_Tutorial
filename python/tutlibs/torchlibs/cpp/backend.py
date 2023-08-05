import os

from torch.utils.cpp_extension import load

_cpp_src_path = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../cu_cpp")
)
_backend = load(
    name="_backend",
    extra_cflags=["-O3", "-std=c++17"],
    sources=[
        os.path.join(_cpp_src_path, f)
        for f in [
            "knn/k_nearest_neighbors.cpp",
            "knn/k_nearest_neighbors.cu",
            "bindings.cpp",
        ]
    ],
)

__all__ = ["_backend"]
