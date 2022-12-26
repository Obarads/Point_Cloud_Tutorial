import numpy as np

def convert_rae_to_xyz(sph_coords):
    """

    Args:
        spherical_coords: 
    """
    ranges = sph_coords[:, 0]
    azimuthes = sph_coords[:, 1]
    elevations = sph_coords[:, 2]

    x = ranges * np.cos(elevations) * np.cos(azimuthes)
    y = ranges * np.cos(elevations) * np.sin(azimuthes)
    z = ranges * np.sin(elevations)

    cart_coords = np.stack([x, y, z], axis=-1)
    return cart_coords

