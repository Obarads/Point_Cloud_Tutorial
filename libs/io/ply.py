from plyfile import PlyData, PlyElement

def write(output_file_path, point_data):
    ply = PlyData([PlyElement.describe(point_data, "vertex")], text=False)
    ply.write(output_file_path)

