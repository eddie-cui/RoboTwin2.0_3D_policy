import h5py

def print_hdf5_structure(name, obj):
    """回调函数，用于打印HDF5文件结构"""
    indent = name.count('/')
    print('  ' * indent + name)

def explore_hdf5_file(file_path):
    """遍历并打印HDF5文件结构"""
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"打开文件: {file_path}")
            print("HDF5 文件结构如下:")
            f.visititems(print_hdf5_structure)
    except Exception as e:
        print(f"打开文件时出错: {e}")

# 示例使用方法：
if __name__ == "__main__":
    # 替换成你的实际文件路径
    hdf5_file_path = '/home/sealab/RoboTwin2.0/RoboTwin/data/beat_block_hammer/demo_clean/data/episode0.hdf5'
    explore_hdf5_file(hdf5_file_path)
