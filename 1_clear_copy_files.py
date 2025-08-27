import os
import shutil

root_path = r'C:/Users/yukun/Desktop/Test'

# 清除/processed/文件夹内的所有文件
folder = root_path + '/Data/BGL/processed'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

folder = root_path + '/Data/BGL/Raw'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))

# 将所有文件从一个目录复制到另一个目录
src_dir = root_path + '/Data/BGL/Graph/Raw/'
dest_dir = root_path + '/Data/BGL/Raw/'
my_files = os.listdir(src_dir)
for file_name in my_files:
    print(file_name)
    print(type(dest_dir))
    src_file_name = src_dir + file_name
    dest_file_name = dest_dir + file_name
    shutil.copy(src_file_name, dest_file_name)
