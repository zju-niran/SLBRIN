import os


def rename(cur_path, old_name, new_name):
    files = os.listdir(cur_path)
    for file in files:
        file = os.path.join(cur_path, file)
        if os.path.isdir(file):
            rename(file, old_name, new_name)
        else:
            if old_name in file:
                os.rename(file, file.replace(old_name, new_name))  # 重命名文件


if __name__ == '__main__':
    path = r'E:\model'
    rename(path, 'sbrin', 'slbrin')
