import os

def del_file(path='/ziyuanqin/projects/smallobj/data/PLANE'):
    ls = os.listdir(path)
    for f in ls:
        f_path = os.path.join(path, f)
        if f[-3:] == 'txt':
            os.remove(f_path)

del_file()
