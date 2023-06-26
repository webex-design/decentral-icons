import os
current_file_dir = os.path.dirname(os.path.realpath(__file__))
path_root = os.path.abspath(os.path.join(current_file_dir, '../../'))
path_public = os.path.join(path_root, 'src/public')

config =  {
    "path_root": path_root,
    "path_public": path_public
}
