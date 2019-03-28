import os
import shutil

os.mkdir('./dataset')
count = 1

for root, dirs, files in os.walk('./Dresses'):  # replace the . with your starting directory
    for file in files:
        if "front" in file:
            path_file = os.path.join(root,file)
            shutil.copy2(path_file,'./dataset') # change you destination dir
            flag = os.rename(os.path.join('./dataset', file), './dataset/%d.jpg' % count)
            count+=1
