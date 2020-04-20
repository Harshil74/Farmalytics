from PIL import Image
import os, sys


path = "../path_of_folder/"
dirs = os.listdir(path)

def resize():
    count = 0
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            imResize = im.resize((224, 138), Image.ANTIALIAS)
            imResize.save('.../path_to_save/'+ str(count) + '.jpg', 'JPEG', quality=5000)
            count = count + 1
            print(count)


resize()
