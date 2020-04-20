import numpy as np
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img 
from os import listdir
from PIL import Image as PImage

# for Loading all images from given folder into list..
def loadImages(path):
    # return array of images
    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        loadedImages.append(img)

    return loadedImages

# datagenretor
datagen = ImageDataGenerator( 
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2
        ) 


# folder path from where you want to take images..
path = "../path_of_folder/"

# folder path where you want to save Augmented images
save_here = '../path_of_distenation_folder/'

# Number of time you want ot augment the image.
no = 10

# calling funcation to store images in an array
imgs = loadImages(path)

# augmentation
for img in imgs:
    x = img_to_array(img) 
    x = x.reshape((1, ) + x.shape)
    i = 1
    for batch in datagen.flow(x, batch_size = 1, 
                          save_to_dir =save_here,  
                          save_prefix = 'image' , save_format ='jpeg'): 
	    print(str(i)+ " " + "images augmented")
	    i += 1
	    # givent specific number defines how many times you want to augmente image.
	    if i > no: 
	        break