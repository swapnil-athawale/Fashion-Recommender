import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

loaded_model = load_model('./models/VGG_model_best.h5')
print('ok')


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

img = Image.open('./test/Blazer/Asymmetrical_Open-Front_Blazer/img_00000106.jpg')
plt.imshow(img)
processed_image = preprocess_image(img, target_size=(200, 200))

prediction = loaded_model.predict(processed_image)
#print(prediction)


dict_classes = {'Button-Down': 4, 'Bomber': 3, 'Blazer': 1, 'Coat': 9, 'Tee': 42, 'Jeans': 20, 'Sweatpants': 39, 'Peacoat': 31, 'Shorts': 36, 'Trunks': 44, 'Kaftan': 26, 'Sweater': 38, 'Flannel': 14, 'Leggings': 28, 'Onesie': 29, 'Jacket': 19, 'Jeggings': 21, 'Jodhpurs': 23, 'Henley': 17, 'Sarong': 35, 'Tank': 41, 'Anorak': 0, 'Hoodie': 18, 'Parka': 30, 'Chinos': 8, 'Blouse': 2, 'Dress': 13, 'Jumpsuit': 25, 'Gauchos': 15, 'Halter': 16, 'Coverup': 10, 'Turtleneck': 45, 'Joggers': 24, 'Caftan': 5, 'Capris': 6, 'Jersey': 22, 'Skirt': 37, 'Cutoffs': 12, 'Cardigan': 7, 'Kimono': 27, 'Sweatshorts': 40, 'Top': 43, 'Culottes': 11, 'Poncho': 32, 'Robe': 33, 'Romper': 34}





import shutil
import os
import re
#import cv2
import pickle


from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np




## reading fro pickle files
pickle_in = open("dict_train.pickle","rb")
dict_train = pickle.load(pickle_in)

pickle_in = open("dict_test.pickle","rb")
dict_test = pickle.load(pickle_in)

pickle_in = open("dict_val.pickle","rb")
dict_val = pickle.load(pickle_in)


from keras.models import Model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K

'''
train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()
'''



###for custom
class DirectoryIteratorWithBoundingBoxes(DirectoryIterator):
    def __init__(self, directory, image_data_generator, bounding_boxes: dict = None, target_size=(256, 256),
                 color_mode: str = 'rgb', classes=None, class_mode: str = 'categorical', batch_size: int = 32,
                 shuffle: bool = True, seed=None, data_format=None, save_to_dir=None,
                 save_prefix: str = '', save_format: str = 'jpeg', follow_links: bool = False):
        super().__init__(directory, image_data_generator, target_size, color_mode, classes, class_mode, batch_size,
                         shuffle, seed, data_format, save_to_dir, save_prefix, save_format, follow_links)
        self.bounding_boxes = bounding_boxes

    def __next__(self):
        """
        # Returns
            The next batch.
        """
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        batch_x = np.zeros((len(index_array),) + self.image_shape, dtype=K.floatx())
        locations = np.zeros((len(batch_x),) + (4,), dtype=K.floatx())

        grayscale = self.color_mode == 'grayscale'
        # build batch of image data
        for i, j in enumerate(index_array):
            fname = self.filenames[j]
            img = image.load_img(os.path.join(self.directory, fname),
                                 grayscale=grayscale,
                                 target_size=self.target_size)
            x = image.img_to_array(img, data_format=self.data_format)
            x = self.image_data_generator.random_transform(x)
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x

            if self.bounding_boxes is not None:
                bounding_box = self.bounding_boxes[fname]
                locations[i] = np.asarray(
                    [bounding_box['origin']['x'], bounding_box['origin']['y'], bounding_box['width'],
                     bounding_box['height']],
                    dtype=K.floatx())
        # optionally save augmented images to disk for debugging purposes
        # build batch of labels
        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]
        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype(K.floatx())
        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), 46), dtype=K.floatx())
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, label] = 1.
        else:
            return batch_x

        if self.bounding_boxes is not None:
            return batch_x, [batch_y, locations]
        else:
            return batch_x, batch_y


'''
## same size iamges
train_iterator = DirectoryIteratorWithBoundingBoxes("./train", train_datagen, bounding_boxes=dict_train, target_size=(200, 200), batch_size=32)

print (train_iterator.class_indices)
print('done')
#print('ooo',len(train_iterator))

test_iterator = DirectoryIteratorWithBoundingBoxes("./val", test_datagen, bounding_boxes=dict_val,target_size=(200, 200),  batch_size=32)
'''




def custom_generator(iterator):
    while True:
        batch_x, batch_y = next(iterator)
        yield (batch_x, batch_y)



test_datagen = ImageDataGenerator()
test_iterator = DirectoryIteratorWithBoundingBoxes("./test", test_datagen, bounding_boxes=dict_test, target_size=(200, 200))

scores = loaded_model.evaluate_generator(custom_generator(test_iterator), steps=500)

#print(('Multi target loss: ' + str(scores[0])))
#print(('Image loss: ' + str(scores[1])))
#print(('Bounding boxes loss: ' + str(scores[2])))
print(('Image accuracy: ' + str(scores[3])))
print(('Top-5 image accuracy: ' + str(scores[4])))
print(('Bounding boxes error: ' + str(scores[5])))










































