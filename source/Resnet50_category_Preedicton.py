# -*- coding: utf-8 -*-

import shutil
import os
import re
#import cv2
import pickle

# will use them for creating custom directory iterator
import numpy as np
from six.moves import range
#from __future__ import print

# regular expression for splitting by whitespace
#splitter = re.compile("\s+")
#base_path = '/home/b15cs038/fashion_btp/fashion_attribute/' # Set this path to your folder
#img_path = './img/'

from itertools import compress


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

## reading fro pickle files
pickle_in = open("dict_train.pickle","rb")
dict_train = pickle.load(pickle_in)

pickle_in = open("dict_test.pickle","rb")
dict_test = pickle.load(pickle_in)

pickle_in = open("dict_val.pickle","rb")
dict_val = pickle.load(pickle_in)



from keras.models import Model
#from keras.models import load_model
from keras.layers import Dense
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.preprocessing.image import DirectoryIterator, ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import backend as K


model_resnet = ResNet50(weights='imagenet', include_top=False, pooling='avg')

for layer in model_resnet.layers[:-12]:
    # 6 - 12 - 18 have been tried. 12 is the best.
    layer.trainable = False

print('ok1')

#Now, let’s build the category classification branch in the model.

x = model_resnet.output
x = Dense(512, activation='elu', kernel_regularizer=l2(0.001))(x) # THIS IS ELU, NOT RELU
y = Dense(46, activation='softmax', name='img')(x)


#Then, we will build the bounding box detection branch in the model.
print('ok2')
x_bbox = model_resnet.output
x_bbox = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
x_bbox = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x_bbox)
bbox = Dense(4, kernel_initializer='normal', name='bbox')(x_bbox)

#final omodel
final_model = Model(inputs=model_resnet.input,
                    outputs=[y, bbox])

opt = SGD(lr=0.0001, momentum=0.9, nesterov=True)



#loaded_model = load_model('./models/Resnet_model.h5')

final_model.compile(optimizer=opt,
                    loss={'img': 'categorical_crossentropy',
                          'bbox': 'mean_absolute_error'},
                    metrics={'img': ['accuracy', 'top_k_categorical_accuracy'], # default: top-5
                             'bbox': ['mae']})


###################Loading the data#################
#If you try to load all images with at least 100x100 size to your less than 64 GB memory,
#it will be out of bounds for memory.
train_datagen = ImageDataGenerator(rotation_range=30.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator()



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



## same size iamges
train_iterator = DirectoryIteratorWithBoundingBoxes("./train", train_datagen, bounding_boxes=dict_train, target_size=(200, 200), batch_size=32)

#print('ooo',len(train_iterator))

test_iterator = DirectoryIteratorWithBoundingBoxes("./val", test_datagen, bounding_boxes=dict_val,target_size=(200, 200),  batch_size=32)




####It is the time to add some helpful features to our model.
#First, we will define a learning rate reducer in order to get rid of the plateaus in the loss function.
lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               patience=12,
                               factor=0.5,
                               verbose=1)

tensorboard = TensorBoard(log_dir='./logs')

early_stopper = EarlyStopping(monitor='val_loss',
                              patience=30,
                              verbose=1)


checkpoint = ModelCheckpoint('./models/Resnet_model_best_mean_absolute_error.h5', verbose=1, monitor='val_bbox_mean_absolute_error',save_best_only=True, mode='min') 


def custom_generator(iterator):
    while True:
        batch_x, batch_y = next(iterator)
        yield (batch_x, batch_y)


history = final_model.fit_generator(custom_generator(train_iterator),
                          steps_per_epoch=2000,
                          epochs=100, validation_data=custom_generator(test_iterator),
                          validation_steps=200,
                          verbose=2,
                          shuffle=True,
                          callbacks=[lr_reducer, checkpoint, early_stopper, tensorboard],
                          workers=12,
                         use_multiprocessing=True
			)


with open('/trainHistoryDict', 'wb') as file_pi:
	pickle.dump(history.history, file_pi)



##testing
#test_datagen = ImageDataGenerator()
#test_iterator = DirectoryIteratorWithBoundingBoxes("./test", test_datagen, bounding_boxes=dict_test, target_size=(200, 200))

scores = final_model.evaluate_generator(custom_generator(test_iterator), steps=2000)

#print(('Multi target loss: ' + str(scores[0])))
#print(('Image loss: ' + str(scores[1])))
#print(('Bounding boxes loss: ' + str(scores[2])))
print(('Image accuracy: ' + str(scores[3])))
print(('Top-5 image accuracy: ' + str(scores[4])))
print(('Bounding boxes error: ' + str(scores[5])))






















