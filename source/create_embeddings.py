import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
loaded_model = load_model('./models/Resnet_model_best_mean_absolute_error.h5')



test_generator = ImageDataGenerator()

batch_size=32

test_batches=test_generator.flow_from_directory('./test', 
                                                         target_size=(224,224),
                                                         batch_size=batch_size,
                                                  shuffle=False
                                                  
                                                 )




model_output = []
model_imgs = []
model_labels = []
for i in range(test_batches.samples//batch_size):
  imgs,labels = test_batches.next()
  output_i = loaded_model.predict_on_batch(imgs)
  #model_output.append(output_i[0].reshape(output_i.shape[0],output_i.shape[1]*output_i.shape[2]*output_i.shape[3] ))
  model_output.extend(output_i[0])  
  model_imgs.extend(imgs)
  model_labels.extend(labels)
  
model_output = np.array(model_output)
model_imgs = np.array(model_imgs)
model_labels = np.array(model_labels)

np.save('embedding.npy', model_output)



