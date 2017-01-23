
# coding: utf-8

# ## Jumpstart to Keras

# The package **keras** is an abstraction layer on top of a backend. It supports Theano or Tensorflow as the backend.
# 
# In the following example, we will use Theano as the backend.

# In[16]:

## keras - test the installation
import keras

## Use Theano as backend.
## update the backend entry to "theano" in the ~/.keras/keras.json file.
print(keras.__file__)


# In[17]:

## Keras provides an abstraction on top of backend - Theano or Tensorflow.
##
## Using Python 2.7.12, Theano 0.8.2, Keras 1.2.0 
## import the needed modules
import numpy as np
import pandas as pd

import cv2 #open cv


import matplotlib.pyplot as plt  #for ploting

#to connect notebook to GUI loop
get_ipython().magic(u'matplotlib inline')

# models - sequential provides a linear stack of layers.
from keras.models import Sequential

#get the "core" layers from keras
## Dense      : is a fully connected NN layer
## Dropout    : to prevent overfitting, drop out some of the connections from input to output
## Activation : applies activation function to the output layer
## Flatten    : flatten the inputs?
from keras.layers import Dense, Dropout, Activation, Flatten

#get the CNN layers
#from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils


# In[18]:

#print the configuration
print np.__version__
print pd.__version__
print keras.__version__

np.random.seed(234)


# ## Read the images

# In[19]:

## just pick few images for both cat & dog from train, as running it local machine

#pick three random numbers from 0 to 12499.
img_index_arr = np.random.randint(0, 12999, 3)
img_index_arr

TRAIN_DIR = "/Users/admin/myData/DataScience/CatsVsDogs/datasets/train/"
train_images = []

for idx in range(len(img_index_arr)):
    #print img_index_arr[idx]
    cat_image_name = "cat." + str(img_index_arr[idx]) + ".jpg"
    dog_image_name = "dog." + str(img_index_arr[idx]) + ".jpg"
    #print cat_image_name, dog_image_name
    train_images.append(TRAIN_DIR+cat_image_name)
    train_images.append(TRAIN_DIR+dog_image_name)
    
train_images



# In[20]:

TEST_DIR = "/Users/admin/myData/DataScience/CatsVsDogs/datasets/test/"
test_images= []

print("test")
#test_images = [TEST_DIR+i for i in os.listdir(TEST_DIR)]
# load only few test images.
for i in range(5):
   #print i
   test_image_name = str(i+1) + ".jpg"
   test_images.append(TEST_DIR+test_image_name)

test_images


# In[21]:

# input : color image of size 499x375

#resize to 64x64, 

ROWS = 64
COLUMNS = 64
CHANNELS = 3 #RGB

# read image cv and resize input image
def read_n_resize(image_location):
    img = cv2.imread(image_location, cv2.IMREAD_COLOR)
    return cv2.resize(img, (ROWS, COLUMNS), interpolation=cv2.INTER_CUBIC)


# function to read image
def read_images(image_list) :
    count = len(image_list)
    # create N dimensional array
    image_arr = np.ndarray(shape = (count, CHANNELS, ROWS, COLUMNS), dtype= np.uint8)
    
    for i, image_path in enumerate(image_list):
       print(i, image_path)
       image = read_n_resize(image_path)
       image_arr[i] = image.T
    
    return image_arr
    
train = read_images(train_images)
print "train : {}" .format(train.shape)


# In[22]:

test = read_images(test_images)
print "test : {}" .format(test.shape)


# ### Open and examine the images

# In[23]:

# display the images
for i in range(len(train_images)):
    print i
    plt.figure(figsize = (10,5))
    plt.imshow(read_n_resize(train_images[i]))
    plt.show()
    


# ### Generate the labels
# It's is a classification problem. Tag 1 for dog and 0 for cat.

# In[24]:

labels = []

for i, image_path in enumerate(train_images):
    print(i, image_path)
    if 'dog' in image_path:
       labels.append(1)
    else:
       labels.append(0)

labels


# ###  Build Model based on VGG16 Net
# 
# [Refer to Keras pre trained VGG16 model](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3)
# 
# [CNN @ Stanford](http://cs231n.github.io/convolutional-networks/)
# 
# [Rectified Linear Unit is preferred over sigmoid/tanh](http://cs231n.github.io/neural-networks-1/)

# In[25]:

# build the model

#stack up the layers usind .add()
    
# apply 3x3 convolution with 32 output filters on 64x64 input image
#
# this is the first convolution layer, so specify the input_shape()
# input is 3 layers (RGB) of 64x64, so input_shape(3,64,64)
#
# border_mode:same/valid/full ?
# activation : instead of linear, use rectified linear unit.
#
# MaxPooling is one way to reduce the number of parameters.
# in this case, run 2x2 filter and pick the max number amoung them.
#
# Flatten:
#
# Dropout: will reduce the overfit. It is a regularizing parameter.
#
#
def build_cat_dog_model():
    model = Sequential()

    # apply 3x3 convolution with 32 output filters on 64x64 input image
    model.add(Convolution2D(32, 3, 3, border_mode='same', input_shape=(3,64,64), activation='relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
#     model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss="binary_crossentropy", optimizer=RMSprop(lr=0.0001), metrics=['accuracy'])
    return model


model = build_cat_dog_model()


# 
# ## Use Custom callback to track the model

# In[26]:


# extend the class from keras callback
## Callback for loss logging per epoch
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')        
        


# In[28]:


nb_epoch = 10
batch_size = 16

# fit the model 
# register the callback
history = LossHistory()
model.fit(train, labels, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])

# do the predictions
predictions = model.predict(test, verbose=0)

print history.losses
print predictions


# ## Examine the results

# In[15]:

loss = history.losses
val_loss = history.val_losses

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('VGG-16 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::2])
plt.legend()
plt.show()


# ### Let's look at the results

# In[29]:

for i in range(0,5):
    if predictions[i, 0] >= 0.5: 
        print('I am {:.2%} sure this is a Dog'.format(predictions[i][0]))
    else: 
        print('I am {:.2%} sure this is a Cat'.format(1-predictions[i][0]))
        
    plt.imshow(test[i].T)
    plt.show()


# ### Generate submission file

# In[56]:

# generate the submission file

file = open("./submission.csv", "w")
file_prob = open("./submission_prob.csv", "w")

file.write("id, label\n")
file_prob.write("id, label\n")

for i in range(0,5):
    pred = 0
    if predictions[i, 0] >= 0.5:
         pred = 1 #dog
    file.write('%d,' % (i+1) );
    file.write('%d \n' % pred);
    
    #generate the probability file
    file_prob.write('%d,' % (i+1) )
    file_prob.write('%1.1f \n' % predictions[i, 0])
    
file.close()
file_prob.close()
print("Done!")


# In[49]:

predictions


# In[50]:

predictions[2]


# In[ ]:



