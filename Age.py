import pandas as pd
import numpy as np
import os


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train['ID'][0]

for i in range(0,19906):
    source = './Train/'+train['ID'][i]
    destination = './train/'+train['Class'][i]+'/'+train['ID'][i]
    os.rename(source , destination)
    print(train['ID'][i] + ' -> ' + train['Class'][i])
    

## Now Doing CNN

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense,Dropout
from keras.preprocessing import image    
from keras.models import load_model    


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(100, (3, 3), input_shape = (64,64,3),padding='same', activation = 'relu'))
classifier.add(Dropout(0.2))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(0.5))
# Adding a second convolutional layer
classifier.add(Conv2D(60, (3, 3), activation = 'relu',padding='same'))
classifier.add(Dropout(0.2))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(0.2))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(MaxPooling2D(pool_size = (2, 2)))


classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(Dropout(0.2))

classifier.add(MaxPooling2D(pool_size = (2, 2)))
#classifier.add(Dropout(0.2))

# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units=500, activation='relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units=100, activation='relu'))
classifier.add(Dropout(0.2))

classifier.add(Dense(units=32, activation='relu'))
classifier.add(Dropout(0.2))


#classifier.add(Dense(units=10, activation='relu'))
#classifier.add(Dropout(0.2))


classifier.add(Dense(units=3, activation='softmax'))
#classifier.add(Dropout(0.2))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   rotation_range=45,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('train/',
                                                 target_size = (64,64),
                                                 batch_size = 64,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('test/',
                                            target_size = (64,64),
                                            batch_size = 64,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 592,
                         epochs = 40,
                         validation_data = test_set,
                         validation_steps = 75)
                         
classifier.save('model4.h5')  

classifier = load_model('model.h5') 

import imageio
import cv2
im = imageio.imread('./Test/2.jpg')
test_image = cv2.resize(im, (32, 32))    
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

list = []
for i in range(0,6636):
    im = imageio.imread('./Test/'+test['ID'][i])
    test_image = cv2.resize(im, (32, 32))    
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)
    if(result[0][0]==1):
        list.append('MIDDLE')
    elif(result[0][1]==1):
        list.append('OLD')
    elif(result[0][2]==1):
        list.append('YOUNG')
    else:
        list.append('MIDDLE')

df = pd.DataFrame()
df['ID'] = test['ID']

from collections import OrderedDict
sales = OrderedDict([ ('Class', list) ] )    
df1 = pd.DataFrame.from_dict(sales)
df['Class']=df1
df.to_csv('sub.csv')