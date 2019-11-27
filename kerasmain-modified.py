from __future__ import print_function
from configs import train_path, test_path
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 40
num_classes = 2
epochs = 30
data_augmentation = True
save_dir = os.path.join(os.getcwd(), 'archive')
model_name = 'kerasfoodsmall-mRMS.h5'

# Optimizers
opt1 = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)
opt3 = keras.optimizers.Adagrad(learning_rate=0.01)
opt5 = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

# Added parameters
dense_layers = [1, 2]
layer_sizes = [32, 64]
#optimzers = [opt1, opt3, opt5]

model_path = os.path.join(save_dir, model_name)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(128, 128),
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed = 1)

test_generator = test_datagen.flow_from_directory(
        directory=test_path,
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='categorical')

for dense_layer in dense_layers:
    for layer_size in layer_sizes:

        namedir = "Optimizedmodel-RMS-optimizer-{}-nodes-{}-dense".format( layer_size, dense_layer)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir='foodsmallmodifiedruns1\{}'.format(namedir))
        model = Sequential()

        model.add(Conv2D(32, (3, 3), padding='same', input_shape=(128,128,3)))
        model.add(Activation('relu'))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        
        for _ in range(dense_layer):
            model.add(Dense(layer_size))
            model.add(Activation('relu'))
        
        model.add(Dropout(0.5))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        opt = opt1 # RMSprop optimizer
        #opt = opt3 # Adagrad optimizer
        #opt = opt5 # Adma optimizer

        if os.path.exists(model_path):
            print("LOADING OLD MODEL")
            model.load_weights(model_path)

        model.compile(loss= 'categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

        model.fit_generator(train_generator,
                            epochs=epochs,
                            steps_per_epoch=batch_size, # Specfiying batch size
                            validation_data=test_generator,
                            callbacks=[tensorboard_callback])


model.save(model_path)
