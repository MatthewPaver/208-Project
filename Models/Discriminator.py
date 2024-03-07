import tensorflow as tf
from tensorflow.keras import layers

# please do not put <-ft.tensor after my declarations, as it defaults to requiring the tensor output of the pre-processors be the same size as the input functions, and create_discriminator
# does not return an ft.keras.Model object to the caller, it returns it to a tensorflow function, which in turn finds the calling object and overrides the return, so requiring the function
# to directly return a model object crashes the code.

# I have changed the preprocessing functions in the discriminator to image_preprocessor1 and label_preprocessor1, so that they do not overload the definitions for the smamge functions
# in generator.py, when they are both imported to train.py

IMAGE_DIMENSIONS = (28, 28, 1) #(H,W,C)

def build_discriminator():
    #input preprocessing for first stream
    con_label = layers.Input(shape=(1,))
    x = layers.Embedding(3, 50)(con_label)
    x = layers.Dense((128*128*3))(x)
    stream2_input = layers.Reshape((128, 128, 3))(x)
    # input preprocessing for second stream
    stream1_input = layers.Input(shape=(128,128,3))
    # concat label as a channel
    merge = layers.Concatenate()([stream1_input, stream2_input])
    
    x = layers.Conv2D(64,4) (merge)
    x = layers.BatchNormalization() (x)
    x = layers.ReLU() (x)

    x = layers.Conv2D(64,4) (x)
    x = layers.BatchNormalization() (x)
    x = layers.ReLU() (x)

    x = layers.Conv2D(64,4) (x)
    x = layers.BatchNormalization() (x)
    x = layers.ReLU() (x)

    x = layers.Flatten() (x)
    x = layers.Dropout(0.3) (x)
    x = layers.Dense(1,activation ='sigmoid') (x)

    model = tf.keras.Model([stream1_input, con_label], x)

    return model


