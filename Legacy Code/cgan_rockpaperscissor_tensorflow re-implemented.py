import tensorflow as tf
from tensorflow.keras import layers
from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow_datasets as tfds
from tensorflow.keras import backend as K
from matplotlib import gridspec

ds = tfds.load('RockPaperScissors', split='train', as_supervised=True, shuffle_files=True)

ds = ds.shuffle(1000).batch(128)

# Create dictionary of target classes
label_dict = {
 0: 'Rock',
 1: 'Paper',
 2: 'Scissors'
}

plt.figure(figsize=(10, 10))
for image, label in ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        lab = np.array(label[i])
        plt.text(0.5, -0.1, s = label_dict[int(lab)], horizontalalignment='center',
     verticalalignment='center', transform = ax.transAxes, fontsize=20)
        plt.imshow(image[i])
        plt.axis("off")
    plt.show

@tf.function
def normalization(tensor):
    tensor = tf.image.resize(
    tensor, (128,128))
    tensor = tf.subtract(tf.divide(tensor, 127.5), 1)
    return tensor

for img, label in ds.take(1):
    img = tf.cast(img, tf.float32)
    imgs = normalization(img)
    print(imgs.shape)

BATCH_SIZE=128
latent_dim = 100

def normalized_tanh(x):
    return (tf.tanh(x) + 1) / 2

def build_generator():
    #pre-processing for first input stream
    inputs1 = layers.Input(shape=(latent_dim,))
    x = layers.Dense(512*4*4)(inputs1)
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    input_stream1 = layers.Reshape((4,4,512))(x)
    #pre_processing for second input stream
    inputs2 = layers.Input(shape=(1,))
    x = layers.Embedding(3,50)(inputs2)
    x = layers.Dense((4*4)) (x)
    input_stream2 = layers.Reshape((4,4,1))(x)
    #input_stream1 = image_preprocessing()
    x = layers.Concatenate() ([input_stream1,input_stream2])
    
    #Activation function will be Tanh and ELU (Can change to Leaky ReLU/ReLU after further experiments)
    #Strides are doubling the input size
    #Batch normalization to ensure smooth training
    #Tanh is typically better and outputs to [-1, 1] but using a normlized tanh function it outputs [0,1] instead
    
    x = layers.Conv2DTranspose(64*16, kernel_size=4, strides=2, padding='same')(x)
    #8 x 8
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
   
    
    x = layers.Conv2DTranspose(64*8, kernel_size=4, strides=2, padding='same')(x)
    #16 x 16
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)

    x = layers.Conv2DTranspose(64*4, kernel_size=4, strides=2, padding='same')(x)
    #32 x 32
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)

    x = layers.Conv2DTranspose(64*2, kernel_size=4, strides=2, padding='same')(x)
    #64 x 64
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)

    x = layers.Conv2DTranspose(64*1, kernel_size=4, strides=2, padding='same')(x)
    #128 x 128
    x = layers.BatchNormalization()(x)
    x = layers.ELU()(x)
    
    
    outputs = layers.Conv2D(3, kernel_size=4, padding='same', activation=normalized_tanh)(x)
    
    
    model = tf.keras.Model(inputs=[inputs1,inputs2], outputs=outputs, name='generator')
    return model

generator = build_generator()

print(generator.summary())

def create_discriminator():
    #input preprocessing for first stream
    con_label = layers.Input(shape=(1,))
    x = layers.Embedding(3, 50)(con_label)
    x = layers.Dense((128*128*3))(x)
    stream2_input = layers.Reshape((128, 128, 3))(x)
    # input preprocessing for second stream
    stream1_input = layers.Input(shape=in_shape)
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


discriminator = create_discriminator()

print(discriminator.summary())

embeddings = generator.layers[3]

weights = embeddings.get_weights()[0]

print(weights.shape)

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()



num_examples_to_generate = 25
seed = tf.random.normal([num_examples_to_generate, latent_dim])

seed.dtype

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    labels = label_gen(n_classes=3)
    predictions = model([test_input, labels], training=False)
    print(predictions.shape)
    fig = plt.figure(figsize=(8,8))

    print("Generated Images are Conditioned on Label:", label_dict[np.array(labels)[0]])
    for i in range(predictions.shape[0]):
        plt.subplot(5, 5, i+1)
        pred = (predictions[i, :, :, :] + 1 ) * 127.5
        pred = np.array(pred)  
        plt.imshow(pred.astype(np.uint8))
        plt.axis('off')

    plt.savefig('rock-paper-scissors/images/image_at_epoch_{:d}.png'.format(epoch))
    plt.show()

learning_rate1 = 0.1
generator_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate1, beta_1 = 0.5, beta_2 = 0.999 )

def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        i = 0
        D_loss_list, G_loss_list = [], []
        for image_batch,target in dataset:
            print(target.shape)
            i += 1
            img = tf.cast(image_batch, tf.float32)
            imgs = normalization(img)
            # new
            noise = tf.random.normal([target.shape[0], latent_dim])
            generated_images = generator([noise,target], training=True)
            #random_image_noise = np.random.rand(128,128,128,3) #I am assuming a batch size of 128
            #random_image_noise = tf.convert_to_tensor(random_image_noise, dtype=tf.float32)
            random_image_noise = tf.random.normal([target.shape[0], latent_dim])
            fake_images = generator([random_image_noise,target],training=True)
            discriminator.compile('Adam','binary_crossentropy')
            K.set_value(discriminator.optimizer.learning_rate, learning_rate1)
            discriminator.fit((imgs,target),np.ones(128,),128,1)
            discriminator.fit((fake_images,target),np.zeros(128,),128,1)
            # the generator has a non-standard loss function (using the loss of another model to update)
            # therefore we cannot use the .fit() function and must manually instantiate the optimizer and loss function
            with tf.GradientTape() as gen_tape:
                generated_images = generator([random_image_noise,target], training=True) # this is a forward pass of the generator before 
                fake_output = discriminator([generated_images,target], training=True) # this which is a forward pass of the discriminator
                real_targets = tf.ones_like(fake_output) # this generates an array of ones the size of the discriminator output so that
                gen_loss = binary_cross_entropy(real_targets, fake_output) # it can be compared with the values output from the discriminator to calculate loss
                #binary cross entropy is a pre-built loss function from the tensorflow core library
            gradients_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables) # the information recorded by the GradientTape() object is then applied via a black box
            # tensorflow process to calculate updated weights for the generator
            generator_optimizer.apply_gradients(zip(gradients_of_gen, generator.trainable_variables)) # which is then backp

            #start of pre-existing reporting code
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                              epoch + 1,
                              seed)

        generator.save_weights('rock-paper-scissors/training_weights/gen_'+ str(epoch)+'.h5')
        discriminator.save_weights('rock-paper-scissors/training_weights/disc_'+ str(epoch)+'.h5')    
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                            epochs,
                            seed)

def label_gen(n_classes):
    lab = tf.random.uniform((1,), minval=0, maxval=n_classes, dtype=tf.dtypes.int32, seed=None, name=None)
    return tf.repeat(lab, [25], axis=None, name=None)

train(ds, 2)

generator.load_weights('rock-paper-scissors/training_weights/gen_99.h5')

def generate_images(model, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
    output = None
    for label in range(3):
        labels = tf.ones(10) * label
#         predictions = model([labels, test_input], training=False)
        predictions = model([test_input, labels], training=False)
        if output is None:
            output = predictions
        else:
            output = np.concatenate((output,predictions))
     
    nrow = 3
    ncol = 10
    fig = plt.figure(figsize=(25,25))
    gs = gridspec.GridSpec(nrow, ncol, width_ratios=[1, 1, 1,1, 1,1, 1, 1, 1, 1],
         wspace=0.0, hspace=0.0, top=0.2, bottom=0.00, left=0.17, right=0.845) 

    #output = output.reshape(-1, 128, 128, 3)
    #print("Generated Images are Conditioned on Label:", label_dict[np.array(labels)[0]])
    k = 0
    for i in range(nrow):
        for j in range(ncol):
            pred = (output[k, :, :, :] + 1 ) * 127.5
            pred = np.array(pred)  
            ax= plt.subplot(gs[i,j])
            ax.imshow(pred.astype(np.uint8))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axis('off')
            k += 1   

    plt.savefig('result.png',  dpi=300)
    plt.show()

num_examples_to_generate = 10
latent_dim = 100
noise = tf.random.normal([num_examples_to_generate, latent_dim])

generate_images(conditional_gen, noise)

