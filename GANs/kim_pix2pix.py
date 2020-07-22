from matplotlib import pyplot as plt
from argparse import ArgumentParser
from imutils import paths
import keras.backend as K
import tensorflow as tf
from PIL import Image
import numpy as np
import math as mt
import datetime
import time
import cv2
import os

ap = ArgumentParser()
ap.add_argument('-d', '--dataset', required = True, help = 'path to input dataset')
ap.add_argument('-e', '--epoch', required = False, type = int, default = 100, help = 'how many iterate learning')
ap.add_argument('-b', '--batch_size', required = False, type = int, default = 1)
ap.add_argument('-r', '--resize', required = False, type = int, default=256)
ap.add_argument('-c', '--channels', required = False, type = int, default = 3)
ap.add_argument('-s', '--strides', required = False, type = int, default = 2)
args = vars(ap.parse_args())

K.set_image_data_format('channels_last')
PATH = args['dataset']+'/'

BUFFER_SIZE, BATCH_SIZE, LAMBDA = 400, args['batch_size'], 100
IMG_WIDTH, IMG_HEIGHT = args['resize'], args['resize']
OUTPUT_CHANNELS = args['channels']
EPOCHS = args['epoch']
STRIDE = args['strides']

def imgLoad(imgFile):
    image = tf.io.read_file(imgFile)
    image = tf.image.decode_jpeg(image)

    w = tf.shape(image)[1]
    w = w // 2

    realImg = image[:, :w, :]
    inputImg = image[:, w:, :]

    inputImg = tf.cast(inputImg, tf.float32)
    realImg = tf.cast(realImg, tf.float32)

    return inputImg, realImg

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, OUTPUT_CHANNELS])

  return cropped_image[0], cropped_image[1]

def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image

@tf.function()
def random_jitter(input_image, real_image):
  # resizing to resize + 30, resize + 30, 3
  input_image, real_image = resize(input_image, real_image, IMG_WIDTH + 30, IMG_HEIGHT + 30)

  # randomly cropping to 256 x 256 x 3
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image

def load_image_train(image_file):
  input_image, real_image = imgLoad(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = imgLoad(image_file)
  input_image, real_image = resize(input_image, real_image, IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def downsample(filters, size, apply_batchnorm=True):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  result.add(tf.keras.layers.LeakyReLU())

  return result
  
def upsample(filters, size, apply_dropout=False):
  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=STRIDE, padding='same', kernel_initializer=initializer, use_bias=False))

  result.add(tf.keras.layers.BatchNormalization())

  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result

def Generator():
  inputs = tf.keras.layers.Input(shape=[IMG_WIDTH,IMG_HEIGHT, OUTPUT_CHANNELS])

  size = IMG_WIDTH
  iterate = int(mt.log(size) / mt.log(STRIDE))

  down_stack = [
    downsample(size // 4, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
    downsample(size //2, 4), # (bs, 64, 64, 128)
    downsample(size, 4), # (bs, 32, 32, 256)
    downsample(size*2, 4), # (bs, 16, 16, 512)
    downsample(size*2, 4), # (bs, 8, 8, 512)
    downsample(size*2, 4), # (bs, 4, 4, 512)
    downsample(size*2, 4), # (bs, 2, 2, 512)
    downsample(size*2, 4), # (bs, 1, 1, 512)
  ]

  up_stack = [
    upsample(size*2, 4, apply_dropout=True), # (bs, 2, 2, 1024)
    upsample(size*2, 4, apply_dropout=True), # (bs, 4, 4, 1024)
    upsample(size*2, 4, apply_dropout=True), # (bs, 8, 8, 1024)
    upsample(size*2, 4), # (bs, 16, 16, 1024)
    upsample(size, 4), # (bs, 32, 32, 512)
    upsample(size // 2, 4), # (bs, 64, 64, 256)
    upsample(size //4, 4), # (bs, 128, 128, 128)
  ]

  initializer = tf.random_normal_initializer(0., 0.02)
  last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                         strides=2,
                                         padding='same',
                                         kernel_initializer=initializer,
                                         activation='tanh') # (bs, 256, 256, 3)

  x = inputs

  # Downsampling through the model
  skips = []
  for down in down_stack:
    x = down(x)
    skips.append(x)

  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    x = tf.keras.layers.Concatenate()([x, skip])

  x = last(x)

  return tf.keras.Model(inputs=inputs, outputs=x)

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

generator = Generator()
generator.summary()

# tf.keras.utils.plot_model(generator, show_shapes=True)

def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  inp = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS], name='input_image')
  tar = tf.keras.layers.Input(shape=[IMG_WIDTH, IMG_HEIGHT, OUTPUT_CHANNELS], name='target_image')

  x = tf.keras.layers.concatenate([inp, tar]) # (bs, 256, 256, channels*2)

  down1 = downsample(64, 4, False)(x) # (bs, 128, 128, 64)
  down2 = downsample(128, 4)(down1) # (bs, 64, 64, 128)
  down3 = downsample(256, 4)(down2) # (bs, 32, 32, 256)

  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 34, 34, 256)
  conv = tf.keras.layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1) # (bs, 31, 31, 512)

  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 33, 33, 512)
  last = tf.keras.layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2) # (bs, 30, 30, 1)

  return tf.keras.Model(inputs=[inp, tar], outputs=last)

discriminator = Discriminator()
# tf.keras.utils.plot_model(discriminator, show_shapes=True)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer, discriminator_optimizer=discriminator_optimizer, generator=generator, discriminator=discriminator)

def generate_images(model, test_input, tar, epoch, cnt):
  prediction = model(test_input, training=True)
  display_list = [test_input[0], tar[0], prediction[0]]
  # title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for idx in range(3):

    # plt.subplot(1, 3, i+1)
    # plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    # plt.imshow(display_list[i] * 0.5 + 0.5)

    display_list[idx] = np.array(display_list[idx]*0.5 + 0.5)
    
    # tf.keras.preprocessing.image.save_img(f'genImgs/{time.time()}.png', display_list[idx])
    display_list[idx] = tf.keras.preprocessing.image.array_to_img(display_list[idx])

    # title = 'input image' if idx == 0 else ('ground truth' if idx == 1 else 'predicted image')


  images = np.hstack([display_list[0], display_list[1], display_list[2]])
  # cv2.putText(images, 'input image', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,	0.3, (255, 255, 0), 1)
  # cv2.putText(images, 'ground truth', (10 + IMG_HEIGHT, 25),  cv2.FONT_HERSHEY_SIMPLEX,	0.3, (255, 255, 0), 1)
  # cv2.putText(images, 'predicted image', (10 + IMG_HEIGHT*2, 25),  cv2.FONT_HERSHEY_SIMPLEX,	0.3, (255, 255, 0), 1)

  if not os.path.isdir(f'genImgs/{epoch}_epoch'):
    os.makedirs(f'genImgs/{epoch}_epoch')


  cv2.imwrite(f'genImgs/{epoch}_epoch/genImg_{cnt}.jpg', images)
  #   plt.axis('off')
  # plt.show()                               

train_dataset = tf.data.Dataset.list_files(PATH+'train/*.jpg')
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(PATH+'test/*.jpg')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


@tf.function
def train_step(input_image, target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

  return gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss
def fit(train_ds, epochs, test_ds):
  for epoch in range(epochs):
    start = time.time()

    # display.clear_output(wait=True)
    print("Epoch: ", epoch)
    if epoch % 50 == 0:
      cnt = 1
      for example_input, example_target in test_ds.take(5):
        generate_images(generator, example_input, example_target, epoch, cnt)
        cnt += 1
    
    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss = train_step(input_image, target, epoch)
    print()

    # saving (checkpoint) the model every 20 epochs
    if epoch % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

      if not os.path.isdir('models/genModels'):
        os.makedirs('models/genModels')

      if not os.path.isdir('models/disModels'):
        os.makedirs('models/disModels')
        
      generator.save(f'models/genModels/gen_{epoch}.hdf5')
      discriminator.save(f'models/disModels/dis_{epoch}.hdf5')

    spendTime = (time.time() - start)
    print('-'*16+'info'+'-'*16)
    print (f'Time taken for epoch {epoch + 1} is {spendTime:.2f} sec\n')
    print(f'gen loss : {gen_total_loss:.2f}, adv loss : {gen_gan_loss:.2f}\nl1 loss : {gen_l1_loss:.2f} disc loss : {disc_loss:.2f}')
    print('-'*36+'\n')
  checkpoint.save(file_prefix = checkpoint_prefix)

fit(train_dataset, EPOCHS, test_dataset)  