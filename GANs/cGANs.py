from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential, Model
from keras.utils import to_categorical
from argparse import ArgumentParser
from keras.optimizers import Adam
# from keras.datasets import mnist
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import cv2
import os

ap = ArgumentParser()
ap.add_argument('-n', '--num_classes', required = False, default = 10, type = int, help = 'how many classes on your dataset')
ap.add_argument('-e', '--epoch', required = False, default = 1500, type = int, help = 'how many iterate for learning')
ap.add_argument('-c', '--channel', required = False, default = 1, type = int, help = 'what is the your image channel')
ap.add_argument('-d', '--dataset', required = True, help = 'path to your image dataset')
ap.add_argument('-s', '--sample_interval', required = False, default= 50, type = int)
ap.add_argument('-b', '--batch_size', required = False, default = 32, type = int)
ap.add_argument('-r', '--resize', required = False, default = 130, type = int)
opt = ap.parse_args()

class CGAN():
    def __init__(self, *args):
        # Input shape

        self.dataset = args[0]
        self.dataset = list(sorted(paths.list_images(self.dataset)))
        self.channels = args[1]
        self.num_classes = args[2]

        shapeConfirm = cv2.imread(self.dataset[0])
        shapeConfirm = cv2.resize(shapeConfirm, (opt.resize, opt.resize))
        
        self.img_cols, self.img_rows = shapeConfirm.shape[:2]
        self.img_shape = (self.img_rows, self.img_cols, self.channels)

        self.latent_dim = 100
        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model([noise, label], valid)
        self.combined.compile(loss=['binary_crossentropy'],
            optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        iw, ih = (int(self.img_rows / 4) + int(self.img_rows % 4)), (int(self.img_cols / 4) + int(self.img_cols % 4))
        model.add(Dense(iw*ih*256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Reshape([iw, ih, 256]))
        model.add(Conv2D(64, (3, 3), padding = 'same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('tanh'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(128, (3, 3), padding = 'same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation('tanh'))
        model.add(UpSampling2D((2, 2)))
        model.add(Conv2D(1, (2, 2), activation = 'tanh', padding = 'same'))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(self.img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.img_shape))(label))
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def dataLoad(self, resize):
        
        imgList = []
        lbList = []

        for img in self.dataset:
            
            image = cv2.imread(img)
            lb = img.split(os.path.sep)[-2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (resize, resize))

            imgList.append(image)
            lbList.append(lb)
        
        imgList = np.array(imgList)
        lbList = np.array(lbList)

        lb = LabelBinarizer()
        lbList = lb.fit_transform(lbList)
        lbList = to_categorical(lbList, self.num_classes)

        return imgList, lbList

    def train(self, *args):

        resize = args[2]
        # Load the dataset
        (X_train, y_train) = self.dataLoad(resize)

        # Configure input
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((args[1], 1))
        fake = np.zeros((args[1], 1))

        for epoch in range(args[0]):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], args[1])
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (args[1], 100))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels
            sampled_labels = np.random.randint(0, 10, args[1]).reshape(-1, 1)

            # Train the generator
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % args[3] == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 1, self.num_classes
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = self.generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(c):
            axs[i].imshow(gen_imgs[cnt,:,:,0], cmap='gray')
            axs[i].set_title("lb Idx: %d" % sampled_labels[cnt])
            axs[i].axis('off')
            cnt += 1

        os.makedirs('images', exist_ok = True)
        fig.savefig("images/%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    cgan = CGAN(opt.dataset, opt.channel, opt.num_classes)
    cgan.train(opt.epoch, opt.batch_size, opt.resize, opt.sample_interval)