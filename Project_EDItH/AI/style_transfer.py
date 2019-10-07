# -*- coding : utf-8 -*-

from keras.preprocessing.image import img_to_array, load_img, save_img
from keras.applications import vgg19
import numpy as np

from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import time, os


class setting_image:

    def __init__(self, style_path, target_path):
        style_img = load_img(style_path)
        target_img = load_img(target_path)

        w, h = target_img.size
        img_h = 400
        img_w = int(w*img_h / h)

        self.img_h, self.img_w = img_h, img_w
        self.style_img, self.target_img = style_img, target_img

        target_img = K.constant(self.preprocess(target_path))
        style_img = K.constant(self.preprocess(style_path))
        self.style_img, self.target_img = style_img, target_img

    def preprocess(self, img_path):

        img_w, img_h = self.img_w, self.img_h
        img = load_img(img_path, target_size=(img_h, img_w))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return img

    def deprocess(self, img):
        # NOTE : preprocess input에서 일어난 변환 복원
        img[:, :, 0] += 103.939
        img[:, :, 1] += 116.779
        img[:, :, 2] += 123.68

        # NOTE : 이미지 채널을 BGR에서 RGB로 변환
        img = img[:, :, ::-1]

        # NOTE : 최대값을 255, 최소값을 0으로 고정
        img = np.clip(img, 0, 255).astype('uint8')
        return img

class losses:

    def __init__(self, style_path, target_path):

        data = setting_image(style_path, target_path)
        self.img_w, self.img_h = data.img_w, data.img_h

    # NOTE : 컨텐츠 손실
    def content_loss(self, base, comb):
        return np.sum(np.square(comb - base))

    # NOTE : Gram 행렬
    def gram_matrix(self, img):
        feat = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
        gram = np.dot(feat, np.transpose(feat))
        return gram

    # NOTE : 스타일 손실
    def style_loss(self, style, comb):
        img_w, img_h = self.img_w, self.img_h

        S = self.gram_matrix(style)
        C = self.gram_matrix(comb)
        channels = 3

        size = img_h*img_w
        return np.sum(np.square(S - C)) / (4.*(channels**2) *(size**2))

    # NOTE : 총 변위 손실
    def total_loss(self, img):
        img_h, img_w = self.img_h, self.img_w

        a = np.square(img[:, :img_h - 1, :img_w - 1, :] - img[:, 1:, :img_w - 1, :])
        b = np.square(img[:, :img_h - 1, :img_w - 1, :] - img[:, :img_h - 1, 1:, :])
        return np.sum(np.power(a+b, 1.25))

class painter:

    def __init__(self, style_path, target_path, epochs = 20):

        self.cost = losses(style_path, target_path)
        self.data = setting_image(style_path, target_path)

        img_h, img_w = self.data.img_h, self.data.img_w
        # NOTE : 생성이미지가 담길 플레이스홀더
        self.comb_img = K.placeholder((1, img_h, img_w, 3))

        # NOTE : 스타일이미지, 원본이미지, 생성이미지를 하나의 배치로 합침.
        input_tensor = K.concatenate([self.data.target_img, self.data.style_img, self.comb_img], axis = 0)
        self.input_tensor = input_tensor

        self.loss_val = None
        self.grad_val = None

        x = self.data.preprocess(target_path)
        x = x.flatten()

        for epoch in range(epochs):
            print('반복 횟수 : ', epoch)
            start_time = time.time()
            x, min_val, info = fmin_l_bfgs_b(self.loss, x, fprime = self.grads, maxfun = 20)
            img = x.copy().reshape((img_h, img_w, 3))

            img = self.data.deprocess(img)
            fname = f'style_transfer_at_{epoch}_epochs.png'

            save_img(fname, img)
            end_time = time.time()
            print('저장 이미지: ', fname)
            print(f'{epoch} 번째 반복 완료: {end_time - start_time}s')

    def build_model(self):

        model = vgg19.VGG19(input_tensor = self.input_tensor, weights = 'imagenet', include_top = False)
        layer_dict = dict([(layer.name, layer.output) for layer in model.layers])

        content_layer = 'block5_conv2'
        style_layer = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

        total_weights = 1e-4
        style_weights = 1.
        content_weights = 0.025

        losses = K.variable(0.)
        layer_feat = layer_dict[content_layer]
        tar_img_feat = layer_feat[0, :, :, :]
        comb_feat = layer_feat[2, :, :, :]
        losses += content_weights*self.cost.content_loss(tar_img_feat, comb_feat)

        for layer_name in style_layer:
            layer_feat = layer_dict[layer_name]
            style_refer_feat = layer_feat[1, :, :, :]
            comb_feat = layer_feat[2, :, :, :]
            s_loss = self.cost.style_loss(style_refer_feat, comb_feat)
            losses += (style_weights / len(style_layer))*s_loss


        losses += total_weights * self.cost.total_loss(self.comb_img)
        grads = K.gradients(losses, self.comb_img)[0]
        self.loss_n_grads = K.function([self.comb_img], [losses, grads])

    def loss(self, img):
        assert self.loss_val is None
        data = self.data

        self.build_model()
        img = img.reshape((1, data.img_h, data.img_w, 3))
        out = loss_n_grad([img])
        loss_val = out[0]
        grads_val = out[1].flatten().astype('float64')
        self.loss_val = loss_val
        self.grads_val = grads_val

        return self.loss_val

    def grads(self, x):
        assert self.loss_val is not None
        grads_val = np.copy(self.grads_val)

        self.loss_val = None
        self.grads_val = None
        return grads_val


if __name__ == '__main__':
    style_path = './Wheat Field with Cypresses.jpg'
    target_path = './seoul.jpg'
    m = painter(style_path, target_path)
