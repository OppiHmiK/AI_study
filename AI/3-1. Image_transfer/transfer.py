from keras.preprocessing.image import img_to_array, load_img, save_img
from keras.applications import vgg19
import numpy as np

from keras import backend as K
from scipy.optimize import fmin_l_bfgs_b
import time
import os

tar_image_path = '../Datas/CNN/style_transfer/style/dove_mayo4.jpg'
style_image_path = '../Datas/CNN/style_transfer/style/Self-Portrait 3.jpg'


width, height = load_img(tar_image_path).size
img_height = 400
img_width = int(width * img_height / height)

def preprocessing(img_path):
    img = load_img(img_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    img = vgg19.preprocess_input(img)
    return img


def deprocessing(image):
    # NOTE : imagenet의 평균 픽셀 값을 더해 preprocess_input에서 일어나는 변환 복원
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:, :, 2] += 123.68

    # NOTE : 이미지 채널을 BGR에서 RGB로 변환
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255).astype('uint8')
    return image

# NOTE : 사전훈련된 VGG19 네트워크 로딩, 3개 이미지에 적용
target_image = K.constant(preprocessing(tar_image_path))
style_image = K.constant(preprocessing(style_image_path))

# NOTE : 생성이미지가 담길 플레이스홀더
comb_image = K.placeholder((1, img_height, img_width, 3))

# NOTE : 스타일이미지, 타깃이미지, 생성이미지를 하나의 배치로 합침.
input_tensor = K.concatenate([target_image, style_image, comb_image], axis = 0)
model = vgg19.VGG19(input_tensor=input_tensor, weights = 'imagenet', include_top=False)
print('model load complete')


# NOTE : 컨텐츠 손실
def content_loss(base, comb):
    return K.sum(K.square(comb - base))*0.5

# NOTE : 스타일 손실
def gram_matrix(img):
    feat = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
    gram = K.dot(feat, K.transpose(feat))
    return gram

def style_loss(style, comb):
    S = gram_matrix(style)
    C = gram_matrix(comb)
    channels = 3

    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

# NOTE : 총 변위 손실
def total_variation_loss(img):
    a = K.square(img[:, :img_height - 1, :img_width - 1, :] - img[:, 1:, :img_width - 1, :])
    b = K.square(img[:, :img_height - 1, :img_width - 1, :] - img[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


# NOTE : 층 이름과 활성화 텐서를 매핑한 딕셔너리
out_dict = dict([(layer.name, layer.output) for layer in model.layers])

# NOTE : 컨텐츠 손실에 사용할 층
content_layer = 'block5_conv2'

# NOTE : 스타일 손실에 사용할 층
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# NOTE : 손실 항목의 가중치 평균에 사용할 가중치
total_variation_weights = 1e-4
style_weights = 1.
content_weights = 0.025

# NOTE :컨텐츠 손실을 더함.
# NOTE : 모든 손실 요소를 더해 하나의 스칼라 변수로 손실을 정의
loss = K.variable(0.)
layer_feat = out_dict[content_layer]
tar_img_feat = layer_feat[0, :, :, :]
comb_feat = layer_feat[2, :, :, :]
loss += content_weights * content_loss(tar_img_feat, comb_feat)

# NOTE : 각 타깃 층에 대한 스타일 손실을 더함.
for layer_name in style_layers:
    layer_feat = out_dict[layer_name]
    style_refer_feat = layer_feat[1, :, :, :]
    comb_feat = layer_feat[2, :, :, :]
    sl = style_loss(style_refer_feat, comb_feat)
    loss += (style_weights / len(style_layers)) * sl

# NOTE : 총 변위 손실을 더함.
loss += total_variation_weights * total_variation_loss(comb_image)

# NOTE : 경사 하강법 단계 설정

# NOTE : 손실에 대한 생성된 이미지의 Gradient를 구함.
grads = K.gradients(loss, comb_image)[0]

# NOTE : 현재 손실과 Gradient 값을 추출하는 케라스 함수 객체.
loss_n_grads = K.function([comb_image], [loss, grads])


class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_value = None

    def loss(self, image):
        assert self.loss_value is None
        image = image.reshape((1, img_height, img_width, 3))
        out = loss_n_grads([image])
        loss_value = out[0]
        grads_value = out[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grads_value = grads_value
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grads_value = np.copy(self.grads_value)

        self.loss_value = None
        self.grads_value = None
        return grads_value


evaluator = Evaluator()

result_prefix = 'style_transfer_result'
idx = 50

save_path = '../Datas/CNN/style_transfer/transfer/van_gogh/gogh_for_malnyeon'
x = preprocessing(tar_image_path)
x = x.flatten()

if not os.path.isdir(save_path):
    os.mkdir(save_path)
os.chdir(save_path)

# NOTE : 초깃값은 타깃 이미지

for rep in range(idx+1):
    print('반복 횟수:', rep+1)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x,
                                     fprime=evaluator.grads, maxfun=20)
    print('현재 손실 값:', min_val)
    # 생성된 현재 이미지를 저장합니다
    img = x.copy().reshape((img_height, img_width, 3))

    img = deprocessing(img)
    fname = result_prefix + f'gogh_for_malnyeon{rep+1}th.png'

    if rep % 10 == 0:
        save_img(fname, img)

    end_time = time.time()
    print('저장 이미지: ', fname)
    print('%d 번째 반복 완료: %ds' % (rep+1, end_time - start_time))


