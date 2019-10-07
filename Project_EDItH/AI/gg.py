import EDItH_sub_pack as esp
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras import models

image = esp.preprocess(paths = './datas', is_sub = False, format='jpg')
resized = image.resize(cropped=False)
print(resized[1].shape)

plt.imshow(resized[1])
plt.show()
resized = np.expand_dims(resized, axis=1)


custom_obj = {'model' : models.Model}
m = load_model('./weights/human_faces.h5', custom_objects=custom_obj)
m2 = load_model('./weights/human_faces_b.h5', custom_objects=custom_obj)
m3 = load_model('./weights/human_faces_c.h5', custom_objects=custom_obj)
m4 = load_model('./weights/human_faces_d.h5', custom_objects=custom_obj)
m5 = load_model('./weights/human_faces_e.h5', custom_objects=custom_obj)

for img in resized:
    p = m.predict(img)
    p2 = m.predict(img)
    p3 = m.predict(img)
    p4 = m.predict(img)
    p5 = m.predict(img)

    ensemble = 0.2*(p+p2+p3+p4+p5)
    print(np.argmax(ensemble))