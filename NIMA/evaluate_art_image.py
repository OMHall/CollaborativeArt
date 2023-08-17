import numpy as np

from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.utils.image_utils import load_img, img_to_array
import tensorflow as tf

from NIMA.utils.score_utils import mean_score


# Idea: serve as fitness function, evaluate one image and return value

def evaluation(img_path):
    
    with tf.device('/CPU:0'):
        # set up model
        base_model = InceptionResNetV2(input_shape=(None, None, 3), include_top=False, pooling='avg', weights=None)
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)
        model = Model(base_model.input, x)
        model.load_weights('./NIMA/weights/inception_resnet_weights.h5')

        # load image, unfortunately from path due to problems in converting torch.tensor to PilImage
        target_size = (224, 224)
        image = load_img(img_path, target_size=target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # evaluate the image
        score = model.predict(image, batch_size=1, verbose=0)[0]
        score = mean_score(score)

    return score

