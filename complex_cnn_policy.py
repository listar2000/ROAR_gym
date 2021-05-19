from stable_baselines.ddpg.policies import FeedForwardPolicy
from stable_baselines.common.tf_layers import conv_to_fc, linear
import numpy as np

basenet = None 
preprocess = None
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

def initialize_basenet(img_shape):
    global basenet, preprocess
    basenet = tf.keras.applications.MobileNetV2(include_top=False, input_shape=img_shape)
    basenet.trainable = False # use as feature extractor
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

def mobile_v2_pretrained(scaled_images, **kwargs):
    if not basenet:
        img_shape = scaled_images.get_shape()[1:]
        initialize_basenet(img_shape=img_shape)

    activ = tf.nn.relu

    x = preprocess(scaled_images)
    x = basenet(x, training=False)

    use_global_avg = kwargs.get("use_global_avg", True)

    if use_global_avg:
        x = global_average_layer(x)
    else:
        x = conv_to_fc(x)

    return activ(linear(x, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))

class ComplexCnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(ComplexCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        cnn_extractor=mobile_v2_pretrained, feature_extraction="cnn", **_kwargs)