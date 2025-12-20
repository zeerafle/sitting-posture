import tensorflow as tf
import numpy as np
from data import BodyPart

def get_center(land, a, b):
    l = tf.gather(land, a.value, axis=1)
    r = tf.gather(land, b.value, axis=1)
    return (l + r) * 0.5

def normalize(landmarks):
    center = get_center(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    landmarks -= tf.expand_dims(center,1)
    # pose‐size as before…
    size = tf.maximum(
      tf.linalg.norm(get_center(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER))*2.5,
      tf.reduce_max(tf.norm(landmarks - tf.expand_dims(center,1), axis=-1))
    )
    return landmarks / size

def landmarks_to_embedding(row):
    L = tf.reshape(np.array(row),(-1,13,3))
    norm = normalize(L[:,:,0:2])
    return tuple(tf.reshape(norm,(13*2,)).numpy())
