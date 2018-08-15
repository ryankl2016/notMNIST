import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

parentDir = 'notMNIST_small/'
print(parentDir)
data = []
total = 0
good = 0
for folder in os.listdir(parentDir):
    if folder != '.DS_Store':
        for file in os.listdir(parentDir + folder):
            if total % 10000 == 0:
                print(total, good)
            total += 1
            try:
                img_path = parentDir + folder + '/' + file
                img = Image.open(img_path)
                data.append([img_path, folder])
                good += 1
            except:
                pass

dataset = pd.DataFrame(data)
dataset.head()

batch_size = 8
num_epochs = 500
def input_func(features, labels, batch_size):

    def parser(image, label):

        img = tf.image.decode_png(tf.read_file(image))
        img = tf.image.resize_images(img, tf.constant([1, 784]))
        img = tf.reshape(img, [28, 28, 1])
        img = tf.cast(img, tf.float32, "cast")
#         image = tf.reshape(image, [28, 28, 1])
#         label = tf.one_hot(indices = label, depth = 10)
        return img, label

# #     features = tf.convert_to_tensor(data[[i for i in range(784)]])
# #     labels = tf.convert_to_tensor(pd.factorize(data['label'])[0])

#     return tf.estimator.inputs.numpy_input_fn(
#         x = {'x' : features},
#         y = labels,
#         batch_size = batch_size,
#         num_epochs = num_epochs,
#         shuffle = True
#     )


    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.map(parser)
    dataset = dataset.batch(batch_size)
    return dataset
#     feature_dict = {feature : tf.convert_to_tensor(data[feature]) for feature in data if feature != 'label'}
#     labels = tf.convert_to_tensor(pd.factorize(data['label'])[0])
#     dataset = tf.data.Dataset.from_tensor_slices((feature_dict, labels))
    dataset = dataset.batch(batch_size)
#     dataset = dataset.repeat(num_epochs)
    return dataset

def my_model(features, labels, mode, params):
    #initialize input by reshaping and casting for network
    #img = tf.image.decode_png(tf.read_file(features['x'][0]))
    # img = np.array( img, dtype='uint8' ).flatten()

    # FIRST LAYER
    # ---conv layer with 32 filters, 5x5 kernel, and relu activation
    # ---pool layer with 2x2 pool window and stride of 2x2
    conv1 = tf.layers.conv2d(inputs=features, filters=32, kernel_size=(5, 5), padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=(2, 2), strides=(2, 2))

    # SECOND LAYER
    # ---conv layer with 64 filters, 5x5 kernel, and relu activation
    # ---pool layer with 2x2 pool window and stride of 2x2
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=(5, 5), padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=(2, 2), strides=(2, 2))

    # DENSE LAYER
    # ---flatten output into vector
    # ---dropout to prevent overfitting
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    onehot_labels = tf.reshape(onehot_labels, [-1, 10])
    loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
#     loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    print(labels.shape)
    print(predictions["classes"].shape)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


# Fetch the data
X_train, X_test, y_train, y_test = train_test_split(dataset[0], pd.factorize(dataset[1])[0], test_size=0.33, random_state=42)

# Build CNN.
classifier = tf.estimator.Estimator(model_fn=my_model)

# Train the Model.
X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

# print(X_train, y_train)

# train_input_func = tf.estimator.inputs.numpy_input_fn(x = {'x' : X_train},
#                                                       y = y_train,
#                                                       batch_size = batch_size,
#                                                       num_epochs = num_epochs,
#                                                       shuffle = True
#                                                      )
classifier.train(input_fn=lambda:input_func(X_train, y_train, batch_size), steps = 22000)

# Evaluate the model.
# eval_input_func = tf.estimator.inputs.numpy_input_fn(x = {'x' : X_test},
#                                                       y = y_test,
#                                                       batch_size = batch_size,
#                                                       num_epochs = num_epochs,
#                                                       shuffle = True
#                                                      )
eval_result = classifier.evaluate(input_fn=lambda:input_func(X_test, y_test, batch_size))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
