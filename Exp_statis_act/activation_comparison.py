# -*- coding: utf-8 -*-
from keras import Model
from keras.models import load_model
import h5py
import os
import numpy as np
from sklearn.metrics import accuracy_score
from keras import backend as K


def HDF5_structure(data):
    root = list(map(lambda x: x[0], data.items()))
    print(root)
    final_path = []
    data_path = []
    while True:
        if len(root) == 0:
            break
        else:
            for item in root:
                if isinstance(data[item], h5py._hl.dataset.Dataset) or len(data[item].items()) == 0:
                    root.remove(item)
                    final_path.append(item)
                    if isinstance(data[item], h5py._hl.dataset.Dataset):
                        data_path.append(item)
                else:
                    for sub_item in data[item].items():
                        root.append(os.path.join(item, sub_item[0]))
                    root.remove(item)
    return data_path


def accuracy(model, images, labels):
    '''
    model: DNN_model
    return : acc of mnist
    '''
    pred = model.predict(images)
    # print(mnist.test.images.shape) 10000*784
    # print(mnist.test.images.dtype) float32
    # print(list(mnist.test.images[0])) 784 vector
    pred = list(map(lambda x: np.argmax(x), pred))
    test_label = list(map(lambda x: np.argmax(x), labels))
    return accuracy_score(test_label, pred)


def getActivationLayers(model):
    intermediate_layer_model_1 = Model(inputs=model.input, outputs=model.get_layer("activation_1").output)
    intermediate_layer_model_2 = Model(inputs=model.input, outputs=model.get_layer("activation_2").output)
    intermediate_layer_model_3 = Model(inputs=model.input, outputs=model.get_layer("activation_3").output)
    return intermediate_layer_model_1, intermediate_layer_model_2, intermediate_layer_model_3


if __name__ == '__main__':

    model_path = 'model.hdf5'
    model = load_model(model_path)
    images_data = []
    labels_data = []
    for i in range(10):
        data = np.load("data/" + str(i) + ".npy")
        images_data.append(data)
        label = np.zeros(10, np.int32)
        label[i] = 1
        labels_data.append(label)
    images_data = np.array(images_data)
    labels_data = np.array(labels_data)
    # print(np.array(images_data).shape)
    # print(np.array(images_data)[0].shape)
    # print(np.array(labels_data).shape)

    # print activation 1
    print("activation_1")
    activation1, activation2, activition3 = getActivationLayers(model)
    test_image_1 = images_data[1][0]
    test_image_2 = images_data[2][0]
    images = np.append(test_image_1, test_image_2).reshape((-1, 784))
    print(images.shape)

    hidden_activation_1_output = activation1.predict(images)
    # print(hidden_activation_1_output.shape)
    # print(hidden_activation_1_output)
    hidden_activation_1_output[hidden_activation_1_output != 0] = 1
    # 设置为int型好做&操作
    hidden_activation_1_output.astype(np.int32)
    # print(hidden_activation_1_output)
    # print activation 2
    print("activation_2")

    hidden_activation_2_output = activation2.predict(images)
    # print(hidden_activation_2_output.shape)
    # print(hidden_activation_2_output)
    hidden_activation_2_output[hidden_activation_2_output != 0] = 1
    # 设置为int型 好做&操作
    hidden_activation_2_output.astype(np.int32)
    # print(hidden_activation_2_output)
    # print activation 3
    print("activation_3")

    hidden_activation_3_output = activition3.predict(images)
    # print(type(hidden_activation_1_output))
    # print(hidden_activation_3_output.shape)
    # print(hidden_activation_3_output)

    # 激活的神经元 与 操作
    common2 = np.bitwise_and(hidden_activation_2_output[0].astype(np.int32),
                             hidden_activation_2_output[1].astype(np.int32))
    common1 = np.bitwise_and(hidden_activation_1_output[0].astype(np.int32),
                             hidden_activation_1_output[1].astype(np.int32))
    # print(common2)
    # print(common1)
    result1 = np.where(common1 > 0)
    result2 = np.where(common2 > 0)
    print(result1)
    print(result2)
    
    K.clear_session()
