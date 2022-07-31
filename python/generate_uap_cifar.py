import tensorflow.compat.v1 as tf
import numpy as np
from tensorflow.python.platform import gfile
import os.path
from prepare_imagenet_data import preprocess_image_batch, create_imagenet_npy, undo_image_avg
import matplotlib.pyplot as plt
import sys, getopt
import zipfile
from timeit import time
import h5py
from keras.preprocessing.image import ImageDataGenerator

if sys.version_info[0] >= 3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve


from universal_pert import universal_perturbation

tf.disable_v2_behavior()

device = '/gpu:0'
num_classes = 10
NUM_CLASSES = 10

DATA_DIR = 'datasets2'
DATA_FILE = 'cifar_dataset.h5'
BATCH_SIZE = 1024

def jacobian(y_flat, x, inds):
    n = num_classes # Not really necessary, just a quick fix.
    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]
    _, jacobian = tf.while_loop(
        lambda j,_: j < n,
        lambda j,result: (j+1, result.write(j, tf.gradients(y_flat[inds[j]], x))),
        loop_vars)
    return jacobian.stack()

def load_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    if not os.path.exists(data_file):
        print(
            "The data file does not exist. Please download the file and put in data/ directory")
        exit(1)

    dataset = _load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = dataset['X_train']/255
    Y_train = dataset['Y_train']
    X_test = dataset['X_test']/255
    Y_test = dataset['Y_test']

    # Scale images to the [0, 1] range
    x_train = X_train.astype("float32")
    x_test = X_test.astype("float32")
    # Make sure images have shape (28, 28, 1)
    #x_train = np.expand_dims(x_train, -1)
    #x_test = np.expand_dims(x_test, -1)
    #

    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    y_train = tf.keras.utils.to_categorical(Y_train, NUM_CLASSES)
    y_test = tf.keras.utils.to_categorical(Y_test, NUM_CLASSES)

    train_gen = build_data_loader(x_train, y_train)
    test_gen = build_data_loader(x_test, y_test)
    x_train, y_train = train_gen.next()
    x_test, y_test = test_gen.next()

    return x_train, y_train, x_test, y_test


def build_data_loader(X, Y):
    datagen = ImageDataGenerator()
    generator = datagen.flow(
        X, Y, batch_size=BATCH_SIZE, shuffle=True)
    return generator

def _load_dataset(data_filename, keys=None):
    ''' assume all datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset


if __name__ == '__main__':

    # Parse arguments
    argv = sys.argv[1:]

    with tf.device(device):
        persisted_sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
        inception_model_path = os.path.join('model', 'cifar_graph.pb')

        model = os.path.join(inception_model_path)

        # Load the Inception model
        with gfile.FastGFile(model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')

        persisted_sess.graph.get_operations()

        persisted_input = persisted_sess.graph.get_tensor_by_name("x:0")
        persisted_output = persisted_sess.graph.get_tensor_by_name("sequential_1/dense_2/BiasAdd:0")

        print('>> Computing feedforward function...')
        def f(image_inp): return persisted_sess.run(persisted_output, feed_dict={persisted_input: np.reshape(image_inp, (-1, 32, 32, 3))})

        file_perturbation = os.path.join('data', 'universal_cifar.npy')

        # Load/Create data
        train_X, train_Y, test_X, test_Y = load_dataset()

        if os.path.isfile(file_perturbation) == 0:

            # TODO: Optimize this construction part!
            print('>> Compiling the gradient tensorflow functions. This might take some time...')
            y_flat = tf.reshape(persisted_output, (-1,))
            inds = tf.placeholder(tf.int32, shape=(num_classes,))
            dydx = jacobian(y_flat,persisted_input,inds)

            print('>> Computing gradient function...')
            def grad_fs(image_inp, indices): return persisted_sess.run(dydx, feed_dict={persisted_input: image_inp, inds: indices}).squeeze(axis=1)

            X = train_X

            #debug
            '''
            out = persisted_sess.run(persisted_output, feed_dict={persisted_input: np.reshape(X[0:100], (100, 32, 32, 3))})
            print (np.argmax(out, axis=1))
            print (np.argmax(train_Y[0:100], axis=1))
            test_idx = np.arange(num_classes)
            grad = grad_fs(np.reshape(X[0], (-1, 32, 32, 3)), test_idx)
            #print(grad)
            '''

            # Running universal perturbation
            v = universal_perturbation(X, f, grad_fs, delta=0.2, xi=0.1, num_classes=num_classes)

            # Saving the universal perturbation
            np.save(os.path.join(file_perturbation), v)

        else:
            print('>> Found a pre-computed universal perturbation! Retrieving it from ", file_perturbation')
            v = np.load(file_perturbation)

        print('>> Testing the universal perturbation on test set')
        labels = np.argmax(test_Y[0])
        label_original = np.argmax(persisted_sess.run(persisted_output, feed_dict={persisted_input: np.reshape(test_X[0], (-1, 32, 32, 3))}))



        # Test the perturbation on the image
        for img in test_X:
            #labels = open(os.path.join('data', 'labels.txt'), 'r').read().split('\n')

            #image_original = preprocess_image_batch([path_test_image], img_size=(256, 256), crop_size=(224, 224), color_mode="rgb")
            image_original = img
            label_original = np.argmax(f(image_original), axis=1).flatten()
            #str_label_original = labels[np.int(label_original)-1].split(',')[0]

            # Clip the perturbation to make sure images fit in uint8
            #clipped_v = np.clip(undo_image_avg(image_original[0,:,:,:]+v[0,:,:,:]), 0, 255) - np.clip(undo_image_avg(image_original[0,:,:,:]), 0, 255)
            clipped_v = v

            image_perturbed = image_original + clipped_v[None, :, :, :]
            label_perturbed = np.argmax(f(image_perturbed), axis=1).flatten()
            #str_label_perturbed = labels[np.int(label_perturbed)-1].split(',')[0]
            if label_original != label_perturbed:
                break

        # Show original and perturbed image
        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow((image_original[:, :, :] * 255).astype(dtype='uint8'), interpolation=None)
        #plt.title(str_label_original)
        plt.title(label_original)

        plt.subplot(1, 3, 2)
        plt.imshow((image_perturbed[0, 0, :, :, :] * 255).astype(dtype='uint8'), interpolation=None)
        #plt.title(str_label_perturbed)
        plt.title(label_perturbed)

        plt.subplot(1, 3, 3)
        plt.imshow((clipped_v[0, :, :, :] * 255).astype(dtype='uint8'), interpolation=None)
        #plt.title(str_label_perturbed)
        plt.title('mask')

        plt.show()
