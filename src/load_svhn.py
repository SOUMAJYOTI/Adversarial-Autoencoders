import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from scipy.io import loadmat
import h5py

# %matplotlib inline
plt.rcParams['figure.figsize'] = (16.0, 4.0)


import urllib.request


def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']


def plot_images(img, labels, nrows, ncols):
    """ Plot nrows x ncols images
    """
    fig, axes = plt.subplots(nrows, ncols)
    for i, ax in enumerate(axes.flat):
        if img[i].shape == (32, 32, 3):
            ax.imshow(img[i])
        else:
            ax.imshow(img[i,:,:,0])
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(labels[i])

    plt.show()


def rgb2gray(images):
    """Convert images from rbg to grayscale
    """
    return np.expand_dims(np.dot(images, [0.2989, 0.5870, 0.1140]), axis=3)


def load_transform():
    ''' Load the SVHN data '''

    X_train, y_train = load_data('Data/SVHN/train_32x32.mat')
    X_test, y_test = load_data('Data/SVHN/test_32x32.mat')

    print("Training", X_train.shape, y_train.shape)
    print("Test", X_test.shape, y_test.shape)

    print("Transposing.......")
    # Transpose the image arrays
    X_train, y_train = X_train.transpose((3, 0, 1, 2)), y_train[:, 0]
    X_test, y_test = X_test.transpose((3, 0, 1, 2)), y_test[:, 0]

    print("Training", X_train.shape)
    print("Test", X_test.shape)
    print('')

    # Calculate the total number of images
    num_images = X_train.shape[0] + X_test.shape[0]

    print("Total Number of Images", num_images)

    ##################################################################################
    ''' Some plot stats '''
    # Plot some training set images# Plot s
    # plot_images(X_train, y_train, 2, 8)
    #
    # # Plot some test set images
    # plot_images(X_test, y_test, 2, 8)

    # fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True)

    # fig.suptitle('Class Distribution', fontsize=14, fontweight='bold', y=1.05)
    #
    # ax1.hist(y_train, bins=10)
    # ax1.set_title("Training set")
    # ax1.set_xlim(1, 10)
    #
    # ax2.hist(y_test, color='g', bins=10)
    # ax2.set_title("Test set")

    # fig.tight_layout()
    #
    # plt.show()

    ############################################################################

    # Transform the images to greyscale
    train_greyscale = rgb2gray(X_train).astype(np.float32)
    test_greyscale = rgb2gray(X_test).astype(np.float32)

    # Keep the size before convertion
    size_before = (X_train.nbytes, X_test.nbytes)

    # Size after transformation
    size_after = (train_greyscale.nbytes, test_greyscale.nbytes)

    print("Dimensions")
    print("Training set", X_train.shape, train_greyscale.shape)
    print("Test set", X_test.shape, test_greyscale.shape)
    print('')

    print("Data Type")
    print("Training set", X_train.dtype, train_greyscale.dtype)
    print("Test set", X_test.dtype, test_greyscale.dtype)
    print('')

    # plot_images(X_train, y_train, 1, 10)
    plot_images(train_greyscale, y_train, 1, 10)

    ''' Save the grayscale images '''
    # Create file
    h5f = h5py.File('Data/SVHN/SVHN_single_grey.h5', 'w')

    # Store the datasets
    h5f.create_dataset('X_train', data=train_greyscale)
    h5f.create_dataset('y_train', data=y_train)
    h5f.create_dataset('X_test', data=test_greyscale)
    h5f.create_dataset('y_test', data=y_test)

    # Close the file
    h5f.close()


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

if __name__ == "__main__":
    ''' This is executed once '''
    load_transform()

    ''' Read the images '''
    f = h5py.File('Data/SVHN/SVHN_single_grey.h5', 'r')

    X_tr = f['X_train']
    y_tr = f['y_train']
    X_te = f['X_test']
    y_te = f['y_test']

    print("Dimensions")
    print("Training set", X_tr.shape)
    print("Test set", X_te.shape)

    X_next, y_next = next_batch(20, X_tr, y_tr)
    print("Dimensions")
    print("Training set", X_next.shape)
    print("Test set", y_next.shape)