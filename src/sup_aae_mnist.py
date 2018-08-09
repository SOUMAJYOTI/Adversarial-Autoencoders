import tensorflow as tf
import numpy as np
import datetime
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tensorflow.examples.tutorials.mnist import input_data


'''
The main idea is to separate style and labels - see the coding space
once it is separated

'''

# Get the MNIST data
mnist = input_data.read_data_sets('./Data', one_hot=True)


# Parameters
input_dim = 784
n_l1 = 1000
n_l2 = 1000
z_dim = 10
y_dim = 10
batch_size = 100
n_epochs = 100
learning_rate = 0.001
beta1 = 0.9
results_path = 'Results/Supervised'
n_labels = 10
n_labeled = 1000
n_unlabeled = 10000


# Placeholders for input data and the targets
x_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Input')
y_input = tf.placeholder(dtype=tf.float32, shape=[batch_size, n_labels], name='Labels')
x_target = tf.placeholder(dtype=tf.float32, shape=[batch_size, input_dim], name='Target')
real_distribution = tf.placeholder(dtype=tf.float32, shape=[batch_size, z_dim], name='Real_distribution')
manual_decoder_input = tf.placeholder(dtype=tf.float32, shape=[1, z_dim + n_labels], name='Decoder_input')



def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = ""#""/{0}_{1}_{2}_{3}_{4}_{5}_Semi_Supervised". \
        # format(datetime.datetime.now(), z_dim, learning_rate, batch_size, n_epochs, beta1)
    tensorboard_path = results_path + '/Tensorboard'
    saved_model_path = results_path + '/Saved_models/'
    log_path = results_path  + '/log'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)



    return tensorboard_path, saved_model_path, log_path


def generate_image_grid(sess, op):
    """
    Generates a grid of images by passing a set of numbers to the decoder and getting its output.
    :param sess: Tensorflow Session required to get the decoder output
    :param op: Operation that needs to be called inorder to get the decoder output
    :return: None, displays a matplotlib window with all the merged images.
    """
    nx, ny = 10, 10
    random_inputs = np.random.randn(10, z_dim) * 5.
    sample_y = np.identity(10)
    plt.subplot()
    gs = gridspec.GridSpec(nx, ny, hspace=0.05, wspace=0.05)
    i = 0
    for r in random_inputs:
        for t in sample_y:
            r, t = np.reshape(r, (1, z_dim)), np.reshape(t, (1, y_dim))
            dec_input = np.concatenate((t, r), 1)
            x = sess.run(op, feed_dict={manual_decoder_input: dec_input})
            ax = plt.subplot(gs[i])
            i += 1
            img = np.array(x.tolist()).reshape(28, 28)
            ax.imshow(img, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('auto')
    plt.show()


def dense(x, n1, n2, name):
    """
    Used to create a dense layer.
    :param x: input tensor to the dense layer
    :param n1: no. of input neurons
    :param n2: no. of output neurons
    :param name: name of the entire dense layer.i.e, variable scope name.
    :return: tensor with shape [batch_size, n2]
    """
    with tf.variable_scope(name, reuse=None):
        weights = tf.get_variable("weights", shape=[n1, n2],
                                  initializer=tf.random_normal_initializer(mean=0., stddev=0.01))
        bias = tf.get_variable("bias", shape=[n2], initializer=tf.constant_initializer(0.0))
        out = tf.add(tf.matmul(x, weights), bias, name='matmul')
        return out



# The autoencoder network
def encoder(x, reuse=False):
    """
    Encode part of the autoencoder.
    :param x: input to the autoencoder
    :param reuse: True -> Reuse the encoder variables, False -> Create or search of variables before creating
    :param supervised: True -> returns output without passing it through softmax,
                       False -> returns output after passing it through softmax.
    :return: tensor which is the classification output and a hidden latent variable of the autoencoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Encoder'):
        e_dense_1 = tf.nn.relu(dense(x, input_dim, n_l1, 'e_dense_1'))
        e_dense_2 = tf.nn.relu(dense(e_dense_1, n_l1, n_l2, 'e_dense_2'))
        latent_variable = dense(e_dense_2, n_l2, z_dim, 'e_latent_variable')
        return latent_variable


def decoder(x, reuse=False):
    """
    Decoder part of the autoencoder.
    :param x: input to the decoder
    :param reuse: True -> Reuse the decoder variables, False -> Create or search of variables before creating
    :return: tensor which should ideally be the input given to the encoder.
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Decoder'):
        d_dense_1 = tf.nn.relu(dense(x, z_dim + n_labels, n_l2, 'd_dense_1'))
        d_dense_2 = tf.nn.relu(dense(d_dense_1, n_l2, n_l1, 'd_dense_2'))
        output = tf.nn.sigmoid(dense(d_dense_2, n_l1, input_dim, 'd_output'))
        return output


def discriminator(x, reuse=False):
    """
    Discriminator that is used to match the posterior distribution with a given prior distribution.
    :param x: tensor of shape [batch_size, z_dim]
    :param reuse: True -> Reuse the discriminator variables,
                  False -> Create or search of variables before creating
    :return: tensor of shape [batch_size, 1]
    """
    if reuse:
        tf.get_variable_scope().reuse_variables()
    with tf.name_scope('Discriminator'):
        dc_den1 = tf.nn.relu(dense(x, z_dim, n_l1, name='dc_den1'))
        dc_den2 = tf.nn.relu(dense(dc_den1, n_l1, n_l2, name='dc_den2'))
        output = dense(dc_den2, n_l2, 1, name='dc_output')
        return output

def next_batch(x, y, batch_size):
    """
    Used to return a random batch from the given inputs.
    :param x: Input images of shape [None, 784]
    :param y: Input labels of shape [None, 10]
    :param batch_size: integer, batch size of images and labels to return
    :return: x -> [batch_size, 784], y-> [batch_size, 10]
    """
    index = np.arange(n_labeled)
    random_index = np.random.permutation(index)[:batch_size]
    return x[random_index], y[random_index]


def plot_latent_2d(latent_vars, target=None, fig_dirpath=None, fig_name=None):
    """
    Plot in 2D samples from the latent space.
    Args:
        latent_vars: ndarray, the latent samples with shape (N, 2)
        target: ndarray, the numeric labels used for coloring of the latent samples, shape is (N, 1)
        fig_dirpath: str, optional path to folder where the figure will be saved and not showed
        fig_name: str, optional name of the plot figure
    Returns:
    """
    fig_name = fig_name or 'latent_samples.png'
    if not fig_name.endswith('.png'):
        fig_name += '.png'
    print("Plotting 2D latent space.")
    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap('viridis')
    if target is not None:
        if target.dtype.type is np.str_:
            map2digit = {k: i for i, k in enumerate(np.unique(target))}
            target = list(map(lambda x: map2digit[x], target))
        plt.scatter(latent_vars[:, 0], latent_vars[:, 1], c=target, s=1, cmap=cmap)
    else:
        raise NotImplementedError
    # n_distinct = np.unique(target).size
    # _colorbar_index(ncolors=n_distinct, cmap=cmap)
    # plt.colorbar()
    # if fig_dirpath is not None:
    #     if not os.path.exists(fig_dirpath):
    #         os.makedirs(fig_dirpath)
    #     plt.savefig(os.path.join(fig_dirpath, fig_name))
    # else:
    plt.show()

# def load_images():


def train(train_model=True):
    """
      Used to train the autoencoder by passing in the necessary inputs.
      :param train_model: True -> Train the model, False -> Load the latest trained model and show the image grid.
      :return: does not return anything
      """
    with tf.variable_scope(tf.get_variable_scope()):
        encoder_output = encoder(x_input)
        # Concat class label and the encoder output
        decoder_input = tf.concat([y_input, encoder_output], 1)
        decoder_output = decoder(decoder_input)

    with tf.variable_scope(tf.get_variable_scope()):
        d_real = discriminator(real_distribution)
        d_fake = discriminator(encoder_output, reuse=True)

    with tf.variable_scope(tf.get_variable_scope()):
        decoder_image = decoder(manual_decoder_input, reuse=True)

    # Autoencoder loss
    autoencoder_loss = tf.reduce_mean(tf.square(x_target - decoder_output))

    # Discriminator Loss
    dc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real), logits=d_real))
    dc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake), logits=d_fake))
    dc_loss = dc_loss_fake + dc_loss_real

    # Generator loss
    generator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake), logits=d_fake))

    all_variables = tf.trainable_variables()
    dc_var = [var for var in all_variables if 'dc_' in var.name]
    en_var = [var for var in all_variables if 'e_' in var.name]

    # Optimizers
    autoencoder_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   beta1=beta1).minimize(autoencoder_loss)
    discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                     beta1=beta1).minimize(dc_loss, var_list=dc_var)
    generator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                 beta1=beta1).minimize(generator_loss, var_list=en_var)

    init = tf.global_variables_initializer()

    # # Reshape images to display them
    # input_images = tf.reshape(x_input, [-1, 28, 28, 1])
    # generated_images = tf.reshape(decoder_output, [-1, 28, 28, 1])

    # Tensorboard visualization
    tf.summary.scalar(name='Autoencoder Loss', tensor=autoencoder_loss)
    tf.summary.scalar(name='Discriminator Loss', tensor=dc_loss)
    tf.summary.scalar(name='Generator Loss', tensor=generator_loss)
    tf.summary.histogram(name='Encoder Distribution', values=encoder_output)
    tf.summary.histogram(name='Real Distribution', values=real_distribution)
    # tf.summary.image(name='Input Images', tensor=input_images, max_outputs=10)
    # tf.summary.image(name='Generated Images', tensor=generated_images, max_outputs=10)
    summary_op = tf.summary.merge_all()


    # Saving the model
    saver = tf.train.Saver()
    step = 0

    ''' Start of the Tensorflow Session '''
    with tf.Session() as sess:
        if train_model:
            tensorboard_path, saved_model_path, log_path = form_results()
            sess.run(init)

            writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)

            ''' Get the next test batch of LABELED data '''
            x_l, y_l = mnist.test.next_batch(n_labeled)

            z_out_final = np.array([], dtype=np.float32).reshape(0, 10)
            batch_y_final = np.array([], dtype=np.float32).reshape(0, 1)

            for i in range(n_epochs):
                n_batches = int(n_labeled / batch_size)
                print("------------------Epoch {}/{}------------------".format(i, n_epochs))
                for b in range(1, n_batches + 1):
                    z_real_dist = np.random.randn(batch_size, z_dim) * 5.
                    batch_x, batch_y = mnist.train.next_batch(batch_size)

                    sess.run(autoencoder_optimizer, feed_dict={x_input: batch_x, x_target: batch_x, y_input: batch_y})
                    sess.run(discriminator_optimizer,
                             feed_dict={x_input: batch_x, x_target: batch_x, real_distribution: z_real_dist})
                    sess.run(generator_optimizer, feed_dict={x_input: batch_x, x_target: batch_x})
                    if b % 5 == 0:
                        a_loss, d_loss, g_loss, summary = sess.run(
                            [autoencoder_loss, dc_loss, generator_loss, summary_op],
                            feed_dict={x_input: batch_x, x_target: batch_x,
                                       real_distribution: z_real_dist, y_input: batch_y})


                        writer.add_summary(summary, global_step=step)
                        print("Epoch: {}, iteration: {}".format(i, b))
                        print("Autoencoder Loss: {}".format(a_loss))
                        print("Discriminator Loss: {}".format(d_loss))
                        print("Generator Loss: {}".format(g_loss))
                        with open(log_path + '/log.txt', 'a') as log:
                            log.write("Epoch: {}, iteration: {}\n".format(i, b))
                            log.write("Autoencoder Loss: {}\n".format(a_loss))
                            log.write("Discriminator Loss: {}\n".format(d_loss))
                            log.write("Generator Loss: {}\n".format(g_loss))

                        ''' Check the category to which a random test image is classified to '''
                        ''' Classify random test images into one of the categories by
                            fixing the style of the image to zero.
                        '''

                        # y_pred = sess.run(decoder, feed_dict={x_input: batch_x_l})
                        # print(y_pred[1])
                        # #
                        z_out = sess.run(encoder_output, feed_dict={x_input:batch_x})
                        batch_y_l_val = np.array([ np.where(r==1)[0][0] for r in batch_y ])

                        batch_y_l_val = np.expand_dims(batch_y_l_val, axis =1)

                        z_out_final = np.vstack([z_out_final, z_out])
                        batch_y_final = np.vstack([batch_y_final, batch_y_l_val])

                        if i == 1:
                            plot_latent_2d(np.array(z_out_final), target=batch_y_final)

                            # print(z_out.shape, batch_y_l_val[:2])
                        # plot_latent_2d(z_out, target=batch_y_l_val)


                    step += 1
                acc = 0

            plot_latent_2d(np.array(z_out_final), target=batch_y_final)

                # num_batches = int(mnist.validation.num_examples/batch_size)
                # for j in range(num_batches):
                #     # Classify unseen validation data instead of test data or train data
                #     batch_x_l, batch_y_l = mnist.validation.next_batch(batch_size=batch_size)
                #     encoder_acc = sess.run(accuracy, feed_dict={x_input_l: batch_x_l, y_input: batch_y_l})
                #     acc += encoder_acc
                # acc /= num_batches
                # print("Encoder Classification Accuracy: {}".format(acc))
                # with open(log_path + '/log.txt', 'a') as log:
                #     log.write("Encoder Classification Accuracy: {}".format(acc))
                # saver.save(sess, save_path=saved_model_path, global_step=step)
        else:
            # Get the latest results folder
            all_results = os.listdir(results_path)
            all_results.sort()
            saver.restore(sess, save_path=tf.train.latest_checkpoint(results_path + '/' +
                                                                     all_results[-1] + '/Saved_models/'))
            generate_image_grid(sess, op=decoder_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Autoencoder Train Parameter")
    parser.add_argument('--train', '-t', type=bool, default=True,
                        help='Set to True to train a new model, False to load weights and display image grid')
    args = parser.parse_args()
    train(train_model=args.train)