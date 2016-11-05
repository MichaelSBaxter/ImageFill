import os
import random
import math
import time
import cPickle
import numpy as np
import tensorflow as tf
from PIL import Image

batch_size = 50
learning_rate = 0.001
restore_checkpoint = False
save_checkpoint = True
run_training = True
example_step = 10000
save_step = 10000
network_print_step = 10000
cifar10_batch_size = 10000 # 10000

dim1 = 32
dim2 = 32
dim3 = 3

def mask_gen(size):
    mask_tensor = np.zeros((size, dim1, dim2), dtype=np.uint8)
    mask_entry = np.ones((dim1, dim2), dtype=np.uint8)

    top = random.randint(0, 17)
    bottom = top + random.randint(4, 15)
    left = random.randint(0, 17)
    right = left + random.randint(4, 15)

    for i in xrange(dim1):
        for j in xrange(dim2):
            if i > top and i < bottom and j > left and j < right:
                mask_entry[i][j] = 0

    for i in xrange(size):
        mask_tensor[i] = mask_entry

    return mask_tensor

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict
    
if __name__ == '__main__':
    sess = tf.Session()

    image_input = tf.placeholder(tf.uint8, [None, dim1, dim2, dim3])
    mask_input = tf.placeholder(tf.uint8, [None, dim1, dim2])
    mask_float = tf.cast(mask_input, tf.float32)

    mask_bool = tf.cast(mask_input, tf.bool)
    mask_inverted_bool = tf.logical_not(mask_bool)
    mask_inverted = tf.cast(mask_inverted_bool, tf.float32)

    image_channels_input = tf.unpack(tf.cast(image_input, tf.float32), axis=3)

    channel1_noise = tf.random_uniform([batch_size, dim1, dim2], 0, 255, dtype=tf.float32)
    channel2_noise = tf.random_uniform([batch_size, dim1, dim2], 0, 255, dtype=tf.float32)
    channel3_noise = tf.random_uniform([batch_size, dim1, dim2], 0, 255, dtype=tf.float32)

    channel1_input = tf.add(tf.mul(image_channels_input[0], mask_float), tf.mul(channel1_noise, mask_inverted))
    channel2_input = tf.add(tf.mul(image_channels_input[1], mask_float), tf.mul(channel2_noise, mask_inverted))
    channel3_input = tf.add(tf.mul(image_channels_input[2], mask_float), tf.mul(channel3_noise, mask_inverted))

    combined_input = tf.pack([channel1_input, channel2_input, channel3_input, mask_float], axis=3)

    conv_input = tf.stop_gradient(combined_input)
    loss_input = tf.stop_gradient(tf.cast(image_input, tf.float32))

    kernel1 = tf.Variable(tf.random_uniform([5, 5, dim3+1, 128], -0.01, 0.01))
    b1 = tf.Variable(tf.random_uniform([128], -0.01, 0.01))

    kernel2 = tf.Variable(tf.random_uniform([5, 5, 128, 128], -0.01, 0.01))
    b2 = tf.Variable(tf.random_uniform([128], -0.01, 0.01))

    kernel3 = tf.Variable(tf.random_uniform([5, 5, 128, 128], -0.01, 0.01))
    b3 = tf.Variable(tf.random_uniform([128], -0.01, 0.01))

    kernel4 = tf.Variable(tf.random_uniform([3, 3, 128, dim3], -0.01, 0.01))
    b4 = tf.Variable(tf.random_uniform([3], -0.01, 0.01))

    #w1 =  tf.Variable(tf.random_uniform([32 * 32 * 32, 1024], -0.01, 0.01))
    #b4 =  tf.Variable(tf.random_uniform([1024], -0.01, 0.01))

    #kernel_trans1 = tf.Variable(tf.random_uniform([5, 5, 128, 128], -0.01, 0.01))
    #b4 = tf.Variable(tf.random_uniform([128], -0.01, 0.01))

    #kernel_trans2 = tf.Variable(tf.random_uniform([5, 5, 3, 128], -0.01, 0.01))
    #b5 = tf.Variable(tf.random_uniform([3], -0.01, 0.01))

    #kernel_trans1 = tf.Variable(tf.random_uniform([5, 5, 64, 3], -0.01, 0.01))
    #b6 = tf.Variable(tf.random_uniform([64], -0.01, 0.01))

    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv_input, kernel1, [1, 1, 1, 1], padding='SAME'), b1))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, kernel2, [1, 1, 1, 1], padding='SAME'), b2))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, kernel3, [1, 1, 1, 1], padding='SAME'), b3))
    conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, kernel4, [1, 1, 1, 1], padding='SAME'), b4))

    #conv_trans1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(conv3, kernel_trans1, [50, 16, 16, 128], [1, 2, 2, 1], padding='SAME'), b4))
    #conv_trans2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(conv_trans1, kernel_trans2, [50, 32, 32, 3], [1, 2, 2, 1], padding='SAME'), b5))
    #conv_trans3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d_transpose(conv_trans2, kernel3, [50, 32, 32, 3], [1, 1, 1, 1], padding='SAME'), b6))

    #conv_flat = tf.rehape(conv3, [-1, 32 * 32 * 32])
    #fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(conv_flat, w1), b4)

    conv_output = conv4

    loss = tf.reduce_mean(tf.square(tf.sub(loss_input, conv_output))) 
    training_summary = tf.scalar_summary("Training Loss", loss)

    #cross_entropy = -tf.reduce_sum(loss_input * tf.log(conv4))
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(conv4, loss_input))
    training = tf.train.AdamOptimizer(learning_rate).minimize(loss)  

    sess.run(tf.initialize_all_variables())    

    summary_writer = tf.train.SummaryWriter('summary', sess.graph)
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
  
    if run_training == True:
        print 'Training...'
        example_num = 0
        checkpoint_num = 0
        tick = 0
        image_batch = np.zeros((batch_size, 32, 32, 3))

        for temp in xrange(0, 20):         
            cifar_batch = (temp % 5) + 1
            images = unpickle('cifar-10-batches-py/data_batch_{}'.format(cifar_batch))

            for i in xrange(cifar10_batch_size): # size of a cifar 10 batch
                tick += 1
                if (temp + 1) % 5 == 0:      
                    image = images['data'][i].reshape((3, 32, 32)).swapaxes(0, 2).swapaxes(0, 1)  
                else:
                    image = images['data'][i].reshape((3, 32, 32)).swapaxes(0, 2)
      
                
                image_batch[i % batch_size] = image         

                if (i + 1) % batch_size == 0:
                    start_time = time.time()

                    mask_batch = mask_gen(batch_size)
                    feed = {image_input: image_batch, mask_input: mask_batch}
                    loss_value, train_summ, _ = sess.run([loss, training_summary, training], feed_dict=feed)
                    summary_writer.add_summary(train_summ, tick)

                    elapsed_time = time.time() - start_time

                    print '[loss = {}, training step time = {}]'.format(loss_value, elapsed_time)

                if (i + 1) % example_step == 0:
                    feed = {image_input: image_batch, mask_input: mask_batch}
                    netin, netout  = sess.run([combined_input, conv_output], feed_dict=feed)

                    input_image = np.array(netin[0,:,:,0:3]).astype(np.uint8)
                    output_image = np.array(netout[0]).astype(np.uint8)

                    img = Image.fromarray(input_image, 'RGB')
                    name = 'training_images/training_{}_input_image.png'.format(example_num)
                    img.save(name)

                    img = Image.fromarray(output_image, 'RGB')
                    name = 'training_images/training_{}_output_image.png'.format(example_num)
                    img.save(name)  
                    
                    example_num += 1

                if save_checkpoint and (i + 1) % save_step == 0:
                    saver.save(sess, "saved_models/image_fill.ckpt")

                    with open('saved_models/image_fill_progress.log', 'w') as logfile:
                        logfile.write('{}, {}'.format(temp, i))

                    print '[Checkpoint {} saved]'.format(checkpoint_num)
                    checkpoint_num += 1

    else:
        print 'Evaluating...'
        saver.restore(sess, "saved_models/image_fill.ckpt")
        image_batch = np.zeros((batch_size, 32, 32, 3))

        images = unpickle('cifar-10-batches-py/test_batch') 
        evaluation_num = 0

        for i in xrange(cifar10_batch_size): # size of a cifar 10 batch
            image = images['data'][i].reshape((3, 32, 32)).swapaxes(0, 2).swapaxes(0, 1)             
            image_batch[i % batch_size] = image         

            if (i + 1) % batch_size == 0:
                mask_batch = mask_gen(batch_size)

                feed = {image_input: image_batch, mask_input: mask_batch}
                netin, netout = sess.run([combined_input, conv_output], feed_dict=feed)

                for j in xrange(batch_size):
                    input_image = np.array(netin[j,:,:,0:3]).astype(np.uint8)
                    output_image = np.array(netout[j]).astype(np.uint8)

                    img = Image.fromarray(input_image, 'RGB')
                    name = 'evaluation_images/evaluation_{}_input_image.png'.format(evaluation_num)
                    img.save(name)

                    img = Image.fromarray(output_image, 'RGB')
                    name = 'evaluation_images/evaluation_{}_output_image.png'.format(evaluation_num)
                    img.save(name)  
                
                    evaluation_num += 1

    sess.close() 
