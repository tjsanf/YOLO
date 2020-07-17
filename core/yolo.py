import core.VGG as vgg
import core.resnet_v1
import tensorflow.contrib.slim as slim
import tensorflow as tf

class network:
    def __init__(self, input_data, outpu_data, learning_rate, backbone, is_training):
        '''
        make backbone model
        input (input_data, output_data, learning_rate, backone, is_training)
        output backbone model
        '''
        R_MEAN = 123.68
        G_MEAN = 116.78
        B_MEAN = 103.94
        self.MEAN = [R_MEAN, G_MEAN, B_MEAN]
        self.input_data = input_data
        self.outpu_data = outpu_data
        self.learning_rate = learning_rate
        self.backbone = backbone
        self.is_training = is_training
        if self.backbone == 'vgg_16':
            with slim.arg_scope(vgg.vgg_arg_scope()):
                self.net = vgg.vgg_16(self.input_data, scope='vgg_16')
        elif self.backbone == 'resnet_v1_50':
            with tf.contrib.slim.arg_scope(resnet_v1.resnet_arg_scope()):
                logits, end_points = resnet_v1.resnet_v1_50(x_image, is_training = self.is_training)
            self.net = end_points['resnet_v1_50/block4']
        else:
            print('check your backbone name')
            raise EOFError

    def yolo_v1(self):
        '''
        make yolo v1 model
        input [batch, 448, 448, 3]
        output [batch, 7, 7, 30]
        '''
        b, w, h, g = self.output_data.shape
        if self.backbone == 'vgg_16':
            self.net = tf.nn.conv2d(self.net, 1024, (3,3), strides = (1,1), padding='same', activation=tf.nn.leaky_relu)
            self.net = tf.nn.conv2d(self.net, 1024, (3,3), strides = (2,2), padding='same', activation=tf.nn.leaky_relu)
            self.net = tf.nn.conv2d(self.net, 1024, (3,3), strides = (1,1), padding='same', activation=tf.nn.leaky_relu)
            self.net = tf.nn.conv2d(self.net, 1024, (3,3), strides = (1,1), padding='same', activation=tf.nn.leaky_relu)
            self.net = tf.layers.dense(self.net, 4096, activation=tf.nn.leaky_relu)
            self.net = tf.layers.dense(self.net, w*h*g, activation=tf.nn.leaky_relu)
        elif self.backbone == 'resnet_v1_50':
            self.net = tf.nn.batch_normalization(tf.nn.conv2d(self.net, 1024, (3,3), strides = (1,1), padding='same', activation=tf.nn.leaky_relu), training=self.is_training)
            self.net = tf.nn.batch_normalization(tf.nn.conv2d(self.net, 1024, (3,3), strides = (2,2), padding='same', activation=tf.nn.leaky_relu), training=self.is_training)
            self.net = tf.nn.batch_normalization(tf.nn.conv2d(self.net, 1024, (3,3), strides = (1,1), padding='same', activation=tf.nn.leaky_relu), training=self.is_training)
            self.net = tf.nn.batch_normalization(tf.nn.conv2d(self.net, 1024, (3,3), strides = (1,1), padding='same', activation=tf.nn.leaky_relu), training=self.is_training)
            self.net = tf.layers.dense(self.net, 4096, activation=tf.nn.leaky_relu)
            self.net = tf.layers.dense(self.net, w*h*g, activation=tf.nn.leaky_relu)

        self.output_data = tf.reshape(self.net, [b, w, h, g])

        return self.output_data

    def yolo_v2(self):
        '''
        make yolo v2 model
        input [batch, 416, 416, 3]
        output [batch, 13, 13, 5, 25]
        '''
        b, w, h, x, g = self.output_data.shape
        if self.backbone == 'vgg_16':
            self.net = tf.layers.conv2d(self.net, 1024, (3,3), strides = (1,1), padding='same', activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self.net = tf.layers.conv2d(self.net, x*g, (3,3), strides = (1,1), padding='same', activation=tf.nn.leaky_relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
        elif self.backbone == 'resnet_v1_50':
            self.net = tf.layers.batch_normalization(tf.layers.conv2d(self.net, 1024, (3,3), strides = (1,1), padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer()), training=self.is_training)
            self.net = tf.layers.batch_normalization(tf.layers.conv2d(self.net, x*g, (3,3), strides = (1,1), padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer()), training=self.is_training)

        self.output_data = tf.reshape(self.net, [b, w, h, x, g])

        return self.output_data
