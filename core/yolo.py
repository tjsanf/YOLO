import core.VGG as vgg
import core.resnet_v1 as resnet_v1
import tensorflow.contrib.slim as slim
import tensorflow as tf

class Network:
    def __init__(self, input_data, outpu_data, learning_rate, backbone, is_training):
        '''
        make backbone model
        input (input_data, output_data, learning_rate, backone, is_training)
        output backbone model
        '''
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
                _, end_points = resnet_v1.resnet_v1_50(self.input_data, is_training = self.is_training)
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

class Loss:
    def __init__(self, prediction, groundtruth, batch_size, lambda_object, lambda_nonobject):
        self.prediction = prediction
        self.groundtruth = groundtruth
        self.batch_size = batch_size
        self.lambda_object = lambda_object
        self.lambda_nonobject = lambda_nonobject
    
    def yolo_v1(self):
        obj_mas = tf.expand_dims(self.groundtruth[..., 4], axis = -1)
        nob_mas = 1 - obj_mas

        obj_pre = obj_mas * self.prediction
        nob_pre = nob_mas * self.prediction

        cal_xy = (obj_pre[..., :2] - self.groundtruth[..., :2])**2 + (obj_pre[..., 5:7] - self.groundtruth[..., 5:7])**2
        cal_wh = (tf.square(obj_pre[..., 2:4] + 1e-14) - tf.square(self.groundtruth[..., 2:4] + 1e-14))**2 + (tf.square(obj_pre[..., 7:9] + 1e-14) - tf.square(self.groundtruth[..., 7:9] + 1e-14))**2
        cal_c = (obj_pre[..., 4] - self.groundtruth[..., 4])**2 + self.lambda_nonobject * nob_pre[..., 4]**2 + (obj_pre[..., 9] - self.groundtruth[..., 9])**2 + self.lambda_nonobject * nob_pre[..., 9]**2
        cal_p = (obj_pre[..., 10:] - self.groundtruth[..., 10:])**2
        
        xy_loss = self.lambda_object * tf.reduce_sum(cal_xy) / self.batch_size
        wh_loss = self.lambda_object * tf.reduce_sum(cal_wh) / self.batch_size
        c_loss = tf.reduce_sum(cal_c) / self.batch_size
        p_loss = tf.reduce_sum(cal_p) / self.batch_size

        return xy_loss, wh_loss, c_loss, p_loss

    def yolo_v2(self):
        obj_mas = tf.expand_dims(self.groundtruth[..., 4], axis = -1)
        nob_mas = 1 - obj_mas

        obj_pre = obj_mas * self.prediction
        nob_pre = nob_mas * self.prediction

        cal_xy = (obj_pre[..., :2] - self.groundtruth[..., :2])**2
        cal_wh = (tf.square(obj_pre[..., 2:4] + 1e-14) - tf.square(self.groundtruth[..., 2:4] + 1e-14))**2
        cal_c = (obj_pre[..., 4] - self.groundtruth[..., 4])**2 + self.lambda_nonobject * nob_pre[..., 4]**2
        cal_p = (obj_pre[..., 5:] - self.groundtruth[..., 5:])**2
        
        xy_loss = self.lambda_object * tf.reduce_sum(cal_xy) / self.batch_size
        wh_loss = self.lambda_object * tf.reduce_sum(cal_wh) / self.batch_size
        c_loss = tf.reduce_sum(cal_c) / self.batch_size
        p_loss = tf.reduce_sum(cal_p) / self.batch_size

        return xy_loss, wh_loss, c_loss, p_loss

    def decay(self, variables):
        regularizer = tf.add_n([ tf.nn.l2_loss(v) for v in variables]) * 0.0001

        return regularizer