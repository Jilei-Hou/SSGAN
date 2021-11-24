# -*- coding: utf-8 -*-
from utils import (
  read_data, 
  input_setup, 
  imsave,
  merge,
  gradient,
  lrelu,
  weights_spectral_norm,
  l2_norm
)

import time
import os
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf
from LOSS import SSIM_LOSS

class CGAN(object):

  def __init__(self, 
               sess, 
               image_size=132,
               label_size=120,
               batch_size=32,
               c_dim=1, 
               checkpoint_dir=None, 
               sample_dir=None):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_size = image_size
    self.label_size = label_size
    self.batch_size = batch_size

    self.c_dim = c_dim

    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.build_model()

  def build_model(self):
    with tf.name_scope('IR_input'):
        #红外图像patch
        self.images_ir = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ir')
        self.labels_ir = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_ir')
        self.images_ir_fg = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ir_fg')
        self.labels_ir_fg = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_ir_fg')
        self.images_ir_bg = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_ir_bg')
        self.labels_ir_bg = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_ir_bg')
    with tf.name_scope('VI_input'):
        #可见光图像patch
        self.images_vi = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vi')
        self.labels_vi = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vi')
        self.images_vi_fg = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vi_fg')
        self.labels_vi_fg = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vi_fg')
        self.images_vi_bg = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_vi_bg')
        self.labels_vi_bg = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_vi_bg')
        #self.labels_vi_gradient=gradient(self.labels_vi)
    #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
    with tf.name_scope('input'):
        #self.resize_ir=tf.image.resize_images(self.images_ir, (self.image_size, self.image_size), method=2)
        self.input_image_fg =tf.concat([self.labels_ir_fg,self.labels_ir_fg,self.labels_vi_fg],axis=-1)
        self.input_image_bg =tf.concat([self.labels_vi_bg,self.labels_vi_bg,self.labels_ir_bg],axis=-1)
    #self.pred=tf.clip_by_value(tf.sign(self.pred_ir-self.pred_vi),0,1)


    #融合图像
    with tf.name_scope('fusion'): 
        self.encoder_image_fg=self.encoder_fg(self.input_image_fg)
        self.encoder_image_bg=self.encoder_bg(self.input_image_bg)
        self.fusion_image_fff=tf.concat([self.encoder_image_fg,self.encoder_image_bg],axis=-1)
        self.fusion_image=self.decoder(self.fusion_image_fff)
        #self.fusion_image=self.fusion_model(self.input_image_ir,self.input_image_vi)
    with tf.name_scope('D_input'):
        #self.resize_ir=tf.image.resize_images(self.images_ir, (self.image_size, self.image_size), method=2)
        self.images_D = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.c_dim], name='images_D')
        self.labels_D = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels_D')
        self.images_D1 = self.labels_D

    with tf.name_scope('d_loss'):
        #判决器对可见光图像和融合图像的预测
        #pos=self.discriminator(self.labels_vi,reuse=False)
        pos=self.discriminator(self.fusion_image,reuse=False)
        neg=self.discriminator(self.images_D1,reuse=True,update_collection='NO_OPS')
        #把真实样本尽量判成1否则有损失（判决器的损失）
        pos_loss = tf.reduce_mean(tf.square(pos-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2)))
        #把生成样本尽量判断成0否则有损失（判决器的损失）
        neg_loss = tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0,maxval=0.3,dtype=tf.float32)))
        self.d_loss=neg_loss+pos_loss
        tf.summary.scalar('loss_d',self.d_loss)

    with tf.name_scope('g_loss'):
        #self.g_loss_1=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=neg, labels=tf.ones_like(neg)))
        #self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.ones_like(pos)))
        #self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        #tf.summary.scalar('g_loss_1',self.g_loss_1)
        #self.g_loss_2=tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))
        #self.g_loss_2=tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))+0.3*tf.reduce_mean(tf.square(self.fusion_image - self.labels_vi))+5*tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_vi)))+4*tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_ir)))

        #self.g_loss_total=100*self.g_loss_2
        #tf.summary.scalar('loss_g',self.g_loss_total)
    #self.saver = tf.train.Saver(max_to_keep=50)

        '''self.g_loss_bg=1*tf.reduce_mean(tf.square(self.encoder_image_bg - self.labels_ir_bg))+10*tf.reduce_mean(tf.square(self.encoder_image_bg - self.labels_vi_bg))+5*tf.reduce_mean(tf.square(gradient(self.encoder_image_bg) -gradient (self.labels_ir_bg)))+50*tf.reduce_mean(tf.square(gradient(self.encoder_image_bg) -gradient(self.labels_vi_bg)))
        tf.summary.scalar('g_loss_bg',self.g_loss_bg)

        self.g_loss_total_bg=100*self.g_loss_bg
        tf.summary.scalar('loss_g_bg',self.g_loss_total_bg)
        self.saver = tf.train.Saver(max_to_keep=50)

        self.g_loss_fg=1*tf.reduce_mean(tf.square(self.encoder_image_fg - self.labels_ir_fg))+0.3*tf.reduce_mean(tf.square(self.encoder_image_fg - self.labels_vi_fg))+5*tf.reduce_mean(tf.square(gradient(self.encoder_image_fg) -gradient (self.labels_ir_fg)))+4*tf.reduce_mean(tf.square(gradient(self.encoder_image_fg) -gradient(self.labels_vi_fg)))
        tf.summary.scalar('g_loss_fg',self.g_loss_fg)

        self.g_loss_total_fg=100*self.g_loss_fg
        tf.summary.scalar('loss_g_fg',self.g_loss_total_fg)
        self.saver = tf.train.Saver(max_to_keep=50)'''

        self.g_loss_1=tf.reduce_mean(tf.square(neg-tf.random_uniform(shape=[self.batch_size,1],minval=0.7,maxval=1.2,dtype=tf.float32)))
        tf.summary.scalar('g_loss_1',self.g_loss_1)

        self.g_loss_2=2*tf.reduce_mean(tf.square(self.fusion_image - self.labels_ir))+0.5*tf.reduce_mean(tf.square(self.fusion_image - self.labels_vi))+3*tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_ir)))+7*tf.reduce_mean(tf.square(gradient(self.fusion_image) -gradient (self.labels_vi)))

        ''' SSIM loss'''
        SSIM1 = 1-SSIM_LOSS(self.labels_ir, self.fusion_image)
        SSIM2 = 1-SSIM_LOSS(self.labels_vi, self.fusion_image)
        self.ssim_loss = tf.reduce_mean(0.5*SSIM1 + 0.4*SSIM2)
        tf.summary.scalar('loss_g_ssim',self.ssim_loss)

        self.g_loss_decoder=2*self.g_loss_1+1*self.g_loss_2+2*self.ssim_loss

        tf.summary.scalar('g_loss_decoder',self.g_loss_decoder)

        self.g_loss_total_decoder=self.g_loss_decoder
        tf.summary.scalar('loss_g_decoder',self.g_loss_total_decoder)

    self.saver = tf.train.Saver(max_to_keep=50)



    with tf.name_scope('image'):
        #tf.summary.image('input_ir',tf.expand_dims(self.images_ir[1,:,:,:],0))  
        #tf.summary.image('input_vi',tf.expand_dims(self.images_vi[1,:,:,:],0))  
        #tf.summary.image('fusion_image',tf.expand_dims(self.fusion_image[1,:,:,:],0))   
        tf.summary.image('input_ir',tf.expand_dims(self.images_ir[1,:,:,:],0))  
        tf.summary.image('input_vi',tf.expand_dims(self.images_vi[1,:,:,:],0)) 
        tf.summary.image('images_D',tf.expand_dims(self.images_D1[1,:,:,:],0)) 
        tf.summary.image('input_ir_fg',tf.expand_dims(self.images_ir_fg[1,:,:,:],0))  
        tf.summary.image('input_vi_fg',tf.expand_dims(self.images_vi_fg[1,:,:,:],0)) 
        tf.summary.image('input_ir_bg',tf.expand_dims(self.images_ir_bg[1,:,:,:],0))  
        tf.summary.image('input_vi_bg',tf.expand_dims(self.images_vi_bg[1,:,:,:],0)) 
        tf.summary.image('fusion_image',tf.expand_dims(self.fusion_image[1,:,:,:],0))   
    
  def train(self, config):
    if config.is_train:
      #input_setup(self.sess, config,"Train_ir")
      #input_setup(self.sess,config,"Train_vi")
      input_setup(self.sess, config,"Train_ir")
      input_setup(self.sess, config,"Train_vi")
      input_setup(self.sess, config,"Train_ir_fg")
      input_setup(self.sess, config,"Train_vi_fg")
      input_setup(self.sess, config,"Train_ir_bg")
      input_setup(self.sess, config,"Train_vi_bg")
      input_setup(self.sess, config,"Train_D")
    else:
      #nx_ir, ny_ir = input_setup(self.sess, config,"Test_ir")
      #nx_vi, ny_vi = input_setup(self.sess, config,"Test_vi")
      nx_ir, ny_ir = input_setup(self.sess, config,"Test_ir")
      nx_vi, ny_vi = input_setup(self.sess, config,"Test_vi")
      nx_ir, ny_ir = input_setup(self.sess, config,"Test_ir_fg")
      nx_vi, ny_vi = input_setup(self.sess, config,"Test_vi_fg")
      nx_ir, ny_ir = input_setup(self.sess, config,"Test_ir_bg")
      nx_vi, ny_vi = input_setup(self.sess, config,"Test_vi_bg")

    if config.is_train:     
      #data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir","train.h5")
      #data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi","train.h5")
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir","train.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi","train.h5")  
      data_dir_ir_fg = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir_fg","train.h5")
      data_dir_ir_bg = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi_fg","train.h5")
      data_dir_vi_fg = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir_bg","train.h5")
      data_dir_vi_bg = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi_bg","train.h5")
      data_dir_D = os.path.join('./{}'.format(config.checkpoint_dir), "Train_D","train.h5")
    else:
      #data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir),"Test_ir", "test.h5")
      #data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir),"Test_vi", "test.h5")
      data_dir_ir = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir","train.h5")
      data_dir_vi = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi","train.h5")  
      data_dir_ir_fg = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir_fg","train.h5")
      data_dir_ir_bg = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi_fg","train.h5")
      data_dir_vi_fg = os.path.join('./{}'.format(config.checkpoint_dir), "Train_ir_bg","train.h5")
      data_dir_vi_bg = os.path.join('./{}'.format(config.checkpoint_dir), "Train_vi_bg","train.h5")

    #train_data_ir, train_label_ir = read_data(data_dir_ir)
    #train_data_vi, train_label_vi = read_data(data_dir_vi)
    train_data_ir, train_label_ir = read_data(data_dir_ir)
    train_data_vi, train_label_vi = read_data(data_dir_vi)
    train_data_ir_fg, train_label_ir_fg = read_data(data_dir_ir_fg)
    train_data_vi_fg, train_label_vi_fg = read_data(data_dir_vi_fg)
    train_data_ir_bg, train_label_ir_bg = read_data(data_dir_ir_bg)
    train_data_vi_bg, train_label_vi_bg = read_data(data_dir_vi_bg)
    train_data_D, train_label_D = read_data(data_dir_D)
    #找训练时更新的变量组（判决器和生成器是分开训练的，所以要找到对应的变量）
    t_vars = tf.trainable_variables()
    #self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    #print(self.d_vars)
    #self.g_vars = [var for var in t_vars if 'fusion_model' in var.name]
    #print(self.g_vars)
    self.decoder_vars = [var for var in t_vars if 'decoder' in var.name]
    self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
    print(self.decoder_vars)
    # clip_ops = []
    # for var in self.d_vars:
        # clip_bounds = [-.01, .01]
        # clip_ops.append(
            # tf.assign(
                # var, 
                # tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            # )
        # )
    # self.clip_disc_weights = tf.group(*clip_ops)
    # Stochastic gradient descent with the standard backpropagation
    with tf.name_scope('train_step'):
        #self.train_fusion_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total,var_list=self.g_vars)
        #self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)
        self.train_decoder_op = tf.train.AdamOptimizer(config.learning_rate).minimize(self.g_loss_total_decoder,var_list=self.decoder_vars)
        self.train_discriminator_op=tf.train.AdamOptimizer(config.learning_rate).minimize(self.d_loss,var_list=self.d_vars)
    #将所有统计的量合起来
    self.summary_op = tf.summary.merge_all()
    #生成日志文件
    self.train_writer = tf.summary.FileWriter(config.summary_dir + '/train',self.sess.graph,flush_secs=60)
    
    tf.initialize_all_variables().run()
    
    counter = 0
    start_time = time.time()

    # if self.load(self.checkpoint_dir):
      # print(" [*] Load SUCCESS")
    # else:
      # print(" [!] Load failed...")

    if config.is_train:
      print("Training...")

      for ep in range(config.epoch):
        # Run by batch images
        batch_idxs = len(train_data_ir) // config.batch_size
        for idx in range(0, batch_idxs):
          #batch_images_ir = train_data_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          #batch_labels_ir = train_label_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          #batch_images_vi = train_data_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          #batch_labels_vi = train_label_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_ir = train_data_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_ir = train_label_ir[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_vi = train_data_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_vi = train_label_vi[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_ir_fg = train_data_ir_fg[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_ir_fg = train_label_ir_fg[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_vi_fg = train_data_vi_fg[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_vi_fg = train_label_vi_fg[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_ir_bg = train_data_ir_bg[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_ir_bg = train_label_ir_bg[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_vi_bg = train_data_vi_bg[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_vi_bg = train_label_vi_bg[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_images_D = train_data_D[idx*config.batch_size : (idx+1)*config.batch_size]
          batch_labels_D = train_label_D[idx*config.batch_size : (idx+1)*config.batch_size]

          counter += 1
          #for i in range(2):
           # _, err_d= self.sess.run([self.train_discriminator_op, self.d_loss], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, self.labels_vi: batch_labels_vi,self.labels_ir:batch_labels_ir})
            # self.sess.run(self.clip_disc_weights)
          _, err_d= self.sess.run([self.train_discriminator_op, self.d_loss], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi,
self.labels_ir: batch_labels_ir,self.labels_vi:batch_labels_vi, self.images_ir_fg: batch_images_ir_fg, self.images_vi_fg: batch_images_vi_fg, self.labels_ir_fg: batch_labels_ir_fg,self.labels_vi_fg:batch_labels_vi_fg,self.images_ir_bg: batch_images_ir_bg, self.images_vi_bg: batch_images_vi_bg, self.labels_ir_bg: batch_labels_ir_bg,self.labels_vi_bg:batch_labels_vi_bg,self.images_D1: batch_images_D,self.labels_D: batch_labels_D})
          _, err_g,summary_str_decoder= self.sess.run([self.train_decoder_op, self.g_loss_total_decoder,self.summary_op], feed_dict={self.images_ir: batch_images_ir, self.images_vi: batch_images_vi, self.labels_ir: batch_labels_ir,self.labels_vi:batch_labels_vi,self.images_ir_fg: batch_images_ir_fg, self.images_vi_fg: batch_images_vi_fg, self.labels_ir_fg: batch_labels_ir_fg,self.labels_vi_fg:batch_labels_vi_fg,self.images_ir_bg: batch_images_ir_bg, self.images_vi_bg: batch_images_vi_bg, self.labels_ir_bg: batch_labels_ir_bg,self.labels_vi_bg:batch_labels_vi_bg,self.images_D1: batch_images_D,self.labels_D: batch_labels_D})
          #将统计的量写到日志文件里
          self.train_writer.add_summary(summary_str_decoder,counter)

          if counter % 10 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_decoder:[%.8f], loss_d:[%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err_g, err_d))
            #print(a)

        self.save(config.checkpoint_dir, ep)

    else:
      print("Testing...")

      result = self.fusion_image.eval(feed_dict={self.images_ir: train_data_ir, self.labels_ir: train_label_ir,self.images_vi: train_data_vi, self.labels_vi: train_label_vi})
      result=result*127.5+127.5
      result = merge(result, [nx_ir, ny_ir])
      result = result.squeeze()
      image_path = os.path.join(os.getcwd(), config.sample_dir)
      image_path = os.path.join(image_path, "test_image.png")
      imsave(result, image_path)

  def encoder_fg(self,img):
    with tf.variable_scope('encoder_fg'):
        with tf.variable_scope('layer1_fg'):
            weights=tf.get_variable("w1_fg",[3,3,3,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b1_fg",[16],initializer=tf.constant_initializer(0.0))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        
        with tf.variable_scope('layer2_fg'):
            weights=tf.get_variable("w2_fg",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b2_fg",[16],initializer=tf.constant_initializer(0.0))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        flag1 = tf.concat([conv1,conv2],axis=-1)

        with tf.variable_scope('layer3_fg'):
            weights=tf.get_variable("w3_fg",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b3",[16],initializer=tf.constant_initializer(0.0))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        flag2 = tf.concat([flag1,conv3],axis=-1)

        with tf.variable_scope('layer4_fg'):
            weights=tf.get_variable("w4_fg",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b4_fg",[16],initializer=tf.constant_initializer(0.0))
            conv4= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag2, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)
        
        flag3 = tf.concat([flag2,conv4],axis=-1)
        return flag3

        '''with tf.variable_scope('layer5_fg'):
            weights=tf.get_variable("w5_fg",[1,1,64,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b5_fg",[1],initializer=tf.constant_initializer(0.0))
            conv5= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag3, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv5 = lrelu(conv5)
        
    return conv5'''

  def encoder_bg(self,img):
    with tf.variable_scope('encoder_bg'):
        with tf.variable_scope('layer1_bg'):
            weights=tf.get_variable("w1_bg",[3,3,3,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b1_bg",[16],initializer=tf.constant_initializer(0.0))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        
        with tf.variable_scope('layer2_bg'):
            weights=tf.get_variable("w2_bg",[3,3,16,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b2_bg",[16],initializer=tf.constant_initializer(0.0))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        flag1 = tf.concat([conv1,conv2],axis=-1)

        with tf.variable_scope('layer3_bg'):
            weights=tf.get_variable("w3_bg",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b3_bg",[16],initializer=tf.constant_initializer(0.0))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        flag2 = tf.concat([flag1,conv3],axis=-1)

        with tf.variable_scope('layer4_bg'):
            weights=tf.get_variable("w4_bg",[3,3,48,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b4_bg",[16],initializer=tf.constant_initializer(0.0))
            conv4= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag2, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)
        
        flag3 = tf.concat([flag2,conv4],axis=-1)
        return flag3

        '''with tf.variable_scope('layer5_bg'):
            weights=tf.get_variable("w5_bg",[1,1,64,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b5_bg",[1],initializer=tf.constant_initializer(0.0))
            conv5= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag3, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv5 = lrelu(conv5)
        
    return conv5'''

  def decoder(self,img):
    #Flag1 = tf.concat([ir,vi],axis=-1)
    with tf.variable_scope('decoder'):
        with tf.variable_scope('Layer1'):
            weights=tf.get_variable("W1",[3,3,128,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B1",[64],initializer=tf.constant_initializer(0.0))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)

        with tf.variable_scope('Layer2'):
            weights=tf.get_variable("W2",[3,3,64,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B2",[32],initializer=tf.constant_initializer(0.0))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        with tf.variable_scope('Layer3'):
            weights=tf.get_variable("W3",[3,3,32,16],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B3",[16],initializer=tf.constant_initializer(0.0))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        with tf.variable_scope('Layer4'):
            weights=tf.get_variable("W4",[1,1,16,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B4",[1],initializer=tf.constant_initializer(0.0))
            conv4= tf.nn.conv2d(conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4 = tf.nn.tanh(conv4)
    return conv4

  def discriminator(self,img,reuse,update_collection=None):
    with tf.variable_scope('discriminator',reuse=reuse):
        #print(img.shape)
        with tf.variable_scope('Layer_1'):
            weights=tf.get_variable("W_1",[3,3,1,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("B_1",[32],initializer=tf.constant_initializer(0.0))
            conv1=tf.nn.conv2d(img, weights, strides=[1,2,2,1], padding='VALID') + bias
            conv1 = lrelu(conv1)
            #print(conv1.shape)
        with tf.variable_scope('Layer_2'):
            weights=tf.get_variable("W_2",[3,3,32,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("B_2",[64],initializer=tf.constant_initializer(0.0))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)
            #print(conv2_vi.shape)
        with tf.variable_scope('Layer_3'):
            weights=tf.get_variable("W_3",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("B_3",[128],initializer=tf.constant_initializer(0.0))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3=lrelu(conv3)
            #print(conv3_vi.shape)
        with tf.variable_scope('Layer_4'):
            weights=tf.get_variable("W_4",[3,3,128,256],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("B_4",[256],initializer=tf.constant_initializer(0.0))
            conv4= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3, weights, strides=[1,2,2,1], padding='VALID') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4=lrelu(conv4)
            conv4 = tf.reshape(conv4,[self.batch_size,6*6*256])
        with tf.variable_scope('Line_5'):
            weights=tf.get_variable("W_5",[6*6*256,1],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=weights_spectral_norm(weights,update_collection=update_collection)
            bias=tf.get_variable("B_5",[1],initializer=tf.constant_initializer(0.0))
            line_5=tf.matmul(conv4, weights) + bias
            #conv3_vi= tf.contrib.layers.batch_norm(conv3_vi, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
    return line_5


  def save(self, checkpoint_dir, step):
    model_name = "CGAN.model"
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("CGAN", self.label_size)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        print(ckpt_name)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir,ckpt_name))
        return True
    else:
        return False
