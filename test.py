# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2

#reader = tf.train.NewCheckpointReader("./checkpoint/CGAN_120/CGAN.model-9")


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
    #flatten=True 以灰度图的形式读
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)
  
  
def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))
    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def encoder_fg(img):
    with tf.variable_scope('encoder_fg'):
        with tf.variable_scope('layer1_fg'):
            weights=tf.get_variable("w1_fg",initializer=tf.constant(reader.get_tensor('encoder_fg/layer1_fg/w1_fg')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b1_fg",initializer=tf.constant(reader.get_tensor('encoder_fg/layer1_fg/b1_fg')))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        
        with tf.variable_scope('layer2_fg'):
            weights=tf.get_variable("w2_fg",initializer=tf.constant(reader.get_tensor('encoder_fg/layer2_fg/w2_fg')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b2_fg",initializer=tf.constant(reader.get_tensor('encoder_fg/layer2_fg/b2_fg')))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        flag1 = tf.concat([conv1,conv2],axis=-1)

        with tf.variable_scope('layer3_fg'):
            weights=tf.get_variable("w3_fg",initializer=tf.constant(reader.get_tensor('encoder_fg/layer3_fg/w3_fg')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b3",initializer=tf.constant(reader.get_tensor('encoder_fg/layer3_fg/b3')))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        flag2 = tf.concat([flag1,conv3],axis=-1)

        with tf.variable_scope('layer4_fg'):
            weights=tf.get_variable("w4_fg",initializer=tf.constant(reader.get_tensor('encoder_fg/layer4_fg/w4_fg')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b4_fg",initializer=tf.constant(reader.get_tensor('encoder_fg/layer4_fg/b4_fg')))
            conv4= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag2, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)
        
        flag3 = tf.concat([flag2,conv4],axis=-1)
        return flag3

        '''with tf.variable_scope('layer5_fg'):
            weights=tf.get_variable("w5_fg",initializer=tf.constant(reader.get_tensor('encoder_fg/layer5_fg/w5_fg')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b5_fg",initializer=tf.constant(reader.get_tensor('encoder_fg/layer5_fg/b5_fg')))
            conv5= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag3, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv5 = lrelu(conv5)
    return conv5'''

def encoder_bg(img):
    with tf.variable_scope('encoder_bg'):
        with tf.variable_scope('layer1_bg'):
            weights=tf.get_variable("w1_bg",initializer=tf.constant(reader.get_tensor('encoder_bg/layer1_bg/w1_bg')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b1_bg",initializer=tf.constant(reader.get_tensor('encoder_bg/layer1_bg/b1_bg')))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)
        
        with tf.variable_scope('layer2_bg'):
            weights=tf.get_variable("w2_bg",initializer=tf.constant(reader.get_tensor('encoder_bg/layer2_bg/w2_bg')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b2_bg",initializer=tf.constant(reader.get_tensor('encoder_bg/layer2_bg/b2_bg')))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        flag1 = tf.concat([conv1,conv2],axis=-1)

        with tf.variable_scope('layer3_bg'):
            weights=tf.get_variable("w3_bg",initializer=tf.constant(reader.get_tensor('encoder_bg/layer3_bg/w3_bg')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b3_bg",initializer=tf.constant(reader.get_tensor('encoder_bg/layer3_bg/b3_bg')))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        flag2 = tf.concat([flag1,conv3],axis=-1)

        with tf.variable_scope('layer4_bg'):
            weights=tf.get_variable("w4_bg",initializer=tf.constant(reader.get_tensor('encoder_bg/layer4_bg/w4_bg')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b4_bg",initializer=tf.constant(reader.get_tensor('encoder_bg/layer4_bg/b4_bg')))
            conv4= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag2, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4 = lrelu(conv4)
        
        flag3 = tf.concat([flag2,conv4],axis=-1)
        return flag3

        '''with tf.variable_scope('layer5_bg'):
            weights=tf.get_variable("w5_bg",initializer=tf.constant(reader.get_tensor('encoder_bg/layer5_bg/w5_bg')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("b5_bg",initializer=tf.constant(reader.get_tensor('encoder_bg/layer5_bg/b5_bg')))
            conv5= tf.contrib.layers.batch_norm(tf.nn.conv2d(flag3, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv5 = lrelu(conv5)
    return conv5'''



def decoder(img):
    #Flag1 = tf.concat([ir,vi],axis=-1)
    with tf.variable_scope('decoder'):
        with tf.variable_scope('Layer1'):
            weights=tf.get_variable("W1",initializer=tf.constant(reader.get_tensor('decoder/Layer1/W1')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B1",initializer=tf.constant(reader.get_tensor('decoder/Layer1/B1')))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)

        with tf.variable_scope('Layer2'):
            weights=tf.get_variable("W2",initializer=tf.constant(reader.get_tensor('decoder/Layer2/W2')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B2",initializer=tf.constant(reader.get_tensor('decoder/Layer2/B2')))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        with tf.variable_scope('Layer3'):
            weights=tf.get_variable("W3",initializer=tf.constant(reader.get_tensor('decoder/Layer3/W3')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B3",initializer=tf.constant(reader.get_tensor('decoder/Layer3/B3')))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        with tf.variable_scope('Layer4'):
            weights=tf.get_variable("W4",initializer=tf.constant(reader.get_tensor('decoder/Layer4/W4')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B4",initializer=tf.constant(reader.get_tensor('decoder/Layer4/B4')))
            conv4= tf.nn.conv2d(conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4 = tf.nn.tanh(conv4)
    return conv4




def input_setup(index):
    padding=0
    '''sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir=(imread(data_ir[index])-127.5)/127.5
    input_ir=np.lib.pad(input_ir,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=(imread(data_vi[index])-127.5)/127.5
    input_vi=np.lib.pad(input_vi,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)
    return train_data_ir,train_data_vi'''
    sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir=(imread(data_ir[index])-127.5)/127.5
    input_ir=np.lib.pad(input_ir,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=(imread(data_vi[index])-127.5)/127.5
    input_vi=np.lib.pad(input_vi,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)

    sub_ir_fg_sequence = []
    sub_vi_fg_sequence = []
    input_ir_fg=(imread(data_ir_fg[index])-127.5)/127.5
    input_ir_fg=np.lib.pad(input_ir_fg,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir_fg.shape
    input_ir_fg=input_ir_fg.reshape([w,h,1])
    input_vi_fg=(imread(data_vi_fg[index])-127.5)/127.5
    input_vi_fg=np.lib.pad(input_vi_fg,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi_fg.shape
    input_vi_fg=input_vi_fg.reshape([w,h,1])
    sub_ir_fg_sequence.append(input_ir_fg)
    sub_vi_fg_sequence.append(input_vi_fg)
    train_data_ir_fg= np.asarray(sub_ir_fg_sequence)
    train_data_vi_fg= np.asarray(sub_vi_fg_sequence)

    sub_ir_bg_sequence = []
    sub_vi_bg_sequence = []
    input_ir_bg=(imread(data_ir_bg[index])-127.5)/127.5
    input_ir_bg=np.lib.pad(input_ir_bg,((padding,padding),(padding,padding)),'edge')
    w,h=input_ir_bg.shape
    input_ir_bg=input_ir_bg.reshape([w,h,1])
    input_vi_bg=(imread(data_vi_bg[index])-127.5)/127.5
    input_vi_bg=np.lib.pad(input_vi_bg,((padding,padding),(padding,padding)),'edge')
    w,h=input_vi_bg.shape
    input_vi_bg=input_vi_bg.reshape([w,h,1])
    sub_ir_bg_sequence.append(input_ir_bg)
    sub_vi_bg_sequence.append(input_vi_bg)
    train_data_ir_bg= np.asarray(sub_ir_bg_sequence)
    train_data_vi_bg= np.asarray(sub_vi_bg_sequence)

    return train_data_ir,train_data_vi,train_data_ir_fg,train_data_vi_fg,train_data_ir_bg,train_data_vi_bg

'''for idx_num in range(15):
  num_epoch=0
  while(num_epoch==idx_num):'''
num_epoch=0
while(num_epoch<20):
  
      reader = tf.train.NewCheckpointReader('./checkpoint/CGAN_120/CGAN.model-'+ str(num_epoch))
  
      with tf.name_scope('IR_input'):
          #红外图像patch
          images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir')
          images_ir_fg = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir_fg')
          images_ir_bg = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir_bg')
      with tf.name_scope('VI_input'):
          #可见光图像patch
          images_vi = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi')
          images_vi_fg = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi_fg')
          images_vi_bg = tf.placeholder(tf.float32, [1,None,None,None], name='images_vi_bg')
          #self.labels_vi_gradient=gradient(self.labels_vi)
      #将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
      with tf.name_scope('input'):
          #resize_ir=tf.image.resize_images(images_ir, (512, 512), method=2)
          input_image_fg =tf.concat([images_ir_fg,images_ir_fg,images_vi_bg],axis=-1)
          input_image_bg =tf.concat([images_vi_bg,images_vi_bg,images_ir_bg],axis=-1)

  
      with tf.name_scope('fusion'):
          #fusion_image=fusion_model(input_image_ir,input_image_vi)
          encoder_image_fg=encoder_fg(input_image_fg)
          encoder_image_bg=encoder_bg(input_image_bg)
          fusion_image_fff=tf.concat([encoder_image_fg,encoder_image_bg],axis=-1)
          fusion_image=decoder(fusion_image_fff)
  
      with tf.Session() as sess:
          init_op=tf.global_variables_initializer()
          sess.run(init_op)
          data_ir=prepare_data('Test_ir')
          data_vi=prepare_data('Test_vi')
          data_ir_fg=prepare_data('Test_ir_fg')
          data_vi_fg=prepare_data('Test_vi_fg')
          data_ir_bg=prepare_data('Test_ir_bg')
          data_vi_bg=prepare_data('Test_vi_bg')
          for i in range(len(data_ir)):
              start=time.time()
              train_data_ir,train_data_vi,train_data_ir_fg,train_data_vi_fg,train_data_ir_bg,train_data_vi_bg=input_setup(i)
              result =sess.run(fusion_image,feed_dict={images_ir: train_data_ir,images_vi: train_data_vi,images_ir_fg: train_data_ir_fg,images_vi_fg: train_data_vi_fg,images_ir_bg: train_data_ir_bg,images_vi_bg: train_data_vi_bg})
              result=result*127.5+127.5
              result = result.squeeze()
              image_path = os.path.join(os.getcwd(), 'result1 0.8 3 6_0.4 0.7 1 1 0.5_Dvi','epoch'+str(num_epoch))
              if not os.path.exists(image_path):
                  os.makedirs(image_path)
              if i<=9:
                  #image_path = os.path.join(image_path,'F9_0'+str(i)+".bmp")
                  image_path = os.path.join(image_path,str(i+1)+".bmp")
              else:
                  #image_path = os.path.join(image_path,'F9_'+str(i)+".bmp")
                  image_path = os.path.join(image_path,str(i+1)+".bmp")
              end=time.time()
              # print(out.shape)
              imsave(result, image_path)
              print("Testing [%d] success,Testing time is [%f]"%(i,end-start))
      tf.reset_default_graph()
      num_epoch=num_epoch+1
