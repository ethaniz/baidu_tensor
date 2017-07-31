# -*- coding:utf-8 -*-

import pymongo
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf
import os

# 只识别数字和空格
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
PADING = [' ']
text_set = number+PADING

# 最多识别10位
captcha_size = 10
captcha_len = len(text_set)

# 经过截图和zoom以后，分辨率为146*58
width, height = 146, 58

# 原始文本one-hot编码
def text2vec(text):
    text_len = len(text)
    if text_len < 10:
        text = ' '*(10-text_len) + text
        #print(text)
        #print(len(text))
    vector = np.zeros(10 * 11)

    def char2pos(c):
        if c == ' ':
            k = 10
            return k
        k = ord(c)-48
        if k > 9:
            k = ord(c)-55
            if k > 35:
                k = ord(c) - 61
                if k > 61:
                    raise ValueError('No Map '+c)
        return k
    for i, c in enumerate(text):
        idx = i*captcha_len+char2pos(c)
        vector[idx] = 1
    return vector

# 原始文本one-hot解码
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % captcha_len
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx == 10:
            char_code = ord(' ')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)

# 图片灰度化
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img

# batch训练数据生成
def gen_next_batch(batch_size):
    batch_x = np.zeros([batch_size, width*height])
    batch_y = np.zeros([batch_size, 10*11])
    for i in range(batch_size):
        text, image, pic = gen_captcha_text_and_image()
        image = convert2gray(image)
        batch_x[i, :] = image.flatten()/255
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y, text, image


# 随机从mongo中读取训练数据
db = pymongo.MongoClient('localhost', 27017).test
cnt = np.random.randint(1, 1117)
def gen_captcha_text_and_image():
    global cnt
    if cnt == 1117:
        cnt = 1
    f = db.files.find_one({'index':cnt})
    image = Image.open(BytesIO(f['content']))
    img_2 = image.point(lambda x: 255 if x > 125 else 0).convert('RGB')
    img_x = np.array(img_2)
    img_y = f['label']
    cnt += 1
    return img_y, img_x, img_2

# 3层卷积，前向传播
def inference(X, keep_prob):
    x = tf.reshape(X, shape=[-1, height, width, 1])

    # 初始化参数
    w_alpha = 0.01
    b_alpha = 0.1

    # 第一层卷积
    # 输入146*58*1
    # 过滤器尺寸3*3*1，步长为1， 深度为32
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3,3,1,32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    # 使用全0填充，卷积后尺寸为146*58*32
    conv1_a = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1,1,1,1], padding='SAME'), b_c1))
    # 使用最大池化输出，全0填充，尺寸2*2， 步长为2， 池化后尺寸73*29*32
    conv1_b = tf.nn.max_pool(conv1_a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1 = tf.nn.dropout(conv1_b, keep_prob)

    # 第二层卷积
    # 输入73*29*32
    # 过滤器尺寸3*3*32，步长为1， 深度为64
    w_c2 = tf.Variable(w_alpha * tf.random_normal([3,3,32,64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    # 使用全0填充，卷积后尺寸为73*29*64
    conv2_a = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1,1,1,1], padding='SAME'), b_c2))
    # 使用最大池化输出，全0填充，尺寸2*2， 步长为2， 池化后尺寸37*15*64
    conv2_b = tf.nn.max_pool(conv2_a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv2 = tf.nn.dropout(conv2_b, keep_prob)

    # 第三层卷积
    # 输入37*15*64
    # 过滤器尺寸3*3*64， 步长为1， 深度为64
    w_c3 = tf.Variable(w_alpha * tf.random_normal([3,3,64,64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    # 使用全0填充，卷积后尺寸为37*15*64
    conv3_a = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1,1,1,1], padding='SAME'), b_c3))
    # 使用最大池化输出，全0填充，尺寸2*2， 步长为2， 池化后尺寸19*8*64
    conv3_b = tf.nn.max_pool(conv3_a, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv3 = tf.nn.dropout(conv3_b, keep_prob)

    # 全连接层
    # 输入19*8*64
    # 节点选取1024
    w_d = tf.Variable(w_alpha * tf.random_normal([19*8*64, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense1 = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense2 = tf.nn.relu(tf.add(tf.matmul(dense1, w_d), b_d))
    dense = tf.nn.dropout(dense2, keep_prob)

    # 输出10个one-hot编码，即长度为10*11的向量
    w_out = tf.Variable(w_alpha * tf.random_normal([1024, 10*11]))
    b_out = tf.Variable(b_alpha * tf.random_normal([10*11]))
    out = tf.add(tf.matmul(dense, w_out), b_out)

    return out


# 准确率验证
def verify(out, Y):
    # 验证
    predict = tf.reshape(out, [-1, 10, 11])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, 10, 11]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    return accuracy

# 从测试文件夹读取测试用例
def test(test_path):
    import glob
    images = glob.glob(test_path)

    X = tf.placeholder(tf.float32, [None, width*height])
    keep_prob = tf.placeholder(tf.float32)
    out = inference(X, keep_prob)

    predict = tf.reshape(out, [-1, 10, 11])
    max_idx_p = tf.argmax(predict, 2)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        if os.path.exists('/Users/ethan/MyCodes/baidu_tensor/models/checkpoint'):
            print("Found last check point, restoring...")
            saver.restore(sess, '/Users/ethan/MyCodes/baidu_tensor/models/model.ckpt')

            test_len = len(images)
            positive_cnt = 0
            cnt = 0
            for image_path in images:
                true_label = image_path.split('/')[-1].split('.')[0]
                image = Image.open(image_path)
                #image.show()
                img_2 = image.point(lambda x: 255 if x > 125 else 0).convert('RGB')
                img_x = np.array(img_2)
                image = convert2gray(img_x)
                input_x = np.zeros([1, width*height])
                input_x[0, :] = image.flatten()/255

                text_list = sess.run(max_idx_p, feed_dict={X: input_x, keep_prob: 1.0})
                text = text_list[0].tolist()
                vector = np.zeros(10*11)
                i = 0
                for n in text:
                    vector[i*captcha_len + n] = 1
                    i += 1
                result = vec2text(vector)
                cnt += 1
                print("No. %s : true label is %s and result is %s" % (cnt, true_label, result.strip()))
                if true_label == result.strip():
                    #print("yeah")
                    positive_cnt += 1
                #else:
                    #print("fuck")

            print("Accuracy on test_data is %f" % (float(positive_cnt)/float(test_len)))
        else:
            print("No checkpoint found!")


# 训练函数
# 支持中断及继续训练
def train():
    if not os.path.exists('/Users/ethan/MyCodes/baidu_tensor/models/'):
        os.mkdir('/Users/ethan/MyCodes/baidu_tensor/models/')

    # 输入
    X = tf.placeholder(tf.float32, [None, width*height])
    Y = tf.placeholder(tf.float32, [None, captcha_size*captcha_len])
    keep_prob = tf.placeholder(tf.float32)

    out = inference(X, keep_prob)
    
    # 损失函数
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=out))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    accuracy = verify(out, Y)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        if os.path.exists('/Users/ethan/MyCodes/baidu_tensor/models/checkpoint'):
            print("Found last check point, restoring...")
            saver.restore(sess, '/Users/ethan/MyCodes/baidu_tensor/models/model.ckpt')
        else:
            print("New training...")
            init = tf.global_variables_initializer()
            sess.run(init)

        for i in range(301):
            batch_x, batch_y, text, pic = gen_next_batch(64)
            _, loss_value = sess.run([optimizer, loss], feed_dict={X: batch_x, keep_prob: 0.75, Y: batch_y})
            if i % 100 == 0:
                print(i, loss_value)
                batch_x_test, batch_y_test, text, pic = gen_next_batch(100)
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print("*****%s steps, accuracy is %s" % (i, acc))             
                saver.save(sess, '/Users/ethan/MyCodes/baidu_tensor/models/model.ckpt')


def main():
    #train()
    test('/Users/ethan/MyCodes/baidu_tensor/test/*.jpg')

if __name__ == '__main__':
    main()