import tensorflow as tf
import numpy as np
# 定义解析TFRecord文件操作
reader = tf.TFRecordReader()
filename_queue = tf.train.string_input_producer(["/home/mxqian/Projects/Captcha/captcha_data/Records/"
                                                 "captcha_data.tfrecords"])

_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'data': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })

image = features['data']
label = features['label']

# 组合训练数据
# 每个batch中样本量的大小
batch_size = 1
# 总容量
capacity = 1000 + 3*batch_size
image_batch0, label_batch = tf.train.batch([image, label], batch_size=batch_size, capacity=capacity)
# 转化成图像处理可以识别的格式
image_batch = tf.decode_raw(image_batch0, tf.uint8)

# # 创建对话
# with tf.Session() as sess:
#     tf.global_variables_initializer()
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#
#     for i in range(1):
#         cur_example_batch, cur_label_batch = sess.run([image_batch, label_batch])
#         print(cur_example_batch, cur_label_batch)
#         print(np.shape(cur_example_batch))
#     coord.request_stop()
#     coord.join(threads)



