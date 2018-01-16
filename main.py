import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import math
import cv2
import tensorflow as tf
from model import model_factory
from grad_cam_plus_plus import GradCamPlusPlus


def write_summary(log_dir, names, imgs, sess):
	image_summaries = list()
	for i, name in enumerate(names):
		img_tensor = tf.constant(np.expand_dims(imgs[i][:, :, ::-1], axis=0))
		image_summaries.append(tf.summary.image(name, img_tensor, 1))
	merged_image_summary = tf.summary.merge(image_summaries)
	with tf.summary.FileWriter(log_dir) as summary_writer:
		summary_writer.add_summary(sess.run(merged_image_summary))


def show_image(img, title='title'):
	cv2.imshow(title, img)
	while True:
		if cv2.waitKey(0) == 113:  # 'q' key
			break


def show_result(imgs, classes):
	### Change color channels
	imgs = [img[:, :, ::-1] for img in imgs]

	### Define rows/cols of the figure
	n_imgs = len(imgs)
	max_cols = 4
	max_rows = int(math.ceil(n_imgs / float(max_cols)))

	### Set figure size
	fig = plt.figure()
	fig.set_size_inches(3.5 * max_cols, 2.5 * max_rows)
	fig.subplots_adjust(left=0.04, right=0.70, hspace=0.0)

	idx_img = 0
	plt_idx = 1
	class_idx = 0
	for row in range(max_rows):
		for col in range(max_cols):
			### Show a image
			try:
				fig.add_subplot(max_rows, max_cols, plt_idx)
				if row == 0:
					if col == 3:
						plt.title('box')
					else:
						plt.title('top {}'.format(col + 1))
				plt.axis('off')
				plt.imshow(imgs[idx_img])
			except IndexError:
				continue

			### Show top-3 class
			if idx_img != 0 and (idx_img + 1) % max_cols == 0:
				x = imgs[idx_img].shape[1] + 10
				y = 20
				for i in range(3):
					y = 20 + y
					color = [0.0, 0.0, 0.0]
					color[i] = 1.0
					font = {'color': tuple(color), 'size': 10, 'weight': 'bold'}  # color : rgb
					plt.text(x, y, classes[class_idx], fontdict=font)
					class_idx += 1

			idx_img += 1
			plt_idx += 1

	### Show images
	plt.show()  # bgr <-> rgb, only image


def load_image(path, image_size):
	### Load image as RGB
	img = cv2.imread(path, cv2.IMREAD_COLOR)

	### Resize image
	resized_img = cv2.resize(img, (image_size, image_size))

	return resized_img  # image of shape (image_size, image_size, 3)


def load_images(filenames, image_size):
	imgs = None
	for filename in filenames:
		path = os.path.join('input', filename)
		img = load_image(path, image_size)
		imgs = [img] if imgs is None else np.append(imgs, [img], axis=0)

	return np.array(imgs)


def do_slim_model(model_name, logits_layer_name, last_conv_layer_name, ckpt_file, synset, num_classes, image_size, num_channel, filenames):
	result_imgs = list()
	result_classes = list()
	summary_names = list()

	### Load image
	imgs = load_images(filenames, image_size)

	### Define model
	inputs = tf.placeholder(tf.float32, shape=[None, image_size, image_size, num_channel], name="inputs")
	model_f = model_factory.get_network_fn(model_name, num_classes, weight_decay=0.00004, is_training=False)
	logits, end_points = model_f(inputs)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		### Load weight
		saver = tf.train.Saver()
		saver.restore(sess, ckpt_file)

		### Predict
		probs = sess.run(end_points[logits_layer_name], feed_dict={inputs: imgs})

		### Create CAM image
		grad_cam_plus_plus = GradCamPlusPlus(sess, end_points[logits_layer_name], end_points[last_conv_layer_name], inputs)
		cam_imgs, class_indices = grad_cam_plus_plus.create_cam_img(imgs, probs)

		for i, filename in enumerate(filenames):
			box_img = np.copy(imgs[i])
			filename = os.path.basename(filename)
			for j in range(GradCamPlusPlus.TOP3):
				### Overlay heatmap
				heapmap = grad_cam_plus_plus.convert_cam_2_heatmap(cam_imgs[i][j])
				overlay_img = grad_cam_plus_plus.overlay_heatmap(imgs[i], heapmap)
				result_imgs.append(overlay_img)
				summary_names.append('{}-Top{}'.format(filename, j))

				### Boxing
				color = [0, 0, 0]
				color[j] = 255
				box_img = grad_cam_plus_plus.draw_rectangle(box_img, cam_imgs[i][j], color)

				### Get label
				result_classes.append(synset[class_indices[i][j]])

			result_imgs.append(box_img)
			summary_names.append('{}-Boxing'.format(filename))

		### Write summary
		if os.path.exists('log'):
			for file in os.listdir('log'):
				os.remove(os.path.join('log', file))
		write_summary('log', summary_names, result_imgs, sess)

	return result_imgs, result_classes


def do_vgg_19():
	model_name = "vgg_19"
	logits_layer_name = 'vgg_19/fc8'
	last_conv_layer_name = 'vgg_19/conv5/conv5_4'
	ckpt_file = "slim/checkpoint/vgg_19.ckpt"

	synset = [l.strip() for l in open('slim/checkpoint/synset.txt').readlines()]
	num_classes = 1000

	image_size = 224
	num_channel = 3
	filenames = ['kite1.jpg', 'dog1.jpg', 'cat1.jpg', 'airplane1.jpg']

	result_imgs, result_classes = do_slim_model(model_name, logits_layer_name, last_conv_layer_name, ckpt_file, synset, num_classes, image_size, num_channel, filenames)
	show_result(result_imgs, result_classes)


def do_inception_v4():
	model_name = "inception_v4"
	logits_layer_name = 'Logits'
	last_conv_layer_name = 'Mixed_7c'
	ckpt_file = "slim/checkpoint/inception_v4.ckpt"

	synset = [l.strip() for l in open('slim/checkpoint/synset.txt').readlines()]
	num_classes = 1001

	image_size = 299
	num_channel = 3
	filenames = ['kite1.jpg', 'dog1.jpg', 'cat1.jpg', 'airplane1.jpg']

	result_imgs, result_classes = do_slim_model(model_name, logits_layer_name, last_conv_layer_name, ckpt_file, synset, num_classes, image_size, num_channel, filenames)
	show_result(result_imgs, result_classes)


if __name__ == '__main__':
	do_vgg_19()
	# do_inception_v4()

























