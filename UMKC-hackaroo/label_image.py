import tensorflow as tf
import urllib.request
import os


def download_image(url):

    img_name = 500
    full_name = str(img_name) + '.jpg'
    image_folder = "data_folder"

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    full_file_name = os.path.join(image_folder + '/', full_name)
    urllib.request.urlretrieve(url, full_file_name)

    return full_file_name


def inception_rank(image_url, labels, graph):

    image_path = download_image(image_url)

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(labels)]

    tf.reset_default_graph()

    # Unpersists graph from file
    with tf.gfile.FastGFile(graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        score_list = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            score_list.append((human_string, score))
            print('%s (score = %.5f)' % (human_string, score))

        return score_list


def inception_rank1(labels, graph):

    image_path = "data_folder/500.jpg"

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(labels)]

    tf.reset_default_graph()

    # Unpersists graph from file
    with tf.gfile.FastGFile(graph, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess1:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess1.graph.get_tensor_by_name('final_result:0')

        predictions = sess1.run(softmax_tensor, \
                               {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        score_list = []
        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            score_list.append((human_string, score))
            print('      %s (score = %.5f)' % (human_string, score))
        print('\n')

        return score_list

