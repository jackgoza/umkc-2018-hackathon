import tensorflow as tf, sys

imagePath = 'goodtire1.jpg'

tire_labels = "data/tire_output_labels.txt"
tire_graph = "data/tire_output_graph.pb"
damage_labels = "data/damage_output_labels.txt"
damage_graph = "data/damage_output_graph.pb"
interior_labels = "data/interior_output_labels.txt"
interior_graph = "data/interior_output_graph.pb"
rust_labels = "data/rust_output_labels.txt"
rust_graph = "data/rust_output_graph.pb"


def inception_rank(imagePath, labels, graph):
    #image_path = sys.argv[1]
    image_path = imagePath
    # image_path = '676728-bigthumbnail.jpg'

    # Read in the image_data
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()

    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile(labels)]

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

        for node_id in top_k:
            human_string = label_lines[node_id]
            score = predictions[0][node_id]
            print('%s (score = %.5f)' % (human_string, score))


inception_rank(imagePath, tire_labels, tire_graph)
