import tensorflow as tf


def loadRCNN(model_path):
    return loadSession(model_path, 'import/image_tensor:0', 'import/detection_classes:0', 'import/detection_scores:0', 'import/detection_boxes:0')

def loadFrozenGraph(model_path):
    frozenFileName = model_path + 'frozen_inference_graph.pb'
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(frozenFileName, 'rb') as fid:
            graph_def.ParseFromString(fid.read())
            tf.import_graph_def(graph_def)
    return graph

def loadSession(model_path, input_tensor_name, classes_name, scores_name, boxes_name):
    graph = loadFrozenGraph(model_path)
    x = graph.get_tensor_by_name(input_tensor_name)
    classes = graph.get_tensor_by_name(classes_name)
    scores = graph.get_tensor_by_name(scores_name)
    boxes = graph.get_tensor_by_name(boxes_name)
    sess = tf.Session(graph=graph)
    return sess, x, classes, scores, boxes


