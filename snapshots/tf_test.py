import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.keras import backend as K
sess = K.get_session()
f = gfile.FastGFile("satellite_iv3_ft.pb", 'rb')
graph_def = tf.GraphDef()
# Parses a serialized binary message into the current message.
graph_def.ParseFromString(f.read())
f.close()

sess.graph.as_default()
# Import a serialized TensorFlow `GraphDef` protocol buffer
# and place into the current default `Graph`.
tf.import_graph_def(graph_def)


