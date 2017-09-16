#识花API
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import tensorflow as tf

import os
from flask import Flask, request, url_for, send_from_directory
from werkzeug import secure_filename

#预测结果，全局变量
result = 'NULL'

#图片路径，全局变量
g_filename = '1.jpg'


#配置flask
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './photos'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>图片上传</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file>
         <input type=submit value=上传>
    </form>
    '''




parser = argparse.ArgumentParser()
parser.add_argument(
    '--image',
    # required=True,
    type=str, default='flower_photos/daisy/54377391_15648e8d18.jpg',help='Absolute path to image file.')
parser.add_argument(
    '--num_top_predictions',
    type=int,
    default=5,
    help='Display this many predictions.')
parser.add_argument(
    '--graph',
    # required=True,
    type=str,
    default='./model/retrained_graph.pb',
    help='Absolute path to graph file (.pb)')
parser.add_argument(
    '--labels',
    # required=True,
    type=str,
    default='./model/retrained_labels.txt',
    help='Absolute path to labels file (.txt)')
parser.add_argument(
    '--output_layer',
    type=str,
    default='final_result:0',
    help='Name of the result operation')
parser.add_argument(
    '--input_layer',
    type=str,
    default='DecodeJpeg/contents:0',
    help='Name of the input operation')


def load_image(filename):
  """Read in the image_data to be classified."""
  return tf.gfile.FastGFile(filename, 'rb').read()


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def run_graph(image_data, labels, input_layer_name, output_layer_name,
              num_top_predictions):
  with tf.Session() as sess:
    # Feed the image_data as input to the graph.
    #   predictions will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: image_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    # for node_id in top_k:
    #   human_string = labels[node_id]
    #   score = predictions[node_id]
    #   print('%s (score = %.5f)' % (human_string, score))

    human_string = labels[top_k[0]]
    global result
    result = human_string
    print(human_string)

    return 0



def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            global g_filename
            g_filename = filename
            print(filename)
            #图片接收完成后运行TensorFlow进行预测
            tf.app.run(main=main, argv=sys.argv[:1] + unparsed)
            file_url = url_for('uploaded_file', filename=filename)
            #return html + '<br><img src=' + file_url + '>'
            return result
    return html

def main(argv):

  """Runs inference on an image."""
  if argv[1:]:
    raise ValueError('Unused Command Line Args: %s' % argv[1:])

  if not tf.gfile.Exists(FLAGS.image):
    tf.logging.fatal('image file does not exist %s', FLAGS.image)

  if not tf.gfile.Exists(FLAGS.labels):
    tf.logging.fatal('labels file does not exist %s', FLAGS.labels)

  if not tf.gfile.Exists(FLAGS.graph):
    tf.logging.fatal('graph file does not exist %s', FLAGS.graph)

  global g_filename
  print(g_filename)

  # load image
  # image_data = load_image(FLAGS.image)
  image_data = load_image('./photos/'+g_filename)

  # load labels
  labels = load_labels(FLAGS.labels)

  # load graph, which is stored in the default session
  load_graph(FLAGS.graph)

  run_graph(image_data, labels, FLAGS.input_layer, FLAGS.output_layer,
            FLAGS.num_top_predictions)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  # tf.app.run(main=main, argv=sys.argv[:1]+unparsed)

  #运行flask
  app.run()