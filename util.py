from __future__ import absolute_import, division, print_function

import logging
import os
import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import (signature_constants, tag_constants)


def get_optimizer_by_name(optimizer_name, learning_rate):
  """
    Get optimizer object by the optimizer name.
    
    Args:
      optimizer_name: Name of the optimizer. 
      learning_rate: The learning rate.
    
    Return:
      The optimizer object.
    """

  logging.info("Use the optimizer: {}".format(optimizer_name))
  if optimizer_name == "sgd":
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer_name == "adadelta":
    optimizer = tf.train.AdadeltaOptimizer(learning_rate)
  elif optimizer_name == "adagrad":
    optimizer = tf.train.AdagradOptimizer(learning_rate)
  elif optimizer_name == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate)
  elif optimizer_name == "ftrl":
    optimizer = tf.train.FtrlOptimizer(learning_rate)
  elif optimizer_name == "rmsprop":
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
  else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  return optimizer


def save_model(model_path,
               model_version,
               sess,
               signature_def_map,
               is_save_graph=False):
  """
    Save the model in standard SavedModel format.
    
    Args:
      model_path: The path to model.
      model_version: The version of model.
      sess: The TensorFlow Session object.
      signature_def_map: The map of TensorFlow SignatureDef object.
      is_save_graph: Should save graph file of not.
    
    Return:
      None
    """

  export_path = os.path.join(model_path, str(model_version))
  if os.path.isdir(export_path) == True:
    logging.error("The model exists in path: {}".format(export_path))
    return

  try:
    # Save the SavedModel
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder = saved_model_builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        clear_devices=True,
        signature_def_map=signature_def_map,
        legacy_init_op=legacy_init_op)
    logging.info("Save the model in: {}".format(export_path))
    builder.save()

    # Save the GraphDef
    if is_save_graph == True:
      graph_file_name = "graph.pb"
      logging.info("Save the graph file in: {}".format(model_path))
      tf.train.write_graph(
          sess.graph_def, model_path, graph_file_name, as_text=False)

  except Exception as e:
    logging.error("Fail to export saved model, exception: {}".format(e))


def restore_from_checkpoint(sess, saver, checkpoint_file_path):
  """
    Restore session from checkpoint files.
    
    Args:
      sess: TensorFlow Session object.
      saver: TensorFlow Saver object.
      checkpoint_file_path: The checkpoint file path.
    
    Return:
      True if restore successfully and False if fail
    """
  if checkpoint_file_path:
    logging.info(
        "Restore session from checkpoint: {}".format(checkpoint_file_path))
    saver.restore(sess, checkpoint_file_path)
    return True
  else:
    logging.error("Checkpoint not found: {}".format(checkpoint_file_path))
    return False
