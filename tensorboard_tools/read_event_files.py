#!/usr/bin/env python

import tensorflow as tf


def main():
  event_file_path = "/Users/tobe/code/tensorflow_template_application/tensorboard/events.out.tfevents.1527218992.mbp-2.local"
  for event in tf.train.summary_iterator(event_file_path):

    print("--------------------")
    print("Time: {}".format(event.wall_time))

    for v in event.summary.value:
      if v.tag == "loss_1":
        print("Loss: {}".format(v.simple_value))
      elif v.tag == "train_accuracy":
        print("Train accuracy: {}".format(v.simple_value))
      elif v.tag == "train_auc":
        print("Train auc: {}".format(v.simple_value))
      elif v.tag == "validate_accuracy":
        print("Validate accuracy: {}".format(v.simple_value))
      elif v.tag == "validate_auc":
        print("Validate auc: {}".format(v.simple_value))


if __name__ == "__main__":
  main()
