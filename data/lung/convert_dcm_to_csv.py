#!/usr/bin/env python

from pydicom import dicomio
import numpy as np


def main():
  # Read label file
  label_filename = "./raw_data/stage1_labels.csv"
  id_label_map = {}
  for i in open(label_filename, "r"):
    if i.startswith("id,cancer"):
      continue
    id_label_pair = i.split(",")
    id_label_map[id_label_pair[0]] = int(id_label_pair[1].rstrip())

  # Read dcm file
  input_filename = "fa7a21165ae152b13def786e6afc3edf.dcm"
  output_filename = input_filename + ".csv"
  convert_dcm_to_csv(id_label_map, input_filename, output_filename)


def convert_dcm_to_csv(id_label_map, input_filename, output_filename):
  print("Start to convert dcm: {} to csv: {}".format(input_filename,
                                                     output_filename))

  ds = dicomio.read_file(input_filename)
  image_ndarray = ds.pixel_array
  label = id_label_map[ds.PatientID]
  csv_content = "{},".format(label)

  # Example: 512 * 512
  for i in image_ndarray:
    for j in i:
      csv_content += "{},".format(j)
  csv_content = csv_content[:-1] + "\n"

  with open(output_filename, "w") as f:
    f.write(csv_content)

  print("Successfully convert dcm: {} to csv: {}".format(input_filename,
                                                         output_filename))


if __name__ == "__main__":
  main()
