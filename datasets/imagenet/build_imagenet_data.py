# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts ImageNet data to TFRecords file format with Example protos.

The raw ImageNet data set is expected to reside in JPEG files located in the
following directory structure.

  data_dir/n01440764/ILSVRC2012_val_00000293.JPEG
  data_dir/n01440764/ILSVRC2012_val_00000543.JPEG
  ...

where 'n01440764' is the unique synset label associated with
these images.

The training data set consists of 1000 sub-directories (i.e. labels)
each containing 1200 JPEG images for a total of 1.2M JPEG images.

The evaluation data set consists of 1000 sub-directories (i.e. labels)
each containing 50 JPEG images for a total of 50K JPEG images.

This TensorFlow script converts the training and evaluation data into
a sharded data set consisting of 1024 and 128 TFRecord files, respectively.

  train_directory/train-00000-of-01024
  train_directory/train-00001-of-01024
  ...
  train_directory/train-01023-of-01024

and

  validation_directory/validation-00000-of-00128
  validation_directory/validation-00001-of-00128
  ...
  validation_directory/validation-00127-of-00128

Each validation TFRecord file contains ~390 records. Each training TFREcord
file contains ~1250 records. Each record within the TFRecord file is a
serialized Example proto. The Example proto contains the following fields:

  image/encoded: string containing JPEG encoded image in RGB colorspace
  image/height: integer, image height in pixels
  image/width: integer, image width in pixels
  image/colorspace: string, specifying the colorspace, always 'RGB'
  image/channels: integer, specifying the number of channels, always 3
  image/format: string, specifying the format, always 'JPEG'

  image/filename: string containing the basename of the image file
            e.g. 'n01440764_10026.JPEG' or 'ILSVRC2012_val_00000293.JPEG'
  image/class/label: integer specifying the index in a classification layer.
    The label ranges from [1, 1000] where 0 is not used.
  image/class/synset: string specifying the unique ID of the label,
    e.g. 'n01440764'
  image/class/text: string specifying the human-readable version of the label
    e.g. 'red fox, Vulpes vulpes'

  image/object/bbox/xmin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/xmax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymin: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/ymax: list of integers specifying the 0+ human annotated
    bounding boxes
  image/object/bbox/label: integer specifying the index in a classification
    layer. The label ranges from [1, 1000] where 0 is not used. Note this is
    always identical to the image label.

Note that the length of xmin is identical to the length of xmax, ymin and ymax
for each example.

Running this script using 16 threads may take around ~2.5 hours on a HP Z420.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import random
import sys
import threading
import glob
import shutil
from filelock import FileLock

import numpy as np
from scipy import stats
import tensorflow as tf
import pickle as pkl

#tf.app.flags.DEFINE_string('train_directory', '/tmp/',
#                           'Training data directory')
#tf.app.flags.DEFINE_string('validation_directory', '/tmp/',
#                           'Validation data directory')
tf.app.flags.DEFINE_string('output_directory', '/tmp/',
                           'Output data directory')
#
#tf.app.flags.DEFINE_string('train_list', '/tmp/',
#                           ('training file that contains a newline delimited list '
#                            'of xml files that will be used to generate the '
#                            'training data set. The xml files contain both '
#                            'the image file name and bouding boxs'))
#tf.app.flags.DEFINE_string('validation_list', '/tmp/',
#                           ('validation file that contains a newline delimited list '
#                            'of xml files that will be used to generate the '
#                            'validation data set. The xml files contain both '
#                            'the image file name and bouding boxs'))
#tf.app.flags.DEFINE_string('testing_list', '/tmp/',
#                           ('test file that contains a newline delimited list '
#                            'of xml files that will be used to generate the '
#                            'testing data set. The xml files contain both '
#                            'the image file name and bouding boxs'))

#tf.app.flags.DEFINE_integer('train_shards', 1024,
#                            'Number of shards in training TFRecord files.')
#tf.app.flags.DEFINE_integer('validation_shards', 128,
#                            'Number of shards in validation TFRecord files.')
tf.app.flags.DEFINE_integer('shards', 1024,
                            'Number of shards in training TFRecord files.')

tf.app.flags.DEFINE_string('data_list_file', 'training_files.txt',
                           ('contains a list of the base image file name '
                            'and the corresponding synset that the annotation '
                            'is found in.'))
tf.app.flags.DEFINE_string('name', 'train',
                           'Unique identifer to label the data set')

tf.app.flags.DEFINE_string('data_dir', '',
                           'this directory contains the images and annotations')

# The labels file contains a list of valid labels are held in this file.
# Assumes that the file contains entries as such:
#   n01440764
#   n01443537
#   n01484850
# where each line corresponds to a label expressed as a synset. We map
# each synset contained in the file to an integer (based on the alphabetical
# ordering). See below for details.
#tf.app.flags.DEFINE_string('labels_file',
#                           'imagenet_lsvrc_2015_synsets.txt',
#                           'Labels file')

# This file containing mapping from synset to human-readable label.
# Assumes each line of the file looks like:
#
#   n02119247    black fox
#   n02119359    silver fox
#   n02119477    red fox, Vulpes fulva
#
# where each line corresponds to a unique mapping. Note that each line is
# formatted as <synset>\t<human readable label>.
tf.app.flags.DEFINE_string('imagenet_metadata_file',
                           'imagenet_metadata.txt',
                           'ImageNet metadata file')

# This file is the output of process_bounding_box.py
# Assumes each line of the file looks like:
#
#   n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940
#
# where each line corresponds to one bounding box annotation associated
# with an image. Each line can be parsed as:
#
#   <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>
#
# Note that there might exist mulitple bounding box annotations associated
# with an image file.
#tf.app.flags.DEFINE_string('bounding_box_file',
#                           './imagenet_2012_bounding_boxes.csv',
#                           'Bounding box file')
#
# This arguments is to allow script to run on a distributed system
tf.app.flags.DEFINE_integer('proc_tot', 0,
                            'Total number or processes that this script is being run in.')
tf.app.flags.DEFINE_integer('proc_index', None,
                            'process index that this scrip is being run in.')
tf.app.flags.DEFINE_boolean('distributed', False,
                            'Select weather to use distributed processing accross multiple processes or threading')
tf.app.flags.DEFINE_boolean('gen_bbox_store', False,
                            'process xml bounding boxes')

FLAGS = tf.app.flags.FLAGS


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
  """Wrapper for inserting float features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _spacing(tot_length, num_div):
  """
  """
  return np.linspace(0,tot_length, num_div + 1).astype(np.int)

def _len_spacing(tot_length, num_div):
  """
  """
  return len(np.linspace(0,tot_length, num_div + 1).astype(np.int)) - 1

def _convert_to_example(filename, image_buffer, label, synset, human, bbox,
                        height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    synset: string, unique WordNet ID specifying the label, e.g., 'n02323233'
    human: string, human-readable label, e.g., 'red fox, Vulpes vulpes'
    bbox: list of bounding boxes; each box is a list of integers
      specifying [xmin, ymin, xmax, ymax]. All boxes are assumed to belong to
      the same label as the image label.
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  xmin = []
  ymin = []
  xmax = []
  ymax = []
  for b in bbox:
    assert len(b) == 4
    # pylint: disable=expression-not-assigned
    [l.append(point) for l, point in zip([xmin, ymin, xmax, ymax], b)]
    # pylint: enable=expression-not-assigned

  colorspace = 'RGB'
  channels = 3
  image_format = 'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/colorspace': _bytes_feature(colorspace),
      'image/channels': _int64_feature(channels),
      'image/class/label': _int64_feature(label),
      'image/class/synset': _bytes_feature(synset),
      'image/class/text': _bytes_feature(human),
      'image/object/bbox/xmin': _float_feature(xmin),
      'image/object/bbox/xmax': _float_feature(xmax),
      'image/object/bbox/ymin': _float_feature(ymin),
      'image/object/bbox/ymax': _float_feature(ymax),
      'image/object/bbox/label': _int64_feature([label] * len(xmin)),
      'image/format': _bytes_feature(image_format),
      'image/filename': _bytes_feature(os.path.basename(filename)),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Create a single Session to run all image coding calls.
    self._sess = tf.Session()

    # Initializes function that converts PNG to JPEG data.
    self._png_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_png(self._png_data, channels=3)
    self._png_to_jpeg = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that converts CMYK JPEG data to RGB JPEG data.
    self._cmyk_data = tf.placeholder(dtype=tf.string)
    image = tf.image.decode_jpeg(self._cmyk_data, channels=0)
    self._cmyk_to_rgb = tf.image.encode_jpeg(image, format='rgb', quality=100)

    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def png_to_jpeg(self, image_data):
    return self._sess.run(self._png_to_jpeg,
                          feed_dict={self._png_data: image_data})

  def cmyk_to_rgb(self, image_data):
    return self._sess.run(self._cmyk_to_rgb,
                          feed_dict={self._cmyk_data: image_data})

  def decode_jpeg(self, image_data):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  # File list from:
  # https://groups.google.com/forum/embed/?place=forum/torch7#!topic/torch7/fOSTXHIESSU
  return 'n02105855_2933.JPEG' in filename


def _is_cmyk(filename):
  """Determine if file contains a CMYK JPEG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a JPEG encoded with CMYK color space.
  """
  # File list from:
  # https://github.com/cytsai/ilsvrc-cmyk-image-list
  blacklist = ['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
               'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
               'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
               'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
               'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
               'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
               'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
               'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
               'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
               'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
               'n07583066_647.JPEG', 'n13037406_4650.JPEG']
  return filename.split('/')[-1] in blacklist


def _process_image(filename, coder):
  """Process a single image file.

  Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
  Returns:
    image_buffer: string, JPEG encoding of RGB image.
    height: integer, image height in pixels.
    width: integer, image width in pixels.
  """
  # Read the image file.
  image_data = tf.gfile.FastGFile(filename, 'r').read()

  # Clean the dirty data.
  if _is_png(filename):
    # 1 image is a PNG.
    print('Converting PNG to JPEG for %s' % filename)
    image_data = coder.png_to_jpeg(image_data)
  elif _is_cmyk(filename):
    # 22 JPEG images are in CMYK colorspace.
    print('Converting CMYK to RGB for %s' % filename)
    image_data = coder.cmyk_to_rgb(image_data)

  # Decode the RGB JPEG.
  image = coder.decode_jpeg(image_data)

  # Check that image converted to RGB
  assert len(image.shape) == 3
  height = image.shape[0]
  width = image.shape[1]
  assert image.shape[2] == 3

  return image_data, height, width

def _process_image_files_batch(coder, thread_index, ranges, name, filenames,
                               synsets, labels, humans, bboxes, num_shards,
                               output_dir):
  """Processes and saves list of images as TFRecord in 1 thread.

  Args:
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    thread_index: integer, unique batch to run index is within [0, len(ranges)).
    ranges: list of pairs of integers specifying ranges of each batches to
      analyze in parallel.
    name: string, unique identifier specifying the data set
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    humans: list of strings; each string is a human-readable label
    bboxes: list of bounding boxes for each image. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the image.
    num_shards: integer number of shards for this data set.
  """
  # Each thread produces N shards where N = int(num_shards / num_threads).
  # For instance, if num_shards = 128, and the num_threads = 2, then the first
  # thread would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0],
                             ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_files_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in range(num_shards_per_batch):
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = '%s-%.5d-of-%.5d' % (name, shard, num_shards)
    output_file = os.path.join(output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    files_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in files_in_shard:
      filename = filenames[i]
      label = labels[i]
      synset = synsets[i]
      human = humans[i]
      bbox = bboxes[i]

      image_buffer, height, width = _process_image(filename, coder)

      example = _convert_to_example(filename, image_buffer, label,
                                    synset, human, bbox,
                                    height, width)
      writer.write(example.SerializeToString())
      shard_counter += 1
      counter += 1

      if not counter % 1000:
        print('%s [thread %d]: Processed %d of %d images in thread batch.' %
              (datetime.now(), thread_index, counter, num_files_in_thread))
        sys.stdout.flush()

    writer.close()
    print('%s [thread %d]: Wrote %d images to %s' %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()
    shard_counter = 0
  print('%s [thread %d]: Wrote %d images to %d shards.' %
        (datetime.now(), thread_index, counter, num_files_in_thread))
  sys.stdout.flush()

def _process_image_files(name, output_dir, num_shards, data_list_file,
                         proc_tot, proc_index, distributed, filenames,
                         synsets, labels, humans, bboxes):
  """Process and save list of images as TFRecord of Example protos.

  Args:
    name: string, unique identifier specifying the data set
    output_dir: where tf_record will be placed
    num_shards: integer number of shards for this data set.
    data_list_file: file that contains image names and annotation synset
    proc_tot: 
    proc_index:
    distributed:
    filenames: list of strings; each string is a path to an image file
    synsets: list of strings; each string is a unique WordNet ID
    labels: list of integer; each integer identifies the ground truth
    humans: list of strings; each string is a human-readable label
    bboxes: list of bounding boxes for each image. Note that each entry in this
      list might contain from 0+ entries corresponding to the number of bounding
      box annotations for the image.

  """
  assert len(filenames) == len(synsets)
  assert len(filenames) == len(labels)
  assert len(filenames) == len(humans)
  assert len(filenames) == len(bboxes)
  assert distributed, 'this script only supports a distrubted run on BlueWaters'

  if proc_tot is not None:
    pass
  else:
    assert proc_index == 0

  # Break all images into batches with a [ranges[i][0], ranges[i][1]].
  # e.g.
  #   spacing = np.linspace(0, len(filenames), FLAGS.num_threads + 1).astype(np.int)
  spacing = _spacing(len(filenames), proc_tot)
  ranges = [[spacing[i], spacing[i+1]] for i in xrange(len(spacing) - 1)]

  if distributed:
    num_threads = 1
    launch_message = 'Launched %d process'
  else:
    num_threads = proc_tot
    launch_message = 'Launching %d threads'

  # Launch a thread for each batch.
  print(launch_message % (num_threads))
  sys.stdout.flush()

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()

  # Create a generic TensorFlow-based utility for converting all image codings.
  coder = ImageCoder()

  args = (coder, proc_index, ranges, name, filenames,
          synsets, labels, humans, bboxes, num_shards,
          output_dir)

  if distributed:
    _process_image_files_batch(*args)
  else:
    threads = []
    for thread_index, range in enumerate(ranges):
      args[1] = thread_index
      t = threading.Thread(target=_process_image_files_batch, args=args)
      t.start()
      threads.append(t)
       
      # Wait for all the threads to terminate.
      coord.join(threads)

  print('%s: Finished writing all %d images in data set.' %
        (datetime.now(), len(filenames)))
  sys.stdout.flush()


def _find_image_files(data_dir, data_list_file):
  """Build a list of all images files and labels in the data set.

  Args:
    data_dir: output directory of extract_data_from_archive.pbs
    data_list_file: file used in extract_data_from_archive.pbs to extract JPEG and xml data
                    e.g. training_files.txt

      Assumes that the ImageNet data set resides in JPEG files located in
      the following directory structure.

        data_dir/n01440764/n01440764_293.JPEG
        data_dir/n01440764/n01440764_543.JPEG

      where 'n01440764' is the unique synset label associated with these images.

  Returns:
    filenames: list of strings; each string is a path to an image file.
    synsets: list of strings; each string is a unique WordNet ID.
    labels: list of integer; each integer identifies the ground truth.
  """
  with open(data_list_file) as f:
    data_list = [l.strip() for l in f]

  file_name_dir = [n.split(' ')[1].split('_') for n in data_list]
  challenge_synsets = list(set([n[0] for n in file_name_dir]))

  labels = []
  filenames = []
  synsets = []

  image_data_dir = os.path.join(data_dir, 'Images')

  #get all possible image files keyed by their synset
  synset_image_dict = {k: [] for k in challenge_synsets}

  _ = [synset_image_dict[n[0]].append(os.path.join(*[image_data_dir, n[0] + '_' + n[1] + '.JPEG' ])) \
                                     for n in file_name_dir]

  # check that the data_dir contains the same files as data_list_file
#  jpeg_file_in_data_dir = [l.split('/')[-1] \
#                             for l in glob.glob('%s/*/*.JPEG' % image_data_dir)]
#
#
#  assert set(jpeg_file_in_data_dir) == set([l for v in synset_image_dict.values() for l in v]), \
#      ('image file in %s do not match files from %s, perhaps '
#       'you need to chose a diferent data_list_file') \
#      % (image_data_dir, data_list_file)

  # Construct the list of JPEG files and labels.
  # Leave label index 0 empty as a background class.
  for label_index, synset in enumerate(challenge_synsets, start=1):
    matching_files = synset_image_dict[synset]
    print('synset: {}, matching files {}'.format(synset, len(matching_files)))
    labels.extend([label_index] * len(matching_files))
    synsets.extend([synset] * len(matching_files))
    filenames.extend(matching_files)

    if not label_index % 100:
      print('Finished finding files in %d of %d classes.' % (
          label_index, len(challenge_synsets)))

  # Shuffle the ordering of all image files in order to guarantee
  # random ordering of the images with respect to label in the
  # saved TFRecord files. Make the randomization repeatable.
  shuffled_index = range(len(filenames))
  random.seed(12345)
  random.shuffle(shuffled_index)

  filenames = [filenames[i] for i in shuffled_index]
  synsets = [synsets[i] for i in shuffled_index]
  labels = [labels[i] for i in shuffled_index]

  print('Found %d JPEG files across %d labels inside %s.' %
        (len(filenames), len(challenge_synsets), data_dir))
  return filenames, synsets, labels


def _find_human_readable_labels(synsets, synset_to_human):
  """Build a list of human-readable labels.

  Args:
    synsets: list of strings; each string is a unique WordNet ID.
    synset_to_human: dict of synset to human labels, e.g.,
      'n02119022' --> 'red fox, Vulpes vulpes'

  Returns:
    List of human-readable strings corresponding to each synset.
  """
  humans = []
  for s in synsets:
    assert s in synset_to_human, ('Failed to find: %s' % s)
    humans.append(synset_to_human[s])
  return humans


def _find_image_bounding_boxes(data_dir, data_list_file, proc_tot, proc_index):
  """Find the bounding boxes for a given image file.

  Args:
    data_dir: output directory of extract_data_from_archive.pbs
    data_list_file: file used in extract_data_from_archive.pbs to extract JPEG and xml data
                    e.g. training_files.txt
    image_to_bboxes: dictionary mapping image file names to a list of
      bounding boxes. This list contains 0+ bounding boxes.
  Returns:
    List of bounding boxes for each image. Note that each entry in this
    list might contain from 0+ entries corresponding to the number of bounding
    box annotations for the image.
  """
  path_to_bb_xml_util = os.path.join(*[os.path.dirname(os.path.abspath(__file__)),
                                       '..', '..',
                                       'models', 'research', 'inception',
                                       'inception', 'data'])
  sys.path.append(path_to_bb_xml_util)
  from process_bounding_boxes import ProcessXMLAnnotation
  
  ann_data_dir = os.path.join(data_dir, 'Annotation')

  with open(data_list_file) as f:
    file_names_synset_pair = [tuple(l.strip().split(' ')) for l in f]
    
  spacing = _spacing(len(file_names_synset_pair), proc_tot)

  file_names = [os.path.join(*i) + ".xml" \
                  for i in file_names_synset_pair[spacing[proc_index]:spacing[proc_index+1]]]

  bboxes = []
  for i, f_n in enumerate(file_names):
    sys.stdout.flush()
    print('extracting bb from %s (%d of %d)' % (f_n, i, len(file_names)))
    xml_filepath_name = os.path.join(*[ann_data_dir, f_n])
    bboxes_from_file = ProcessXMLAnnotation(xml_filepath_name)
    if bboxes_from_file is None:
      raise Exception('No bounding boxes found in ' + f_n)
    bboxes.append([])
    for bbox in bboxes_from_file:
      if (bbox.xmin_scaled > bbox.xmax_scaled or
              bbox.ymin_scaled > bbox.ymax_scaled):
        raise Exception( 'malformed bb in ' + f_n)

      bboxes[-1].append([bbox.xmin_scaled, bbox.ymin_scaled,
                         bbox.xmax_scaled, bbox.ymax_scaled])
      

  bb_lengths = [len(b) for b in bboxes]
  print('Found %d bboxes out of %d images' % (
      sum(bb_lengths), len(file_names)))

  bb_stats = stats.describe(bb_lengths)
  print('Basic statistics on bboxes per file are:\n ______________')
  for s_n in ['nobs', 'minmax', 'mean', 'variance', 'skewness', 'kurtosis']:
    print('{}: {}'.format(s_n,getattr(bb_stats, s_n)))
  print('--------------------')
          
    
  return bboxes


def _process_dataset(name, data_dir, num_shards, synset_to_human,
                     data_list_file, output_dir, bboxes, proc_tot=None,
                     proc_index=0, distributed=False):
  """Process a complete data set and save it as a TFRecord.

  Args:
    name: string, unique identifier specifying the data set.
    data_dir: string, root path to the data set.
    num_shards: integer number of shards for this data set.
    synset_to_human: dict of synset to human labels, e.g.,
      'n02119022' --> 'red fox, Vulpes vulpes'
    data_list_file: 
    proc_tot: 
    proc_index: 
    distributed: 
  """

  filenames, synsets, labels = _find_image_files(data_dir, data_list_file)
  sys.stdout.flush()
  humans = _find_human_readable_labels(synsets, synset_to_human)
  sys.stdout.flush()
  _process_image_files(name, output_dir, num_shards, data_list_file,
                       proc_tot, proc_index, distributed, filenames,
                       synsets, labels, humans, bboxes)
  sys.stdout.flush()


def _build_synset_lookup(imagenet_metadata_file):
  """Build lookup for synset to human-readable label.

  Args:
    imagenet_metadata_file: string, path to file containing mapping from
      synset to human-readable label.

      Assumes each line of the file looks like:

        n02119247    black fox
        n02119359    silver fox
        n02119477    red fox, Vulpes fulva

      where each line corresponds to a unique mapping. Note that each line is
      formatted as <synset>\t<human readable label>.

  Returns:
    Dictionary of synset to human labels, such as:
      'n02119022' --> 'red fox, Vulpes vulpes'
  """
  lines = tf.gfile.FastGFile(imagenet_metadata_file, 'r').readlines()
  synset_to_human = {}
  for l in lines:
    if l:
      parts = l.strip().split('\t')
      assert len(parts) == 2
      synset = parts[0]
      human = parts[1]
      synset_to_human[synset] = human
  return synset_to_human


def _build_bounding_box_lookup(bounding_box_file):
  """Build a lookup from image file to bounding boxes.

  Args:
    bounding_box_file: string, path to file with bounding boxes annotations.

      Assumes each line of the file looks like:

        n00007846_64193.JPEG,0.0060,0.2620,0.7545,0.9940

      where each line corresponds to one bounding box annotation associated
      with an image. Each line can be parsed as:

        <JPEG file name>, <xmin>, <ymin>, <xmax>, <ymax>

      Note that there might exist mulitple bounding box annotations associated
      with an image file. This file is the output of process_bounding_boxes.py.

  Returns:
    Dictionary mapping image file names to a list of bounding boxes. This list
    contains 0+ bounding boxes.
  """
  lines = tf.gfile.FastGFile(bounding_box_file, 'r').readlines()
  images_to_bboxes = {}
  num_bbox = 0
  num_image = 0
  for l in lines:
    if l:
      parts = l.split(',')
      assert len(parts) == 5, ('Failed to parse: %s' % l)
      filename = parts[0]
      xmin = float(parts[1])
      ymin = float(parts[2])
      xmax = float(parts[3])
      ymax = float(parts[4])
      box = [xmin, ymin, xmax, ymax]

      if filename not in images_to_bboxes:
        images_to_bboxes[filename] = []
        num_image += 1
      images_to_bboxes[filename].append(box)
      num_bbox += 1

  print('Successfully read %d bounding boxes '
        'across %d images.' % (num_bbox, num_image))
  return images_to_bboxes


def main(unused_argv):
  if FLAGS.gen_bbox_store:
    print('doing _find_image_bounding_boxes()')
    bboxes = _find_image_bounding_boxes(FLAGS.data_dir,
                                        FLAGS.data_list_file,
                                        FLAGS.proc_tot,
                                        FLAGS.proc_index)
    with open('tmp_bbox_stash_index_{}.pkl'.format(FLAGS.proc_index), 'wb') as f:
      print('Saving BB store ' + 'tmp_bbox_stash_index_{}.pkl'.format(FLAGS.proc_index))
      pkl.dump(bboxes, f)
  else:
    print('Saving results to %s' % FLAGS.output_directory)
    bboxes = []
    for i in range(FLAGS.proc_tot):
      with open('tmp_bbox_stash_index_{}.pkl'.format(i), 'rb') as f:
        bboxes.extend(pkl.load(f))
    # Build a map from synset to human-readable label.
    synset_to_human = _build_synset_lookup(FLAGS.imagenet_metadata_file)

    # Run it!
    _process_dataset(FLAGS.name, FLAGS.data_dir, FLAGS.shards, synset_to_human,
                     FLAGS.data_list_file, FLAGS.output_directory, bboxes,
                     FLAGS.proc_tot, FLAGS.proc_index, FLAGS.distributed)

    if FLAGS.distributed:
      print('Done in main: index %d of %d' % (FLAGS.proc_index, FLAGS.proc_tot))
    else:
      print('Done')


if __name__ == '__main__':
  tf.app.run()
