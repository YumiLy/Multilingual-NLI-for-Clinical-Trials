# -*- coding: utf-8 -*-
# @Author: bo.shi
# @Date:   2019-12-01 22:28:41
# @Last Modified by:   bo.shi
# @Last Modified time: 2019-12-02 18:36:50
# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utility functions for GLUE classification tasks."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import csv
import os
import six
import logging

import tensorflow as tf


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, delimiter="\t", quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  @classmethod
  def _read_txt(cls, input_file):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = f.readlines()
      lines = []
      for line in reader:
        lines.append(line.strip().split("_!_"))
      return lines

  @classmethod
  def _read_json(cls, input_file):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = f.readlines()
      lines = []
      for line in reader:
        lines.append(json.loads(line.strip()))
      return lines

  @property
  def logger(self):
      """Returns a logger instance
      """
      level = '.'.join([__name__,type(self).__name__])
      return logging.getLogger(level)
  


# class CMNLIProcessor(DataProcessor):
#   """Processor for the CMNLI data set."""

#   def get_train_examples(self, data_dir,partial_input=False,max_examples=float('inf')):
#     """See base class."""
#     return self._create_examples_json(os.path.join(data_dir, "train.50k.json"),
#                                         "train",
#                                         partial_input=partial_input,
#                                         max_examples=max_examples)

#   def get_dev_examples(self, data_dir,partial_input=False):
#     """See base class."""
#     return self._create_examples_json(os.path.join(data_dir, "dev.json"),
#                                         "dev",
#                                         partial_input=partial_input)

#   def get_arbitrary_examples(self, data_dir,name,partial_input=False):
#     """See base class."""
#     #return self._create_examples_json(os.path.join(data_dir, name), "dev")
#     return self._create_examples_json(data_dir, "dev",
#                                         partial_input=partial_input)

#   def get_test_examples(self, data_dir,partial_input=False):
#     """See base class."""
#     return self._create_examples_json(os.path.join(data_dir, "test.json"), "test")

#   def get_labels(self):
#     """See base class."""
#     return ["contradiction", "entailment", "neutral"]

#   def _create_examples_json(self, file_name, set_type,partial_input=False,max_examples=float('inf')):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     lines = tf.gfile.Open(file_name, "r")
#     index   = 0
#     skipped = 0
#     total   = 0
#     self.logger.info('Reading through json %s set, partial_input=%s, maximum examples=%s' %\
#                        (set_type,str(partial_input),str(max_examples)))
    
#     for k,line in enumerate(lines):
#       ## limit examples
#       if k >= max_examples: break
#       line_obj = json.loads(line)
#       index = index + 1
#       guid = "%s-%s" % (set_type, index)

#       ## put in empty premise 
#       if partial_input: text_a = "######"
#       else: text_a = convert_to_unicode(line_obj["sentence1"])
#       text_b = convert_to_unicode(line_obj["sentence2"])

#       label = convert_to_unicode(line_obj["label"]) if set_type != 'test' else 'neutral'
#       total += 1

#       if label != "-":
#         examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#       else:
#         skipped += 1

#     self.logger.info('Loaded %d (/ %d ) examples (skipped over %d without labels)' % (total-skipped,total,skipped))
#     return examples





class NLI4CTProcessor(object):
  """ 
  Processor for the NLI4CT data set.
  """

  ## 不知道为什么遇到'NLI4CTProcessor' object has no attribute 'logger'
  def __init__(self):
        # 初始化 logger
        self.logger = logging.getLogger(__name__)
        # 设置日志级别，例如 DEBUG, INFO, WARNING, ERROR, CRITICAL
        self.logger.setLevel(logging.INFO)
        # 创建一个日志处理器，这里使用控制台输出
        handler = logging.StreamHandler()
        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        # 将处理器添加到 logger
        self.logger.addHandler(handler)


  def get_train_examples(self, data_dir,partial_input=False,max_examples=float('inf')):
    """See base class."""
    return self._create_examples_json(os.path.join(data_dir, "train.json"),
                                        "train",
                                        partial_input=partial_input,
                                        max_examples=max_examples)

  def get_dev_examples(self, data_dir,partial_input=False):
    """See base class."""
    return self._create_examples_json(os.path.join(data_dir, "dev.json"),
                                        "dev",
                                        partial_input=partial_input)

  def get_arbitrary_examples(self, data_dir,name,partial_input=False):
    """See base class."""
    #return self._create_examples_json(os.path.join(data_dir, name), "dev")
    return self._create_examples_json(data_dir, "dev",
                                        partial_input=partial_input)

  def get_test_examples(self, data_dir,partial_input=False):
    """See base class."""
    return self._create_examples_json(os.path.join(data_dir, "test.json"), "test")

  def get_labels(self):
    """See base class."""
    return ["Contradiction", "Entailment"] # only 2 labels here

  def _create_examples_json(self, file_name, set_type, partial_input=False, max_examples=float('inf')):
    """Creates examples for the training and dev sets."""
    examples = []
    with tf.gfile.Open(file_name, "r") as file:
        data = json.load(file)  # 读取整个 JSON 文件
    index = 0
    total = 0
    self.logger.info('Reading through json %s set, partial_input=%s, maximum examples=%s' %
                     (set_type, str(partial_input), str(max_examples)))
    
    for line_obj in data:  # 直接遍历对象
        if index >= max_examples:
            break
        index += 1
        guid = "%s-%s" % (set_type, index)
        
        if partial_input:
            text_a = "######"
        else:
            text_a = convert_to_unicode(line_obj["hypothesis"])
        text_b = convert_to_unicode(line_obj["CTR_premise"])  # 修正字段名

        label = None
        if set_type != 'test':
            label = convert_to_unicode(line_obj["Label"])  # 仅当不是测试集时获取label

        total += 1
        
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    self.logger.info('Loaded %d examples from %s set' % (total, set_type))
    return examples
