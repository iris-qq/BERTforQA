# _*_ coding:utf-8 _*_
# Author:Atlantis
# Date:2019-06-11

from __future__ import print_function

import requests
from run_squad import *
from datetime import datetime
import tensorflow as tf
from tokenization import FullTokenizer

tf.enable_eager_execution()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

endpoint = "http://localhost:8500"


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_unique_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_unique_ids.append(feature.unique_id)

    """The actual input function."""
    batch_size = FLAGS.predict_batch_size

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "unique_ids":
            tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d


class Client(object):
    def __init__(self, ):
        self.context = "保险可以从我们公司的官网上直接购买，或者咨询我们的客服电话11010100"
        self.output = "./output"
        self.max_seq_length = 384
        self.max_query_length = 64
        self.doc_stride = 128
        self.end_point = "http://localhost:8500"
        self.tokenizer = self.get_tokenizer()

    def get_id(self) -> int:
        timestamp = datetime.timestamp(datetime.now())
        return int(timestamp)

    def is_whitespace(self, c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    def get_tokens(self):
        doc_tokens = []
        prev_is_whitespace = True
        for c in self.context:
            if self.is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
        return doc_tokens

    def get_tokenizer(self):
        tokenization = FullTokenizer(
            vocab_file="./chinese_L-12_H-768_A-12/vocab.txt",
            do_lower_case=False
        )
        return tokenization

    def get_examples(self, question):
        qas_id = self.get_id()
        question_text = question
        start_position = -1
        end_position = -1
        orig_answer_text = ""
        is_impossible = False
        doc_tokens = self.get_tokens()
        return [SquadExample(
            qas_id=qas_id,
            question_text=question_text,
            doc_tokens=doc_tokens,
            orig_answer_text=orig_answer_text,
            start_position=start_position,
            end_position=end_position,
            is_impossible=is_impossible
        )]

    def get_writer(self):
        eval_writer = FeatureWriter(
            os.path.join(self.output, "eval.tf_record"),
            is_training=False
        )
        return eval_writer

    def get_index(self, logits, n_best_size=1):
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes[0]

    def predict(self, question):
        eval_examples = self.get_examples(question)
        eval_features = []

        def append_feature(feature):
            eval_features.append(feature)

        convert_examples_to_features(
            examples=eval_examples,
            tokenizer=self.tokenizer,
            max_seq_length=384,
            doc_stride=128,
            max_query_length=64,
            is_training=False,
            output_fn=append_feature
        )
        predict_dataset = input_fn_builder(
            features=eval_features,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=False
        )
        iterator = predict_dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        for k in next_element.keys():
            next_element[k] = next_element[k].numpy().tolist()
        json_data = {
            "model_name": "default",
            "data": next_element
        }
        result = requests.post(self.end_point, json=json_data)
        result = dict(result.json())
        start_index = self.get_index(result["start_logits"][0])
        end_index = self.get_index(result["end_logits"][0])
        print(start_index, end_index)
        output = self.context[start_index:end_index + 1]
        return output


if __name__ == '__main__':
    c = Client()
    prediction = c.predict("怎么买保险")
    print(prediction)
