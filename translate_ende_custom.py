from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
import tensorflow as tf

from collections import defaultdict

# the virtual zh-en datasets address to prevent downloading while vocab data generation
_NC_TRAIN_DATASETS = [[
    "http://data.com/training-parallel-commoncrawl.tgz",
    ["commoncrawl.de-en.en", "commoncrawl.de-en.de"]
]]

_NC_TEST_DATASETS = [[
    "http://data.com/training-parallel-nc-v13.tgz",
    ("news-commentary-v13.de-en.en", "news-commentary-v13.de-en.de")
]]

# prevent t2t datagen from downloading data and create dummy document
def create_dummy_tar(tmp_dir, dummy_file_name):
    dummy_file_path = os.path.join(tmp_dir, dummy_file_name)
    if not os.path.exists(dummy_file_path):
        tf.logging.info("Generating dummy file: %s" % dummy_file_path)
        tar_dummy = tarfile.open(dummy_file_path, "w:gz")
        tar_dummy.close()
    tf.logging.info("File %s is already exists or created" % dummy_file_name)

@registry.register_problem
class TranslateEndeCustom(text_problems.Text2TextProblem):
    
    # text generated is encoded with a vocabulary for training
    # by default it is a SubwordTextEncoder that is invertible
    # @see Text2TextProblem.vocab_type
    # here a vocabulary has approximately 8000 subwords
    @property
    def approx_vocab_size(self):
        return 2**13

    @property
    def oov_token(self):
        """Out of vocabulary token. Only for VocabType.TOKEN."""
    @property
    def is_generate_per_split(self):
        return False

    @property
    def source_vocab_name(self):
        return "vocab.ende-sub-en.%d" % self.approx_vocab_size

    @property
    def target_vocab_name(self):
        return "vocab.ende-sub-de.%d" % self.approx_vocab_size


    def get_training_dataset(self, tmp_dir):
        full_dataset = _NC_TRAIN_DATASETS
        return full_dataset

    # literally generate encoded samples.
    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        train_dataset = self.get_training_dataset(tmp_dir)
        datasets = train_dataset if train else _NC_TEST_DATASETS
        for item in datasets:
             dummy_file_name = item[0].split("/")[-1]
             create_dummy_tar(tmp_dir, dummy_file_name)
             s_file, t_file = item[1][0], item[1][1]
             if not os.path.exists(os.path.join(tmp_dir, s_file)):
                 raise Exception("Be sure file '%s' is exists in temp dir" % s_file)
             if not os.path.exists(os.path.join(tmp_dir, t_file)):
                 raise Exception("Be sure file '%s' is exists in tmp dir" % t_file)

        source_datasets = [[item[0], [item[1][0]]] for item in train_dataset]
        target_datasets = [[item[0], [item[1][1]]] for item in train_dataset]
        source_vocab = generator_utils.get_or_generate_vocab(
            data_dir,
            tmp_dir,
            self.source_vocab_name,
            self.approx_vocab_size,
            source_datasets,
            file_byte_budget=1e8)
        target_vocab = generator_utils.get_or_generate_vocab(
            data_dir,
            tmp_dir,
            self.target_vocab_name,
            self.approx_vocab_size,
            target_datasets,
            file_byte_budget=1e8)
        tag = "train" if train else "dev"
        filename_base = "wht_ende_%sk_sub_%s" % (self.approx_vocab_size, tag)
        data_path = translate.compile_data(tmp_dir, datasets, filename_base)
        return text_problems.text2text_generate_encoded(
            text_problems.text2text_txt_iterator(data_path + ".lang1",
                                                  data_path + ".lang2"),
            source_vocab, target_vocab)

    def feature_encoders(self, data_dir):
        source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
        target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
        source_token = text_encoder.SubwordTextEncoder(source_vocab_filename)
        target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
        return {
            "inputs": source_token,
            "targets": target_token,
        }

@registry.register_hparams
def transformer_zhen():
    hparams = transformer.transformer_base_single_gpu()
    hparams.num_hidden_layers = 2
    hparams.hidden_size = 128
    hparams.filter_size = 512
    hparams.attention_dropout = 0.6
    return hparams
