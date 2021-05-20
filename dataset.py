#encoding=utf-8
import os
import torch
import random
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset

from m2p_mask import process_train_MAM_data, mask_tokens

HALF_BATCHSIZE_TIME = 2000
HALF_BATCHSIZE_FRAMES = HALF_BATCHSIZE_TIME / 1

class BaseDataset(Dataset):
    def __init__(self, file_path, sets, bucket_size, max_timestep=0, drop=False):
        # Read file
        self.root = file_path
        tables = [pd.read_csv(os.path.join(file_path, s + '.csv')) for s in sets]
        self.table = pd.concat(tables, ignore_index=True).sort_values(by=['length'], ascending=False)
        # drop the results with some blank txt
        self.table = self.table.dropna(axis=0)

        # Crop seqs that are too long
        if drop and max_timestep > 0:
            self.table = self.table[self.table.length < max_timestep]

    def __len__(self):
        return len(self.X_a)


def load_acoustic_data(npy_path, npy_root=None):
    return torch.FloatTensor(np.load(os.path.join(npy_root, npy_path)))


def load_semantic_data(txt_path, txt_root=None, tokenizer=None):
    txt_content = ' '.join(
        [x.strip('\n').split(',')[2].lower() for x in open(os.path.join(txt_root, txt_path), 'r').readlines()])
    txt_content = tokenizer(txt_content)
    return txt_content


class SupervisedDataset(BaseDataset):
    def __init__(self, file_path, sets, bucket_size, max_timestep=0, drop=False,
                 acoustic_config=None, semantic_config=None, tokenizer=None, main_random_noise=False, mask_proportion=0.0):
        super().__init__(file_path, sets, bucket_size, max_timestep, drop)
        self.acoustic_config = acoustic_config
        self.semantic_config = semantic_config
        self.tokenizer = tokenizer
        self.main_random_noise = main_random_noise
        self.mask_proportion = mask_proportion
        self.sample_step = 0

        X_a = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        X_t = self.table['align_path'].tolist()
        # Use bucketing to allow different batch size at run time
        self.X_a, self.X_t = [], []
        batch_x_a, batch_len, batch_x_t = [], [], []

        for x_a, x_len, x_t in zip(X_a, X_lens, X_t):
            batch_x_a.append(x_a)
            batch_len.append(x_len)
            batch_x_t.append(x_t)

            # Fill in batch_x until batch is full
            if len(batch_x_a) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_step == 0:
                    self.X_a.append(batch_x_a[:bucket_size // 2])
                    self.X_a.append(batch_x_a[bucket_size // 2:])
                    self.X_t.append(batch_x_t[:bucket_size // 2])
                    self.X_t.append(batch_x_t[bucket_size // 2:])
                else:
                    self.X_a.append(batch_x_a)
                    self.X_t.append(batch_x_t)
                batch_x_a, batch_len, batch_x_t = [], [], []

        # Gather the last batch
        if len(batch_x_a) > 1:
            self.X_a.append(batch_x_a)
            self.X_t.append(batch_x_t)

        assert len(self.X_a) == len(self.X_t)

    def process_x_pad_batch(self, x_a_pad_batch, x_t_pad_batch):
        # preprocess with the acoustic inputs
        a_valid_batchid, a_inputs, a_mask_labels, a_attention_mask, a_labels = process_train_MAM_data(
            spec=(x_a_pad_batch,), mask_proportion=self.mask_proportion,
            config=self.acoustic_config, tail_masking=False, main_random=self.main_random_noise
        )
        # preprocess with the semantic inputs
        x_t_pad_batch = self.tokenizer.pad(x_t_pad_batch, return_tensors="pt")
        s_inputs, s_labels = mask_tokens(inputs=x_t_pad_batch['input_ids'], mlm_probability=self.mask_proportion,
                                         tokenizer=self.tokenizer, tail_masking=False, main_random=self.main_random_noise)

        s_attention_mask = x_t_pad_batch['attention_mask']
        s_valid_batchid = torch.nonzero(torch.sum(s_labels != -100, dim=1), as_tuple=False).view(-1)
        # ---------- process the valid batch id ----------#
        a_valid = torch.zeros(a_labels.size(0))
        a_valid[a_valid_batchid] = 1
        s_valid = torch.zeros(s_labels.size(0))
        s_valid[s_valid_batchid] = 1
        valid_batchid = a_valid.long() & s_valid.long()
        valid_batchid = torch.nonzero(valid_batchid, as_tuple=False).view(-1)
        # ---------- valid assertation ----------#
        batch_is_valid = len(valid_batchid) > 0
        # ---------- acoustic features ----------#
        a_inputs = a_inputs[valid_batchid]
        a_mask_labels = a_mask_labels[valid_batchid]
        a_attention_mask = a_attention_mask[valid_batchid]
        a_labels = a_labels[valid_batchid]
        # ---------- semantic features ----------#
        s_inputs = s_inputs[valid_batchid]
        s_attention_mask = s_attention_mask[valid_batchid]
        s_labels = s_labels[valid_batchid]
        text_raw = x_t_pad_batch['input_ids'][valid_batchid]

        return batch_is_valid, (a_inputs, a_mask_labels, a_attention_mask, a_labels), (
        s_inputs, s_attention_mask, s_labels, text_raw)

    def __getitem__(self, index):
        acoustic_batch = [load_acoustic_data(x_file, self.root) for x_file in self.X_a[index]]
        x_a_pad_batch = pad_sequence(acoustic_batch, batch_first=True, padding_value=0)
        semantic_batch = [load_semantic_data(x_file, self.root, self.tokenizer) for x_file in self.X_t[index]]
        x_t_pad_batch = dict()
        x_t_pad_batch['input_ids'] = [x['input_ids'] for x in semantic_batch]
        x_t_pad_batch['attention_mask'] = [x['attention_mask'] for x in semantic_batch]
        return self.process_x_pad_batch(x_a_pad_batch, x_t_pad_batch)



class DAEDataset(BaseDataset):
    def __init__(self, file_path, sets, bucket_size, max_timestep=0, drop=False,
                 acoustic_config=None, semantic_config=None, tokenizer=None, main_random_noise=False):
        super().__init__(file_path, sets, bucket_size, max_timestep, drop)
        self.acoustic_config = acoustic_config
        self.semantic_config = semantic_config
        self.tokenizer = tokenizer
        self.main_random_noise = main_random_noise
        self.sample_step = 0

        X_a = self.table['file_path'].tolist()
        X_lens = self.table['length'].tolist()
        X_t = self.table['align_path'].tolist()
        # Use bucketing to allow different batch size at run time
        self.X_a, self.X_t = [], []
        batch_x_a, batch_len, batch_x_t = [], [], []

        for x_a, x_len, x_t in zip(X_a, X_lens, X_t):
            batch_x_a.append(x_a)
            batch_len.append(x_len)
            batch_x_t.append(x_t)

            # Fill in batch_x until batch is full
            if len(batch_x_a) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max(batch_len) > HALF_BATCHSIZE_TIME) and self.sample_step == 0:
                    self.X_a.append(batch_x_a[:bucket_size // 2])
                    self.X_a.append(batch_x_a[bucket_size // 2:])
                    self.X_t.append(batch_x_t[:bucket_size // 2])
                    self.X_t.append(batch_x_t[bucket_size // 2:])
                else:
                    self.X_a.append(batch_x_a)
                    self.X_t.append(batch_x_t)
                batch_x_a, batch_len, batch_x_t = [], [], []

        # Gather the last batch
        if len(batch_x_a) > 1:
            self.X_a.append(batch_x_a)
            self.X_t.append(batch_x_t)

        assert len(self.X_a) == len(self.X_t)

    def process_x_pad_batch(self, x_a_pad_batch, x_t_pad_batch):
        # preprocess with the acoustic inputs
        a_valid_batchid, a_inputs, a_mask_labels, a_attention_mask, a_labels = process_train_MAM_data(
            spec=(x_a_pad_batch,), mask_proportion=self.acoustic_config["dae_mask_proportion"],
            config=self.acoustic_config, tail_masking=False, main_random=self.main_random_noise
        )
        # preprocess with the semantic inputs
        x_t_pad_batch = self.tokenizer.pad(x_t_pad_batch, return_tensors="pt")
        s_inputs, s_labels = mask_tokens(inputs=x_t_pad_batch['input_ids'], mlm_probability=self.semantic_config["dae_mask_proportion"],
                                         tokenizer=self.tokenizer, tail_masking=False, main_random=self.main_random_noise)

        s_attention_mask = x_t_pad_batch['attention_mask']
        s_valid_batchid = torch.nonzero(torch.sum(s_labels != -100, dim=1), as_tuple=False).view(-1)
        # ---------- process the valid batch id ----------#
        a_valid = torch.zeros(a_labels.size(0))
        a_valid[a_valid_batchid] = 1
        s_valid = torch.zeros(s_labels.size(0))
        s_valid[s_valid_batchid] = 1
        valid_batchid = a_valid.long() & s_valid.long()
        valid_batchid = torch.nonzero(valid_batchid, as_tuple=False).view(-1)
        # ---------- valid assertation ----------#
        batch_is_valid = len(valid_batchid) > 0
        # ---------- acoustic features ----------#
        a_inputs = a_inputs[valid_batchid]
        a_mask_labels = a_mask_labels[valid_batchid]
        a_attention_mask = a_attention_mask[valid_batchid]
        a_labels = a_labels[valid_batchid]
        # ---------- semantic features ----------#
        s_inputs = s_inputs[valid_batchid]
        s_attention_mask = s_attention_mask[valid_batchid]
        s_labels = s_labels[valid_batchid]
        text_raw = x_t_pad_batch['input_ids'][valid_batchid]

        return batch_is_valid, (a_inputs, a_mask_labels, a_attention_mask, a_labels), (
        s_inputs, s_attention_mask, s_labels, text_raw)

    def __getitem__(self, index):
        acoustic_batch = [load_acoustic_data(x_file, self.root) for x_file in self.X_a[index]]
        x_a_pad_batch = pad_sequence(acoustic_batch, batch_first=True, padding_value=0)
        semantic_batch = [load_semantic_data(x_file, self.root, self.tokenizer) for x_file in self.X_t[index]]
        x_t_pad_batch = dict()
        x_t_pad_batch['input_ids'] = [x['input_ids'] for x in semantic_batch]
        x_t_pad_batch['attention_mask'] = [x['attention_mask'] for x in semantic_batch]
        return self.process_x_pad_batch(x_a_pad_batch, x_t_pad_batch)



class Speech2TextDTDataset(Dataset):
    def __init__(self, speech2text_data, bucket_size,
                 acoustic_config=None, semantic_config=None, tokenizer=None, main_random_noise=False, mask_proportion=0.0):
        super().__init__()

        self.acoustic_config = acoustic_config
        self.semantic_config = semantic_config
        self.tokenizer = tokenizer
        self.main_random_noise = main_random_noise
        self.sample_step = 0
        self.mask_proportion = mask_proportion

        X_a = []
        X_t = []

        (speech_features, speech_mask), gen_text = speech2text_data

        assert len(speech_features) == len(speech_mask) == len(gen_text) # List of array:[batch, ...]

        for i in range(len(speech_features)):
            for bi in range(len(speech_features[i])):
                assert len(speech_features[i]) == len(speech_mask[i]) == len(gen_text[i])
                speech_feat = speech_features[i][bi]
                mask = speech_mask[i][bi]
                valid_len = np.argmin(mask, axis=0)
                if valid_len == 0 and np.sum(mask) > 0:
                    # mask全1的情况
                    valid_len = len(mask)

                X_a.append(speech_feat[:valid_len])

                t = gen_text[i][bi]  # [output_length, embedding_size]
                X_t.append(t)

        # Use bucketing to allow different batch size at run time
        self.X_a, self.X_t = [], []
        batch_x_a, batch_x_t = [], []

        max_batch_len = -1
        for x_a, x_t in zip(X_a, X_t):
            batch_x_a.append(x_a)
            batch_x_t.append(x_t)

            if len(x_a) > max_batch_len:
                max_batch_len = len(x_a)

            # Fill in batch_x until batch is full
            if len(batch_x_a) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max_batch_len > HALF_BATCHSIZE_FRAMES) and self.sample_step == 0:
                    self.X_a.append(batch_x_a[:bucket_size // 2])
                    self.X_a.append(batch_x_a[bucket_size // 2:])
                    self.X_t.append(batch_x_t[:bucket_size // 2])
                    self.X_t.append(batch_x_t[bucket_size // 2:])
                else:
                    self.X_a.append(batch_x_a)
                    self.X_t.append(batch_x_t)
                batch_x_a, batch_x_t = [], []
                max_batch_len = -1

        # Gather the last batch
        if len(batch_x_a) > 1:
            self.X_a.append(batch_x_a)
            self.X_t.append(batch_x_t)

        assert len(self.X_a) == len(self.X_t)


    def __len__(self):
        return len(self.X_a)


    def process_x_pad_batch(self, x_a_pad_batch, x_t_pad_batch):
        # preprocess with the acoustic inputs
        a_valid_batchid, a_inputs, a_mask_labels, a_attention_mask, a_labels = process_train_MAM_data(
            spec=(x_a_pad_batch,), mask_proportion=self.mask_proportion,
            config=self.acoustic_config, tail_masking=False, main_random=self.main_random_noise,
            do_downsampling=False
        )

        # preprocess with the semantic inputs
        s_inputs = x_t_pad_batch
        s_labels = x_t_pad_batch
        s_attention_mask = torch.ones_like(x_t_pad_batch[:, :, 0], dtype=torch.float)

        # ---------- process the valid batch id ----------#
        a_valid = torch.zeros(a_labels.size(0))
        a_valid[a_valid_batchid] = 1
        valid_batchid = a_valid.long()
        valid_batchid = torch.nonzero(valid_batchid, as_tuple=False).view(-1)
        # ---------- valid assertation ----------#
        batch_is_valid = len(valid_batchid) > 0
        # ---------- acoustic features ----------#
        a_inputs = a_inputs[valid_batchid]
        a_mask_labels = a_mask_labels[valid_batchid]
        a_attention_mask = a_attention_mask[valid_batchid]
        a_labels = a_labels[valid_batchid]
        # ---------- semantic features ----------#
        s_inputs = s_inputs[valid_batchid]
        s_attention_mask = s_attention_mask[valid_batchid]
        s_labels = s_labels[valid_batchid]
        x_t_pad_batch = x_t_pad_batch[valid_batchid]

        return batch_is_valid, (a_inputs, a_mask_labels, a_attention_mask, a_labels), (
            s_inputs, s_attention_mask, s_labels, x_t_pad_batch)

    def __getitem__(self, index):
        acoustic_batch = [torch.FloatTensor(x) for x in self.X_a[index]]
        x_a_pad_batch = pad_sequence(acoustic_batch, batch_first=True, padding_value=0)

        semantic_batch = torch.FloatTensor(self.X_t[index])

        return self.process_x_pad_batch(x_a_pad_batch, semantic_batch)



class Text2SpeechDTDataset(Dataset):
    def __init__(self, text2speech_data, bucket_size,
                 acoustic_config=None, semantic_config=None, tokenizer=None, main_random_noise=False, mask_proportion=0.0):
        super().__init__()

        self.acoustic_config = acoustic_config
        self.semantic_config = semantic_config
        self.tokenizer = tokenizer
        self.main_random_noise = main_random_noise
        self.sample_step = 0
        self.mask_proportion = mask_proportion

        X_a = []
        X_t = []

        (text, text_mask), gen_speech = text2speech_data

        assert len(text) == len(text_mask) == len(gen_speech) # List of array: [batch, ...]

        for i in range(len(text)):
            for bi in range(len(text[i])):
                assert len(text[i]) == len(text_mask[i]) == len(gen_speech[i])
                t = text[i][bi]
                mask = text_mask[i][bi]
                valid_len = np.argmin(mask, axis=0)
                if valid_len == 0 and np.sum(mask) > 0:
                    # mask全1的情况
                    valid_len = len(mask)

                X_t.append(t[:valid_len])

                speech_feat = gen_speech[i][bi]
                X_a.append(speech_feat)

        # Use bucketing to allow different batch size at run time
        self.X_a, self.X_t = [], []
        batch_x_a, batch_x_t = [], []

        max_batch_len = -1
        for x_a, x_t in zip(X_a, X_t):
            batch_x_a.append(x_a)
            batch_x_t.append(x_t)

            if len(x_a) > max_batch_len:
                max_batch_len = len(x_a)

            # Fill in batch_x until batch is full
            if len(batch_x_a) == bucket_size:
                # Half the batch size if seq too long
                if (bucket_size >= 2) and (max_batch_len > HALF_BATCHSIZE_FRAMES) and self.sample_step == 0:
                    self.X_a.append(batch_x_a[:bucket_size // 2])
                    self.X_a.append(batch_x_a[bucket_size // 2:])
                    self.X_t.append(batch_x_t[:bucket_size // 2])
                    self.X_t.append(batch_x_t[bucket_size // 2:])
                else:
                    self.X_a.append(batch_x_a)
                    self.X_t.append(batch_x_t)
                batch_x_a, batch_x_t = [], []
                max_batch_len = -1

        # Gather the last batch
        if len(batch_x_a) > 1:
            self.X_a.append(batch_x_a)
            self.X_t.append(batch_x_t)

        assert len(self.X_a) == len(self.X_t)


    def __len__(self):
        return len(self.X_a)


    def process_x_pad_batch(self, x_a_pad_batch, x_t_pad_batch):
        # preprocess with the acoustic inputs
        a_valid_batchid, a_inputs, a_mask_labels, a_attention_mask, a_labels = process_train_MAM_data(
            spec=(x_a_pad_batch,), mask_proportion=self.mask_proportion,
            config=self.acoustic_config, tail_masking=False, main_random=self.main_random_noise,
            do_downsampling=False
        )
        # preprocess with the semantic inputs
        x_t_pad_batch = self.tokenizer.pad(x_t_pad_batch, return_tensors="pt")
        s_inputs, s_labels = mask_tokens(inputs=x_t_pad_batch['input_ids'], mlm_probability=self.mask_proportion,
                                         tokenizer=self.tokenizer, tail_masking=False, main_random=self.main_random_noise)

        s_attention_mask = x_t_pad_batch['attention_mask']
        s_valid_batchid = torch.nonzero(torch.sum(s_labels != -100, dim=1), as_tuple=False).view(-1)
        # ---------- process the valid batch id ----------#
        a_valid = torch.zeros(a_labels.size(0))
        a_valid[a_valid_batchid] = 1
        s_valid = torch.zeros(s_labels.size(0))
        s_valid[s_valid_batchid] = 1
        valid_batchid = a_valid.long() & s_valid.long()
        valid_batchid = torch.nonzero(valid_batchid, as_tuple=False).view(-1)
        # ---------- valid assertation ----------#
        batch_is_valid = len(valid_batchid) > 0
        # ---------- acoustic features ----------#
        a_inputs = a_inputs[valid_batchid]
        a_mask_labels = a_mask_labels[valid_batchid]
        a_attention_mask = a_attention_mask[valid_batchid]
        a_labels = a_labels[valid_batchid]
        # ---------- semantic features ----------#
        s_inputs = s_inputs[valid_batchid]
        s_attention_mask = s_attention_mask[valid_batchid]
        s_labels = s_labels[valid_batchid]
        text_raw = x_t_pad_batch['input_ids'][valid_batchid]

        return batch_is_valid, (a_inputs, a_mask_labels, a_attention_mask, a_labels), (
        s_inputs, s_attention_mask, s_labels, text_raw)

    def __getitem__(self, index):
        acoustic_batch = [torch.FloatTensor(x) for x in self.X_a[index]]
        x_a_pad_batch = pad_sequence(acoustic_batch, batch_first=True, padding_value=0)

        semantic_batch = [x.tolist() for x in self.X_t[index]]
        x_t_pad_batch = dict()
        x_t_pad_batch['input_ids'] = [x for x in semantic_batch]
        x_t_pad_batch['attention_mask'] = [[1] * len(x) for x in semantic_batch]

        return self.process_x_pad_batch(x_a_pad_batch, x_t_pad_batch)






