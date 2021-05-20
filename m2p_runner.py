import os
import gc
import math
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm
from tensorboardX import SummaryWriter
from transformers import RobertaConfig

from models import EncoderDecoder
from dataset import Speech2TextDTDataset, Text2SpeechDTDataset, SupervisedDataset, DAEDataset
from m2p_optimization import BertAdam, Lamb, WarmupLinearSchedule


def calc_stop_loss(logits, speech_attention_mask, speech_label_mask):

    speech_label_mask = speech_label_mask[:, :, 0]

    index = speech_attention_mask.argmin(dim=1) #[batch]
    index = index - 1
    labels = torch.zeros_like(logits).to(logits.device) #[batch, seq_len]
    for bi in range(len(labels)):
        assert speech_label_mask[bi, index[bi]] == 1
        labels[bi, index[bi]] = 1.0

    pos_weight = torch.tensor([10.0], dtype=torch.float).to(logits.device)
    loss_fct = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    loss = loss_fct(logits, labels) #[batch, seq_len]

    loss = loss * speech_label_mask
    loss = loss.sum() / speech_label_mask.sum()

    return loss


class Runner():
    ''' Handler for complete pre-training progress of upstream models '''
    def __init__(self, args, config, dae_dataloader, tokenizer, ckpdir):
        
        self.device = torch.device('cuda') if (args.gpu and torch.cuda.is_available()) else torch.device('cpu')
        if torch.cuda.is_available(): print('[Runner] - CUDA is available!')
        self.model_kept = []
        self.global_step = 1
        self.log = SummaryWriter(ckpdir)

        self.args = args
        self.config = config
        self.dae_dataloader = dae_dataloader
        self.tokenizer = tokenizer

        self.ckpdir = ckpdir

        # optimizer
        self.learning_rate = float(config['optimizer']['learning_rate'])
        self.warmup_proportion = config['optimizer']['warmup_proportion']
        self.gradient_accumulation_steps = config['optimizer']['gradient_accumulation_steps']
        self.gradient_clipping = config['optimizer']['gradient_clipping']

        # Training details
        self.apex = config['runner']['apex']
        self.total_steps = config['runner']['total_steps']
        self.warm_up_epochs = config['runner']['warm_up_epochs']
        self.log_step = config['runner']['log_step']
        self.save_step = config['runner']['save_step']
        self.duo_feature = config['runner']['duo_feature']
        self.max_keep = config['runner']['max_keep']

        # Model configs
        self.text_encoder_config = RobertaConfig(**config['semantic'])
        self.text_encoder_config.is_decoder = False
        self.text_encoder_config.add_cross_attention = False
        self.text_decoder_config = RobertaConfig(**config['semantic'])
        self.text_decoder_config.is_decoder = True
        self.text_decoder_config.add_cross_attention = True

        self.speech_encoder_config = RobertaConfig(**config['acoustic'])
        self.speech_encoder_config.is_decoder = False
        self.speech_encoder_config.add_cross_attention = False
        self.speech_decoder_config = RobertaConfig(**config['acoustic'])
        self.speech_decoder_config.is_decoder = True
        self.speech_decoder_config.add_cross_attention = True

    def set_model(self):
        print('[Runner] - Initializing Transformer model...')

        # text是speech2text, speech是text2speech.
        self.text_model = EncoderDecoder(encoder_config=self.speech_encoder_config,
                                         decoder_config=self.text_decoder_config, modality="text")
        self.text_model.to(self.device)
        self.text_model.train()

        self.speech_model = EncoderDecoder(encoder_config=self.text_encoder_config,
                                         decoder_config=self.speech_decoder_config, modality="speech")
        self.speech_model.to(self.device)
        self.speech_model.train()

        if self.args.multi_gpu:
            self.text_model = torch.nn.DataParallel(self.text_model)
            self.speech_model = torch.nn.DataParallel(self.speech_model)
            print('[Runner] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))

        print('[Runner] - Number of parameters: ' + str(sum(p.numel() for p in self.text_model.parameters() if p.requires_grad) + \
                                                        sum(p.numel() for p in self.speech_model.parameters() if p.requires_grad)))

        param_optimizer = list(self.text_model.named_parameters()) + list(self.speech_model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        
        if 'type' not in self.config['optimizer']:
            self.config['optimizer']['type'] = 'adam'
        print('[Runner] - Optimizer: ' + ('apex Fused Adam' if self.apex else str(self.config['optimizer']['type'])))
        if self.apex:
            try:
                from apex.optimizers import FP16_Optimizer
                from apex.optimizers import FusedAdam
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            optimizer = FusedAdam(optimizer_grouped_parameters,
                                    lr=self.learning_rate,
                                    bias_correction=False,
                                    max_grad_norm=1.0)
            if self.config['optimizer']['loss_scale'] == 0:
                self.optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                self.optimizer = FP16_Optimizer(optimizer, static_loss_scale=self.config['optimizer']['loss_scale'])
            self.warmup_linear = WarmupLinearSchedule(warmup=self.warmup_proportion,
                                                      t_total=self.total_steps)
        elif self.config['optimizer']['type'] == 'adam':
            self.optimizer = BertAdam(optimizer_grouped_parameters,
                                      lr=self.learning_rate,
                                      warmup=self.warmup_proportion,
                                      t_total=self.total_steps,
                                      schedule='warmup_linear')
        elif self.config['optimizer']['type'] == 'lamb' or self.config['optimizer']['type'] == 'adamW':
            self.optimizer = Lamb(optimizer_grouped_parameters,
                                      lr=self.learning_rate,
                                      warmup=self.warmup_proportion,
                                      t_total=self.total_steps,
                                      schedule='warmup_linear',
                                      adam=True if self.config['optimizer']['type'] == 'adamW' else False,
                                      correct_bias=True if self.config['optimizer']['type'] == 'adamW' else False)
        else:
            raise NotImplementedError()
        
        if self.args.resume is not None:
            self.load_model(self.args.resume)

    def process_acoustic_data(self, acoustic_inputs):
        """Process training data for the masked acoustic model"""
        with torch.no_grad():
            
            assert(len(acoustic_inputs) == 4), 'dataloader should return (a_inputs, a_mask_labels, a_attn_mask, a_labels)'
            # Unpack and Hack bucket: Bucketing should cause acoustic feature to have shape 1xBxTxD'
            a_inputs = acoustic_inputs[0].squeeze(0)
            a_mask_labels = acoustic_inputs[1].squeeze(0)
            a_attention_mask = acoustic_inputs[2].squeeze(0)
            a_labels = acoustic_inputs[3].squeeze(0)

            a_inputs = a_inputs.float().to(device=self.device)
            a_mask_labels = a_mask_labels.bool().to(device=self.device)
            a_attention_mask = a_attention_mask.float().to(device=self.device)
            a_labels = a_labels.float().to(device=self.device)

        return a_inputs, a_mask_labels, a_attention_mask, a_labels

    def process_semantic_data(self, semantic_inputs):
        with torch.no_grad():
            
            assert(len(semantic_inputs) == 4), 'dataloader should return (s_inputs, s_attention_mask, s_labels, s_raw)'
            s_inputs = semantic_inputs[0].squeeze(0)
            s_attention_mask = semantic_inputs[1].squeeze(0)
            s_labels = semantic_inputs[2].squeeze(0)
            s_raw = semantic_inputs[3].squeeze(0)

            s_inputs = s_inputs.long().to(device=self.device)
            s_attention_mask = s_attention_mask.float().to(device=self.device)
            s_labels = s_labels.long().to(device=self.device)
            s_raw = s_raw.long().to(device=self.device)

        return s_inputs, s_attention_mask, s_labels, s_raw

    def load_model(self, ckptpth):
        ckpt = torch.load(ckptpth)
        self.text_model.load_state_dict(ckpt['semantic_model'])
        self.speech_model.load_state_dict(ckpt['acoustic_model'])
        self.optimizer.load_state_dict(ckpt['Optimizer'])
        self.global_step = ckpt['Global_step']

    def save_model(self, name='states', to_path=None):
        all_states = {
            'semantic_model': self.text_model.state_dict() if not self.args.multi_gpu else self.text_model.module.state_dict(),
            'acoustic_model': self.speech_model.state_dict() if not self.args.multi_gpu else self.speech_model.module.state_dict(),
        }
        all_states['Optimizer'] = self.optimizer.state_dict()
        all_states['Global_step'] = self.global_step
        all_states['Settings'] = { 'Config': self.config, 'Paras': self.args }

        if to_path is None:
            new_model_path = '{}/{}-{}.ckpt'.format(self.ckpdir, name, self.global_step)
        else:
            new_model_path = to_path

        torch.save(all_states, new_model_path)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def train(self,):

        print("Start warm up with parallel data.")

        warmup_dataset = SupervisedDataset(file_path=self.config['dataloader']['data_path'],
                                        sets=self.config['dataloader']['sup_train_set'],
                                        bucket_size=self.config['dataloader']['batch_size'],
                                        max_timestep=self.config['dataloader']['max_timestep'],
                                        drop=True, acoustic_config=self.config['acoustic'],
                                        semantic_config=self.config['semantic'],
                                        tokenizer=self.tokenizer, main_random_noise=False, mask_proportion=1.0)  #全部mask成[MASK]

        warmup_dataloader = DataLoader(dataset=warmup_dataset, batch_size=1, shuffle=True, drop_last=False,
                                    num_workers=self.config['dataloader']['n_jobs'], pin_memory=True)

        tk0 = tqdm(range(self.warm_up_epochs), total=self.warm_up_epochs, desc="warm up training with parallel data.")
        for _ in tk0:

            accum_step = 0
            accum_text_sup_loss = 0
            accum_speech_sup_loss = 0

            for warmup_batch in warmup_dataloader:
                warmup_batch_is_valid, warmup_speech_batch, warmup_text_batch = warmup_batch

                if not warmup_batch_is_valid:
                    continue

                speech_inputs, speech_mask_labels, speech_attention_mask, speech_labels = self.process_acoustic_data(
                    warmup_speech_batch)
                text_inputs, text_attention_mask, text_labels, text_raw = self.process_semantic_data(warmup_text_batch)

                text_sup_loss = self.text_model(
                    encoder_inputs=speech_labels,
                    encoder_attention_mask=speech_attention_mask,
                    decoder_inputs=text_inputs,
                    decoder_attention_mask=text_attention_mask,
                    decoder_labels=text_labels
                )

                speech_sup_loss = self.speech_model(
                    encoder_inputs=text_raw,
                    encoder_attention_mask=text_attention_mask,
                    decoder_inputs=speech_inputs,
                    decoder_attention_mask=speech_attention_mask,
                    decoder_labels=(speech_labels, speech_mask_labels)
                )

                loss = text_sup_loss + speech_sup_loss
                if self.args.multi_gpu:
                    loss = loss.mean()
                    text_sup_loss = text_sup_loss.mean()
                    speech_sup_loss = speech_sup_loss.mean()

                loss.backward()

                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(self.text_model.parameters()) + list(self.speech_model.parameters()),
                    self.gradient_clipping)

                self.optimizer.step()
                self.optimizer.zero_grad()

                batch_size = text_inputs.size(0)
                accum_step += batch_size
                accum_text_sup_loss += text_sup_loss.item() * batch_size
                accum_speech_sup_loss += speech_sup_loss.item() * batch_size

            tk0.set_postfix(text_loss=accum_text_sup_loss / accum_step, speech_loss=accum_speech_sup_loss / accum_step)

        del warmup_dataset, warmup_dataloader
        gc.collect()

        previous_speech2text_dataloader = None
        previous_text2speech_dataloader = None

        epoch = 0

        pbar = tqdm(total=self.total_steps)
        pbar.n = self.global_step - 1
        while self.global_step <= self.total_steps:

            print("\nStart Generation. Epoch: {}.\n".format(epoch))
            self.text_model.eval()
            self.speech_model.eval()

            if previous_speech2text_dataloader is None or previous_text2speech_dataloader is None:

                gen_dataset = DAEDataset(file_path=self.config['dataloader']['data_path'],
                                         sets=self.config['dataloader']['sup_train_set'] + self.config['dataloader'][
                                             'dt_train_set'],
                                         bucket_size=self.config['dataloader']['batch_size'],
                                         max_timestep=self.config['dataloader']['max_timestep'],
                                         drop=True, acoustic_config=self.config['acoustic'],
                                         semantic_config=self.config['semantic'],
                                         tokenizer=self.tokenizer, main_random_noise=False)

                gen_dataloader = DataLoader(dataset=gen_dataset, batch_size=1, shuffle=True, drop_last=False,
                                            num_workers=self.config['dataloader']['n_jobs'], pin_memory=True)

            all_speech = []
            all_speech_mask = []
            all_gen_text = []

            all_text = []
            all_text_mask = []
            all_gen_speech = []

            with torch.no_grad():
                if previous_speech2text_dataloader is None or previous_text2speech_dataloader is None:
                    for gen_batch in tqdm(gen_dataloader, desc="Generating First Time."):
                        gen_batch_is_valid, gen_speech_batch, gen_text_batch = gen_batch
                        if not gen_batch_is_valid:
                            continue

                        # 这里的speech和text不配对。
                        speech_inputs, speech_mask_labels, speech_attention_mask, speech_labels = self.process_acoustic_data(gen_speech_batch)
                        text_inputs, text_attention_mask, text_labels, text_raw = self.process_semantic_data(gen_text_batch)

                        batch_size = speech_labels.size(0)
                        text_mask_inputs = torch.ones((batch_size, self.text_decoder_config.max_output_length), dtype=torch.long).to(self.device) * \
                            self.tokenizer.mask_token_id

                        output_text = self.text_model(
                                encoder_inputs=speech_labels,
                                encoder_attention_mask=speech_attention_mask,
                                decoder_inputs=text_mask_inputs,
                            )

                        all_speech.append(speech_labels.detach().cpu().numpy())
                        all_speech_mask.append(speech_attention_mask.detach().cpu().numpy())
                        all_gen_text.append(output_text.detach().cpu().numpy())

                        batch_size = text_raw.size(0)
                        speech_mask_inputs = torch.zeros(
                            (batch_size, self.speech_decoder_config.max_output_length,
                             self.speech_decoder_config.audio_size * self.speech_decoder_config.downsample_rate),
                            dtype=torch.float
                        ).to(self.device)

                        output_speech = self.speech_model(
                                encoder_inputs=text_raw,
                                encoder_attention_mask=text_attention_mask,
                                decoder_inputs=speech_mask_inputs,
                            )

                        all_text.append(text_raw.detach().cpu().numpy())
                        all_text_mask.append(text_attention_mask.detach().cpu().numpy())
                        all_gen_speech.append(output_speech.detach().cpu().numpy())

                    del gen_dataset, gen_dataloader
                    gc.collect()

                else:
                    for gen_batch in tqdm(previous_speech2text_dataloader, desc="Generating Text."):
                        gen_batch_is_valid, gen_speech_batch, gen_text_batch = gen_batch
                        if not gen_batch_is_valid:
                            continue

                        # 这里的speech和text是配对的。
                        speech_inputs, speech_mask_labels, speech_attention_mask, speech_labels = self.process_acoustic_data(
                            gen_speech_batch)
                        text_inputs, text_attention_mask, text_labels, text_raw = self.process_semantic_data(
                            gen_text_batch)

                        output_text = self.text_model(
                            encoder_inputs=speech_labels,
                            encoder_attention_mask=speech_attention_mask,
                            decoder_inputs_embeds=text_raw,  # 上一轮生成的结果，是token embeds形式。
                        )

                        all_speech.append(speech_labels.detach().cpu().numpy())
                        all_speech_mask.append(speech_attention_mask.detach().cpu().numpy())
                        all_gen_text.append(output_text.detach().cpu().numpy())

                    del speech2text_dt_dataset, speech2text_dt_dataloader, previous_speech2text_dataloader
                    gc.collect()

                    for gen_batch in tqdm(previous_text2speech_dataloader, desc="Generating Speech."):
                        gen_batch_is_valid, gen_speech_batch, gen_text_batch = gen_batch
                        if not gen_batch_is_valid:
                            continue

                        # 这里的speech和text是配对的。
                        speech_inputs, speech_mask_labels, speech_attention_mask, speech_labels = self.process_acoustic_data(
                            gen_speech_batch)
                        text_inputs, text_attention_mask, text_labels, text_raw = self.process_semantic_data(
                            gen_text_batch)

                        output_speech = self.speech_model(
                            encoder_inputs=text_raw,
                            encoder_attention_mask=text_attention_mask,
                            decoder_inputs=speech_labels,  # 上一轮生成的结果，是mel spec形式。
                        )

                        all_text.append(text_raw.detach().cpu().numpy())
                        all_text_mask.append(text_attention_mask.detach().cpu().numpy())
                        all_gen_speech.append(output_speech.detach().cpu().numpy())

                    del text2speech_dt_dataset, text2speech_dt_dataloader, previous_text2speech_dataloader
                    gc.collect()

            speech2text = ((all_speech, all_speech_mask), all_gen_text)
            text2speech = ((all_text, all_text_mask), all_gen_speech)

            current_epoch_dt_mask_prop = min(max(self.config['semantic']['dt_mask_proportion'],
                                                  self.config['acoustic']['dt_mask_proportion']),
                                              0.3 + 0.01 * epoch)

            speech2text_dt_dataset = Speech2TextDTDataset(speech2text,
                                                          bucket_size=self.config['dataloader']['batch_size'],
                                                          acoustic_config=self.config['acoustic'],
                                                          semantic_config=self.config['semantic'],
                                                          tokenizer=self.tokenizer,
                                                          main_random_noise=False,
                                                          mask_proportion=current_epoch_dt_mask_prop)

            speech2text_dt_dataloader = DataLoader(dataset=speech2text_dt_dataset, batch_size=1, shuffle=True, drop_last=False,
                            num_workers=self.config['dataloader']['n_jobs'], pin_memory=True)


            text2speech_dt_dataset = Text2SpeechDTDataset(text2speech,
                                                          bucket_size=self.config['dataloader']['batch_size'],
                                                          acoustic_config=self.config['acoustic'],
                                                          semantic_config=self.config['semantic'],
                                                          tokenizer=self.tokenizer,
                                                          main_random_noise=False,
                                                          mask_proportion=current_epoch_dt_mask_prop)

            text2speech_dt_dataloader = DataLoader(dataset=text2speech_dt_dataset, batch_size=1, shuffle=True,
                                                   drop_last=False,
                                                   num_workers=self.config['dataloader']['n_jobs'], pin_memory=True)

            previous_speech2text_dataloader = speech2text_dt_dataloader
            previous_text2speech_dataloader = text2speech_dt_dataloader

            del speech2text, text2speech
            gc.collect()

            current_epoch_sup_mask_prop = min(max(self.config['semantic']['sup_mask_proportion'],
                                                  self.config['acoustic']['sup_mask_proportion']),
                                              0.3 + 0.01 * epoch)

            sup_dataset = SupervisedDataset(file_path=self.config['dataloader']['data_path'],
                                               sets=self.config['dataloader']['sup_train_set'],
                                               bucket_size=self.config['dataloader']['batch_size'],
                                               max_timestep=self.config['dataloader']['max_timestep'],
                                               drop=True, acoustic_config=self.config['acoustic'],
                                               semantic_config=self.config['semantic'],
                                               tokenizer=self.tokenizer, main_random_noise=False,
                                               mask_proportion=current_epoch_sup_mask_prop)

            sup_dataloader = DataLoader(dataset=sup_dataset, batch_size=1, shuffle=True, drop_last=False,
                                           num_workers=self.config['dataloader']['n_jobs'], pin_memory=True)

            ##################################################

            progress = tqdm(self.dae_dataloader, desc="Main Training Iteration.")

            s2t_dt_iter = speech2text_dt_dataloader.__iter__()
            t2s_dt_iter = text2speech_dt_dataloader.__iter__()
            sup_iter = sup_dataloader.__iter__()

            loss_val = 0
            speech_dt_loss_val, text_dt_loss_val, speech_dt_stop_loss_val = 0, 0, 0
            speech_sup_loss_val, text_sup_loss_val, speech_sup_stop_loss_val = 0, 0, 0
            speech_dae_loss_val, text_dae_loss_val, speech_dae_stop_loss_val = 0, 0, 0

            self.text_model.train()
            self.speech_model.train()

            for dae_batch in progress:

                try:
                    s2t_dt_batch = next(s2t_dt_iter)
                except StopIteration:
                    del s2t_dt_iter
                    gc.collect()

                    s2t_dt_iter = speech2text_dt_dataloader.__iter__()
                    s2t_dt_batch = next(s2t_dt_iter)

                try:
                    t2s_dt_batch = next(t2s_dt_iter)
                except StopIteration:
                    del t2s_dt_iter
                    gc.collect()

                    t2s_dt_iter = text2speech_dt_dataloader.__iter__()
                    t2s_dt_batch = next(t2s_dt_iter)

                try:
                    sup_batch = next(sup_iter)
                except StopIteration:
                    del sup_iter
                    gc.collect()

                    sup_iter = sup_dataloader.__iter__()
                    sup_batch = next(sup_iter)

                dae_batch_is_valid, dae_speech_batch, dae_text_batch = dae_batch
                s2t_dt_batch_is_valid, s2t_dt_speech_batch, s2t_dt_text_batch = s2t_dt_batch
                t2s_dt_batch_is_valid, t2s_dt_speech_batch, t2s_dt_text_batch = t2s_dt_batch
                sup_batch_is_valid, sup_speech_batch, sup_text_batch = sup_batch

                try:
                    if self.global_step > self.total_steps: break
                    if not s2t_dt_batch_is_valid or not t2s_dt_batch_is_valid or not dae_batch_is_valid or not sup_batch_is_valid:
                        continue

                    ######## Dual Transformation ######
                    # 数据集不能混在一起。得分为两部分，生成的文本和原始的音频用来还原音频。vise versa.
                    # 生成的进encoder, 真实的进decoder.

                    speech_inputs, speech_mask_labels, speech_attention_mask, speech_labels = self.process_acoustic_data(t2s_dt_speech_batch)
                    text_inputs, text_attention_mask, text_labels, text_raw = self.process_semantic_data(t2s_dt_text_batch)

                    text_dt_loss = self.text_model(
                        encoder_inputs=speech_labels, # 生成的。
                        encoder_attention_mask=speech_attention_mask,
                        decoder_inputs=text_inputs,
                        decoder_attention_mask=text_attention_mask,
                        decoder_labels=text_labels
                    )

                    speech_inputs, speech_mask_labels, speech_attention_mask, speech_labels = self.process_acoustic_data(
                        s2t_dt_speech_batch)
                    text_inputs, text_attention_mask, text_labels, text_raw = self.process_semantic_data(
                        s2t_dt_text_batch)

                    speech_dt_loss = self.speech_model(
                        encoder_inputs_embeds=text_raw,  # 生成的。
                        encoder_attention_mask=text_attention_mask,
                        decoder_inputs=speech_inputs,
                        decoder_attention_mask=speech_attention_mask,
                        decoder_labels=(speech_labels, speech_mask_labels)
                    )

                    ######## Supervised #######

                    speech_inputs, speech_mask_labels, speech_attention_mask, speech_labels = self.process_acoustic_data(
                        sup_speech_batch)
                    text_inputs, text_attention_mask, text_labels, text_raw = self.process_semantic_data(sup_text_batch)

                    text_sup_loss = self.text_model(
                        encoder_inputs=speech_inputs,
                        encoder_attention_mask=speech_attention_mask,
                        decoder_inputs=text_inputs,
                        decoder_attention_mask=text_attention_mask,
                        decoder_labels=text_labels
                    )

                    speech_sup_loss = self.speech_model(
                        encoder_inputs=text_inputs,
                        encoder_attention_mask=text_attention_mask,
                        decoder_inputs=speech_inputs,
                        decoder_attention_mask=speech_attention_mask,
                        decoder_labels=(speech_labels, speech_mask_labels)
                    )

                    ######## Denoise AutoEncoding ########

                    speech_inputs, speech_mask_labels, speech_attention_mask, speech_labels = self.process_acoustic_data(
                        dae_speech_batch)
                    text_inputs, text_attention_mask, text_labels, text_raw = self.process_semantic_data(dae_text_batch)

                    text_dae_loss = self.text_model(
                        encoder_inputs=speech_inputs,
                        encoder_attention_mask=speech_attention_mask,
                        encoder_labels=(speech_labels, speech_mask_labels)
                    )

                    speech_dae_loss = self.speech_model(
                        encoder_inputs=text_inputs,
                        encoder_attention_mask=text_attention_mask,
                        encoder_labels=text_labels
                    )

                    #######################################

                    if self.args.multi_gpu:
                        text_dt_loss = text_dt_loss.mean()
                        speech_dt_loss = speech_dt_loss.mean()

                        text_sup_loss = text_sup_loss.mean()
                        speech_sup_loss = speech_sup_loss.mean()

                        text_dae_loss = text_dae_loss.mean()
                        speech_dae_loss = speech_dae_loss.mean()

                    loss = (text_dt_loss + speech_dt_loss) + \
                           0.1 * (text_sup_loss + speech_sup_loss) + \
                           (text_dae_loss + speech_dae_loss)

                    # Accumulate Loss
                    if self.gradient_accumulation_steps > 1:
                        loss = loss / self.gradient_accumulation_steps

                    if self.apex and self.args.multi_gpu:
                        raise NotImplementedError
                    elif self.apex:
                        self.optimizer.backward(loss)
                    else:
                        loss.backward()
                    
                    loss_val += loss.item()

                    speech_dt_loss_val += speech_dt_loss.item()
                    text_dt_loss_val += text_dt_loss.item()

                    speech_sup_loss_val += speech_sup_loss.item()
                    text_sup_loss_val += text_sup_loss.item()

                    speech_dae_loss_val += speech_dae_loss.item()
                    text_dae_loss_val += text_dae_loss.item()


                    if (self.total_steps+1) % self.gradient_accumulation_steps == 0:
                        if self.apex:
                            # modify learning rate with special warm up BERT uses
                            # if conifg.apex is False, BertAdam is used and handles this automatically
                            lr_this_step = self.learning_rate * self.warmup_linear.get_lr(self.global_step, self.warmup_proportion)
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = lr_this_step

                        # Step
                        grad_norm = torch.nn.utils.clip_grad_norm_(list(self.text_model.parameters())+list(self.speech_model.parameters()),
                                                                   self.gradient_clipping)
                        if math.isnan(grad_norm):
                            print('[Runner] - Error : grad norm is NaN @ step ' + str(self.global_step))
                        else:
                            self.optimizer.step()
                        self.optimizer.zero_grad()

                        if self.global_step % self.log_step == 0:
                            # Log
                            self.log.add_scalar('lr', self.optimizer.get_lr()[0], self.global_step)
                            self.log.add_scalar('loss', (loss_val), self.global_step)
                            self.log.add_scalar('speech_dt_loss', (speech_dt_loss_val), self.global_step)
                            self.log.add_scalar('text_dt_loss', (text_dt_loss_val), self.global_step)
                            self.log.add_scalar('speech_dt_stop_loss', (speech_dt_stop_loss_val), self.global_step)
                            self.log.add_scalar('speech_sup_loss', (speech_sup_loss_val), self.global_step)
                            self.log.add_scalar('text_sup_loss', (text_sup_loss_val), self.global_step)
                            self.log.add_scalar('speech_sup_stop_loss', (speech_sup_stop_loss_val), self.global_step)
                            self.log.add_scalar('speech_dae_loss', (speech_dae_loss_val), self.global_step)
                            self.log.add_scalar('text_dae_loss', (text_dae_loss_val), self.global_step)
                            self.log.add_scalar('speech_dae_stop_loss', (speech_dae_stop_loss_val), self.global_step)
                            self.log.add_scalar('gradient norm', grad_norm, self.global_step)

                        progress.set_description("Loss {:.4f} - DT Loss {:.4f} - SUP Loss {:.4f} - DAE Loss {:.4f}".format(loss_val,
                                                (speech_dt_loss_val + text_dt_loss_val + speech_dt_stop_loss_val),
                                                (speech_sup_loss_val + text_sup_loss_val + speech_sup_stop_loss_val),
                                                (speech_dae_loss_val + text_dae_loss_val + speech_dae_stop_loss_val)))

                        if self.global_step % self.save_step == 0:
                            self.save_model('states')

                        loss_val = 0
                        speech_dt_loss_val, text_dt_loss_val, speech_dt_stop_loss_val = 0, 0, 0
                        speech_sup_loss_val, text_sup_loss_val, speech_sup_stop_loss_val = 0, 0, 0
                        speech_dae_loss_val, text_dae_loss_val, speech_dae_stop_loss_val = 0, 0, 0

                        pbar.update(1)
                        self.global_step += 1

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print('CUDA out of memory at step: ', self.global_step)
                        torch.cuda.empty_cache()
                        self.optimizer.zero_grad()
                    else:
                        raise

            epoch += 1

            del sup_dataset, sup_dataloader, sup_iter, s2t_dt_iter, t2s_dt_iter
            gc.collect()

        pbar.close()
        self.log.close()


