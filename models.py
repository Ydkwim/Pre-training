#encoding=utf-8
import torch.nn.functional as F
from modules import *


class EncoderDecoder(nn.Module):
    def __init__(self, encoder_config, decoder_config, modality):
        super().__init__()

        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        self.return_dict = (self.encoder_config.use_return_dict and self.decoder_config.use_return_dict)

        self.modality = modality

        # text是speech2text, speech是text2speech.
        if modality == "text":
            self.encoder = RobertaForMaskedAM(encoder_config)
            self.decoder = RobertaForMaskedLM(decoder_config)
        elif modality == "speech":
            self.encoder = RobertaForMaskedLM(encoder_config)
            self.decoder = RobertaForMaskedAM(decoder_config)
        else:
            raise ValueError("Modality: {}, is not support.".format(modality))

    def forward(self,
                encoder_inputs=None,
                encoder_inputs_embeds=None,
                encoder_attention_mask=None,
                encoder_token_type_ids=None,
                encoder_position_ids=None,
                encoder_head_mask=None,
                encoder_labels=None,
                decoder_inputs=None,
                decoder_inputs_embeds=None,
                decoder_attention_mask=None,
                decoder_token_type_ids=None,
                decoder_position_ids=None,
                decoder_head_mask=None,
                decoder_labels=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=None):

        return_dict = return_dict if return_dict is not None else self.return_dict

        if encoder_inputs is not None and encoder_inputs_embeds is not None:
            raise ValueError("You cannot specify both encoder_inputs and encoder_inputs_embeds at the same time")

        encoder_outputs = self.encoder(
            input_ids=encoder_inputs,
            inputs_embeds=encoder_inputs_embeds,
            attention_mask=encoder_attention_mask,
            token_type_ids=encoder_token_type_ids,
            position_ids=encoder_position_ids,
            head_mask=encoder_head_mask,
            labels=encoder_labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Take the last layers' hidden
        encoder_hidden_states = encoder_outputs.hidden_states[-1]

        if decoder_inputs is None and decoder_inputs_embeds is None and encoder_labels is not None:
            # 不进入decoder，为了算dae。
            return encoder_outputs.loss

        if decoder_inputs is not None and decoder_inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_inputs and decoder_inputs_embeds at the same time")

        decoder_outputs = self.decoder(
            input_ids=decoder_inputs,
            inputs_embeds=decoder_inputs_embeds,
            attention_mask=decoder_attention_mask,
            token_type_ids=decoder_token_type_ids,
            position_ids=decoder_position_ids,
            head_mask=decoder_head_mask,
            labels=decoder_labels,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        decoder_hidden_states = decoder_outputs.hidden_states[-1]

        # 生成以及下游任务会到这里
        if decoder_labels is None:
            return decoder_outputs.logits

        # dt和sup会到这。
        return decoder_outputs.loss
















def generate_text_batch(model,
                   encoder_inputs=None,
                   encoder_attention_mask=None,
                   encoder_token_type_ids=None,
                   encoder_position_ids=None,
                   encoder_head_mask=None,
                   output_attentions=False,
                   output_hidden_states=True,
                   return_dict=None,
                   decoder_mask_input=None,
                   max_length=None,
                   ):

    assert model.modality == "text"

    return_dict = return_dict if return_dict is not None else model.return_dict

    batch_size = encoder_inputs.size(0)

    model.eval()

    encoder = torch.nn.DataParallel(model.encoder)
    encoder.eval()

    decoder = torch.nn.DataParallel(model.decoder)
    decoder.eval()

    with torch.no_grad():

        encoder_outputs = encoder(
            encoder_inputs,
            attention_mask=encoder_attention_mask,
            token_type_ids=encoder_token_type_ids,
            position_ids=encoder_position_ids,
            head_mask=encoder_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Take the last layers' hidden
        encoder_hidden_states = encoder_outputs.hidden_states[-1]

        # Decoding
        decoder_mask_input = decoder_mask_input.reshape((1, 1)).expand(batch_size, -1) #[batch, 1]

        outputs = [torch.zeros((batch_size, 1), dtype=torch.long).to(encoder_inputs.device)] # bos

        for _ in range(max_length-3):
            decoder_inputs = torch.cat(outputs + [decoder_mask_input], dim=1)
            output = decoder(
                decoder_inputs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            next_step_logits = output.logits[:, -1, :]  #[batch, vocab_size]
            top1 = next_step_logits.argmax(1).unsqueeze(1)  #[batch, 1]
            outputs.append(top1)

    return torch.cat(outputs, dim=1)


def generate_speech_batch(model,
                   encoder_inputs=None,
                   encoder_attention_mask=None,
                   encoder_token_type_ids=None,
                   encoder_position_ids=None,
                   encoder_head_mask=None,
                   output_attentions=False,
                   output_hidden_states=True,
                   return_dict=None,
                   decoder_mask_input=None,
                   max_length=None,
                   ):

    assert model.modality == "speech"

    return_dict = return_dict if return_dict is not None else model.return_dict

    batch_size = encoder_inputs.size(0)

    model.eval()

    encoder = torch.nn.DataParallel(model.encoder)
    encoder.eval()

    decoder = torch.nn.DataParallel(model.decoder)
    decoder.eval()

    with torch.no_grad():

        encoder_outputs = encoder(
            encoder_inputs,
            attention_mask=encoder_attention_mask,
            token_type_ids=encoder_token_type_ids,
            position_ids=encoder_position_ids,
            head_mask=encoder_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Take the last layers' hidden
        encoder_hidden_states = encoder_outputs.hidden_states[-1]

        # Decoding
        decoder_mask_input = decoder_mask_input.reshape((1, 1, -1)).expand(batch_size, -1, -1) #[batch, 1, feature_size]

        decoder_inputs = decoder_mask_input
        outputs = []
        decoder_hidden_states = []

        for _ in range(max_length):
            output = decoder(
                decoder_inputs,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            next_step_logits = output.logits[:, -1, :]  #[batch, audio_input_size]
            top1 = next_step_logits.unsqueeze(1)  #[batch, 1, audio_input_size]
            outputs.append(top1)
            decoder_inputs = torch.cat(outputs+[decoder_mask_input], dim=1)
            decoder_hidden_states.append(output.hidden_states[-1][:, -1, :])

        outputs = torch.cat(outputs, dim=1)
        decoder_hidden_states = torch.stack(decoder_hidden_states, dim=1)  #[batch, max_len, hidden_size]
        stop_index = model.speech_stop_linear(decoder_hidden_states)

    return outputs, stop_index














class MultiModalEncoderDecoder(nn.Module):
    def __init__(self, ckpt_path, num_classes, pool_type="attention"):
        super().__init__()

        self.pool_type = pool_type

        # First reinitialize the model
        ckpt_states = torch.load(ckpt_path, map_location='cpu')

        # Model configs
        self.text_encoder_config = RobertaConfig(**ckpt_states['Settings']['Config']['semantic'])
        self.text_encoder_config.is_decoder = False
        self.text_encoder_config.add_cross_attention = False
        self.text_decoder_config = RobertaConfig(**ckpt_states['Settings']['Config']['semantic'])
        self.text_decoder_config.is_decoder = True
        self.text_decoder_config.add_cross_attention = True

        self.speech_encoder_config = RobertaConfig(**ckpt_states['Settings']['Config']['acoustic'])
        self.speech_encoder_config.is_decoder = False
        self.speech_encoder_config.add_cross_attention = False
        self.speech_decoder_config = RobertaConfig(**ckpt_states['Settings']['Config']['acoustic'])
        self.speech_decoder_config.is_decoder = True
        self.speech_decoder_config.add_cross_attention = True

        # text是speech2text, speech是text2speech.
        self.text_model = EncoderDecoder(encoder_config=self.speech_encoder_config,
                                         decoder_config=self.text_decoder_config, modality="text")

        self.speech_model = EncoderDecoder(encoder_config=self.text_encoder_config,
                                           decoder_config=self.speech_decoder_config, modality="speech")

        # load the model from pretrained states
        self.text_model.load_state_dict(ckpt_states['semantic_model'])
        self.speech_model.load_state_dict(ckpt_states['acoustic_model'])

        if pool_type == "attention":
            self.speech_att_linear = nn.Linear(self.speech_decoder_config.hidden_size, self.speech_decoder_config.hidden_size)
            self.speech_att_attention = nn.Linear(self.speech_decoder_config.hidden_size, 1)

        in_features = self.text_decoder_config.hidden_size + self.speech_decoder_config.hidden_size

        self.fuse_linear = nn.Sequential(
            nn.Linear(in_features, int(in_features // 2)), nn.ReLU(), nn.Dropout(p=0.2),
            nn.Linear(int(in_features // 2), int(in_features // 2)), nn.ReLU(), nn.Dropout(p=0.2)
        )

        self.output_layer = nn.Linear(int(in_features // 2), num_classes)

    def forward(self,
                text_encoder_inputs=None,
                text_encoder_attention_mask=None,
                text_encoder_token_type_ids=None,
                text_encoder_position_ids=None,
                text_encoder_head_mask=None,
                text_decoder_inputs=None,
                text_decoder_attention_mask=None,
                text_decoder_token_type_ids=None,
                text_decoder_position_ids=None,
                text_decoder_head_mask=None,
                speech_encoder_inputs=None,
                speech_encoder_attention_mask=None,
                speech_encoder_token_type_ids=None,
                speech_encoder_position_ids=None,
                speech_encoder_head_mask=None,
                speech_decoder_inputs=None,
                speech_decoder_attention_mask=None,
                speech_decoder_token_type_ids=None,
                speech_decoder_position_ids=None,
                speech_decoder_head_mask=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=None):

        return_dict = return_dict if return_dict is not None else (
                    self.text_model.return_dict and self.speech_model.return_dict)

        text_outputs = self.text_model(
            encoder_inputs=text_encoder_inputs,
            encoder_attention_mask=text_encoder_attention_mask,
            encoder_token_type_ids=text_encoder_token_type_ids,
            encoder_position_ids=text_encoder_position_ids,
            encoder_head_mask=text_encoder_head_mask,
            decoder_inputs=text_decoder_inputs,
            decoder_attention_mask=text_decoder_attention_mask,
            decoder_token_type_ids=text_decoder_token_type_ids,
            decoder_position_ids=text_decoder_position_ids,
            decoder_head_mask=text_decoder_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        text_hidden_states = text_outputs  #[batch, seq_len, hidden_size]

        speech_outputs = self.speech_model(
            encoder_inputs=speech_encoder_inputs,
            encoder_attention_mask=speech_encoder_attention_mask,
            encoder_token_type_ids=speech_encoder_token_type_ids,
            encoder_position_ids=speech_encoder_position_ids,
            encoder_head_mask=speech_encoder_head_mask,
            decoder_inputs=speech_decoder_inputs,
            decoder_attention_mask=speech_decoder_attention_mask,
            decoder_token_type_ids=speech_decoder_token_type_ids,
            decoder_position_ids=speech_decoder_position_ids,
            decoder_head_mask=speech_decoder_head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        speech_hidden_states = speech_outputs

        text_decoder_attention_mask = text_decoder_attention_mask.unsqueeze(-1)  #[batch, seq_len, 1]
        speech_decoder_attention_mask = speech_decoder_attention_mask.unsqueeze(-1)

        if self.pool_type == "avg":
            text_pooled = torch.sum(text_hidden_states * text_decoder_attention_mask, dim=1) / \
                          torch.sum(text_decoder_attention_mask, dim=1)

            speech_pooled = torch.sum(speech_hidden_states * speech_decoder_attention_mask, dim=1) / \
                            torch.sum(speech_decoder_attention_mask, dim=1)

        elif self.pool_type == "avgmax":
            text_avg = torch.sum(text_hidden_states * text_decoder_attention_mask, dim=1) / \
                          torch.sum(text_decoder_attention_mask, dim=1)

            speech_avg = torch.sum(speech_hidden_states * speech_decoder_attention_mask, dim=1) / \
                            torch.sum(speech_decoder_attention_mask, dim=1)

            text_max = torch.max(text_hidden_states - 99999*(1-text_decoder_attention_mask), dim=1).values

            speech_max = torch.max(speech_hidden_states - 99999 * (1 - speech_decoder_attention_mask), dim=1).values

            text_pooled = text_avg + text_max
            speech_pooled = speech_avg + speech_max

        elif self.pool_type == "attention":
            text_pooled = text_hidden_states[:, 0, :]

            speech_hidden_states = self.speech_att_linear(speech_hidden_states)  #[batch, frames, hidden]
            speech_hidden_states = torch.tanh(speech_hidden_states)

            speech_pooled = self.speech_att_attention(speech_hidden_states)  #[batch, frames, 1]
            speech_pooled = speech_pooled * speech_decoder_attention_mask - \
                            99999.0 * (1.0 - speech_decoder_attention_mask)
            speech_pooled = F.softmax(speech_pooled, dim=1)
            speech_pooled = torch.matmul(speech_pooled.permute(0, 2, 1), speech_hidden_states).squeeze(1)

        else:
            raise ValueError("Not support pool type: {}".format(self.pool_type))

        fuse_states = torch.cat((text_pooled, speech_pooled), dim=-1)
        hiddens = self.fuse_linear(fuse_states)
        logits = self.output_layer(hiddens)

        return logits, hiddens





