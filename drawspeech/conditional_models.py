
import torch
import json
import torch.nn as nn
import torch.nn.functional as F

from drawspeech.modules.text_encoder.encoder import TextEncoder
from drawspeech.modules.fastspeech2.modules import VarianceAdaptor
from drawspeech.modules.contour_predictor.model import Sketch2ContourPredictor

from drawspeech.utilities.tools import wav_mask_to_latent_mask, modify_curve_length, min_max_normalize, wav_mask_to_latent_mask, sketch_extractor, make_decision
from drawspeech.utilities.diffusion_util import conv_nd


"""
The model forward function can return three types of data:
1. tensor: used directly as conditioning signal
2. dict: where there is a main key as condition, there are also other key that you can use to pass loss function and itermediate result. etc.
3. list: the length is 2, in which the first element is tensor, the second element is attntion mask.

The output shape for the cross attention condition should be:
x,x_mask = [bs, seq_len, emb_dim], [bs, seq_len]

All the returned data, in which will be used as diffusion input, will need to be in float type
"""

class PhonemeEncoder(nn.Module):
    def __init__(self, vocabs_size=41, pad_length=250, pad_token_id=None, output_size=192):
        super().__init__()
        """
            encoder = PhonemeEncoder(40)
            data = torch.randint(0, 39, (2, 250))
            output = encoder(data)
            import ipdb;ipdb.set_trace()
        """
        assert pad_token_id is not None

        self.device = None
        self.PAD_LENGTH = int(pad_length)
        self.pad_token_id = pad_token_id
        self.pad_token_sequence = torch.tensor([self.pad_token_id] * self.PAD_LENGTH)

        self.text_encoder = TextEncoder(
            n_vocab=vocabs_size,
            out_channels=output_size,
            hidden_channels=output_size,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
        )

        self.learnable_positional_embedding = torch.nn.Parameter(
            torch.zeros((1, output_size, self.PAD_LENGTH))
        )  # [batchsize, seqlen, padlen]
        self.learnable_positional_embedding.requires_grad = True

    # Required
    def get_unconditional_condition(self, batchsize):
        unconditional_tokens = self.pad_token_sequence.expand(
            batchsize, self.PAD_LENGTH
        )
        return self(unconditional_tokens)  # Need to return float type

    # def get_unconditional_condition(self, batchsize):

    #     hidden_state = torch.zeros((batchsize, self.PAD_LENGTH, 192)).to(self.device)
    #     attention_mask = torch.ones((batchsize, self.PAD_LENGTH)).to(self.device)
    #     return [hidden_state, attention_mask] # Need to return float type

    def _get_src_mask(self, phoneme):
        src_mask = phoneme != self.pad_token_id
        return src_mask

    def _get_src_length(self, phoneme):
        src_mask = self._get_src_mask(phoneme)
        length = torch.sum(src_mask, dim=-1)
        return length

    # def make_empty_condition_unconditional(self, src_length, text_emb, attention_mask):
    #     # src_length: [bs]
    #     # text_emb: [bs, 192, pad_length]
    #     # attention_mask: [bs, pad_length]
    #     mask = src_length[..., None, None] > 1
    #     text_emb = text_emb * mask

    #     attention_mask[src_length < 1] = attention_mask[src_length < 1] * 0.0 + 1.0
    #     return text_emb, attention_mask

    def forward(self, phoneme_idx):
        if self.device is None:
            self.device = self.learnable_positional_embedding.device
            self.pad_token_sequence = self.pad_token_sequence.to(self.device)

        phoneme_idx = phoneme_idx.to(self.device)
        
        src_length = self._get_src_length(phoneme_idx)
        text_emb, m, logs, text_emb_mask = self.text_encoder(phoneme_idx, src_length)
        text_emb = text_emb + self.learnable_positional_embedding

        # text_emb, text_emb_mask = self.make_empty_condition_unconditional(src_length, text_emb, text_emb_mask)

        return [
            text_emb.permute(0, 2, 1),
            text_emb_mask.squeeze(1),
        ]  # [2, 250, 192], [2, 250]


class TextEncoderwithVarianceAdaptor(nn.Module):
    def __init__(
        self,
        vocabs_size=41,
        pad_length=250,
        pad_token_id=None,
        output_size=192,
        latent_t_size=216,
        latent_f_size=20,
        predict_detailed_curve=False,
        prob_drop_pitch=0.0,
        prob_drop_energy=0.0,
        adaptor_params={},
        predictor_params={}
    ):
        super().__init__()
        
        self.pitch_feature_level = adaptor_params["pitch_feature_level"]
        self.energy_feature_level = adaptor_params["energy_feature_level"]
        self.predict_detailed_curve = predict_detailed_curve
        self.max_text_length = pad_length
        self.prob_drop_pitch = prob_drop_pitch
        self.prob_drop_energy = prob_drop_energy
        print(f"prob_drop_pitch: {self.prob_drop_pitch} | prob_drop_energy: {self.prob_drop_energy}")

        assert self.pitch_feature_level == self.energy_feature_level
        self.feature_level = "frame_level" if self.pitch_feature_level == "frame_level" else "phoneme_level"

        self.text_encoder = PhonemeEncoder(
            vocabs_size=vocabs_size,
            pad_length=pad_length,
            pad_token_id=pad_token_id,
            output_size=output_size,
        )
        self.variance_adaptor = VarianceAdaptor(**adaptor_params)
        self.sketch_to_contour_predictor = Sketch2ContourPredictor(**predictor_params) if self.predict_detailed_curve else None
          
        self.conv_layer = conv_nd(1, output_size, latent_f_size, 3, padding=1)

        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size
        # self.PAD_LENGTH = pad_length
        # self.pad_token_id = pad_token_id
        # self.pad_token_sequence = torch.tensor([pad_token_id] * pad_length)
        self.null_cond = nn.Parameter(torch.zeros(latent_t_size, latent_f_size))

        self.infer = False

    def get_loss(self, targets, predictions, src_mask, mel_mask):

        pitch_targets, energy_targets, duration_targets = targets
        pitch_predictions, energy_predictions, log_duration_predictions = predictions

        if pitch_predictions is None and energy_predictions is None and log_duration_predictions is None:
            return 0, 0, 0
        
        src_mask = ~src_mask
        mel_mask = ~mel_mask
        mel_mask = mel_mask[:, :mel_mask.shape[1]]

        if pitch_targets is not None:
            pitch_targets.requires_grad = False
            if self.pitch_feature_level == "phoneme_level":
                pitch_predictions = pitch_predictions.masked_select(src_mask)
                pitch_targets = pitch_targets.masked_select(src_mask)
            elif self.pitch_feature_level == "frame_level":
                pitch_predictions = pitch_predictions.masked_select(mel_mask)
                pitch_targets = pitch_targets.masked_select(mel_mask)
            pitch_loss = torch.nn.functional.mse_loss(pitch_predictions, pitch_targets)
        else:
            pitch_loss = 0
        
        if energy_targets is not None:
            energy_targets.requires_grad = False
            if self.energy_feature_level == "phoneme_level":
                energy_predictions = energy_predictions.masked_select(src_mask)
                energy_targets = energy_targets.masked_select(src_mask)
            if self.energy_feature_level == "frame_level":
                energy_predictions = energy_predictions.masked_select(mel_mask)
                energy_targets = energy_targets.masked_select(mel_mask)
            energy_loss = torch.nn.functional.mse_loss(energy_predictions, energy_targets)
        else:
            energy_loss = 0
        
        if duration_targets is not None:
            log_duration_targets = torch.log(duration_targets.float() + 1)
            log_duration_targets.requires_grad = False
            log_duration_predictions = log_duration_predictions.masked_select(src_mask)
            log_duration_targets = log_duration_targets.masked_select(src_mask)
            duration_loss = torch.nn.functional.mse_loss(log_duration_predictions, log_duration_targets)
        else:
            duration_loss = 0

        return pitch_loss, energy_loss, duration_loss

    # Required
    def get_unconditional_condition(self, batchsize):
        # inputs = {
        #     "phoneme_idx": self.pad_token_sequence.expand(batchsize, self.PAD_LENGTH),
        #     "pitch": None,
        #     "energy": None,
        #     "phoneme_duration": None,
        #     "latent_mask": None,
        # }
        # return self(inputs)

        param = next(self.text_encoder.parameters())
        device = param.device
    
        null_cond = self.null_cond.expand(batchsize, self.null_cond.size(0), self.null_cond.size(1)).to(device)
        # null_cond_mask = torch.zeros((batchsize, self.PAD_LENGTH)).to(device).float()

        return null_cond

    def predict_duration(self, phoneme_idx):
        text_emb, src_mask = self.text_encoder(phoneme_idx)   # # [b, t, c], [b, t]
        _src_mask = ~src_mask.bool()
        duration_rounded = self.variance_adaptor.predict_duration(text_emb, _src_mask)
        return duration_rounded

    def forward(self, inputs):
        '''
        mel_mask, src_mask: 1 for valid token, 0 for padding token (In LDM, the mask is 1 for valid token, 0 for padding token)
        _mel_mask, _src_mask: 0 for valid token, 1 for padding token
        '''
        phoneme_idx = inputs["phoneme_idx"]
        pitch = inputs["pitch"]
        energy = inputs["energy"]
        phoneme_duration = inputs["phoneme_duration"]
        mel_mask = inputs["mel_mask"]
        pitch_length = inputs["pitch_length"]
        energy_length = inputs["energy_length"]
        pitch_sketch = inputs["pitch_sketch"]
        energy_sketch = inputs["energy_sketch"]

        text_emb, src_mask = self.text_encoder(phoneme_idx)   # # [b, t, c], [b, t]
        _src_mask = ~src_mask.bool()

        max_mel_len = mel_mask.size(1)

        if self.training:
            _mel_mask = ~mel_mask.bool()
        elif self.infer:
            _mel_mask = None
            pitch = pitch if isinstance(pitch, torch.Tensor) else None
            pitch_sketch = pitch_sketch if isinstance(pitch_sketch, torch.Tensor) else None
            pitch_length = pitch_length if isinstance(pitch_length, torch.Tensor) else None
            energy = energy if isinstance(energy, torch.Tensor) else None
            energy_sketch = energy_sketch if isinstance(energy_sketch, torch.Tensor) else None
            energy_length = energy_length if isinstance(energy_length, torch.Tensor) else None
            phoneme_duration = phoneme_duration if isinstance(phoneme_duration, torch.Tensor) else None
            print(f"Given pitch_sketch: {pitch_sketch is not None} | " + f"Given energy_sketch: {energy_sketch is not None}")
            print(f"Given pitch: {pitch is not None} | " + f"Given energy: {energy is not None} | " + f"Given phoneme_duration: {phoneme_duration is not None}")
        else:
            # under validation
            _mel_mask = None
            pitch = None
            pitch_sketch = None
            pitch_length = None
            energy = None
            energy_sketch = None
            energy_length = None
            phoneme_duration = None

        # when the feature level is phoneme_level, the returned (expanded_text_emb, expanded_text_mask) are equal to (text_emb, _src_mask)
        expanded_text_emb, expanded_text_mask = self.variance_adaptor.get_expanded_text_embedding(text_emb, _src_mask, max_mel_len, phoneme_duration) 

        if self.feature_level == "frame_level":
            _mel_mask = expanded_text_mask
            mel_mask = ~expanded_text_mask
            curve_mask = mel_mask
        else:
            curve_mask = src_mask

        if self.sketch_to_contour_predictor:
            if self.training:
                # we do not provide the sketches during training with a certain probability
                if make_decision(self.prob_drop_pitch):
                    pitch_sketch = None
                if make_decision(self.prob_drop_energy):
                    energy_sketch = None
            else:
                pitch_sketch = modify_curve_length(pitch_sketch, pitch_length, curve_mask, pad_is_1=False)
                energy_sketch = modify_curve_length(energy_sketch, energy_length, curve_mask, pad_is_1=False)
            detailed_pitch, detailed_energy = self.sketch_to_contour_predictor(expanded_text_emb, pitch_sketch, energy_sketch, curve_mask)
        else:
            detailed_pitch, detailed_energy = None, None
        
        if not self.training:
            # During training, the pitch target is from the utterance to be synthesized. So we do not need to modify the curve.
            # During inference, the pitch target can come from the reference utterance and their lengths may be different. So we need to modify the curve.
            pitch = modify_curve_length(pitch, pitch_length, curve_mask, pad_is_1=False)
            energy = modify_curve_length(energy, energy_length, curve_mask, pad_is_1=False)
            pitch = pitch if pitch is not None else detailed_pitch
            energy = energy if energy is not None else detailed_energy

        output, p_predictions, e_predictions, log_d_predictions, d_rounded, mel_lens, _mel_mask = self.variance_adaptor(text_emb, _src_mask, _mel_mask, max_mel_len, pitch, energy, phoneme_duration)
        mel_mask = ~_mel_mask.bool()
        
        if self.training:
            targets = [pitch, energy, phoneme_duration]
            predictions = [p_predictions, e_predictions, log_d_predictions]
            pitch_loss, energy_loss, duration_loss = self.get_loss(targets, predictions, _src_mask, _mel_mask)

            targets = [pitch, energy, None]
            predictions = [detailed_pitch, detailed_energy, None]
            detailed_pitch_loss, detailed_energy_loss, _ = self.get_loss(targets, predictions, _src_mask, _mel_mask)
        else:
            pitch_loss, energy_loss, duration_loss = 0., 0., 0.
            detailed_pitch_loss, detailed_energy_loss = 0., 0.

        output = output * mel_mask.unsqueeze(-1)  # output shape: [b, target_mel_t, phoneme_emb_dim]
        output = F.interpolate(output.transpose(-1, -2), size=self.latent_t_size, mode="linear", align_corners=True)
        output = self.conv_layer(output).transpose(-1, -2)

        ret_dict = {}
        ret_dict["noncond_loss_pitch_loss"] = pitch_loss
        ret_dict["noncond_loss_energy_loss"] = energy_loss
        ret_dict["noncond_loss_duration_loss"] = duration_loss
        ret_dict["noncond_loss_detailed_pitch_loss"] = detailed_pitch_loss
        ret_dict["noncond_loss_detailed_energy_loss"] = detailed_energy_loss
        ret_dict["concat_text_encoder_with_varianceadaptor"] = output
        ret_dict["noncond_predicted_pitch"] = [detailed_pitch, curve_mask] if detailed_pitch is not None else [p_predictions, curve_mask]
        ret_dict["noncond_predicted_energy"] = [detailed_energy, curve_mask] if detailed_energy is not None else [e_predictions, curve_mask]
        ret_dict["noncond_mel_mask"] = mel_mask
        return ret_dict
 

class SketchEncoder(nn.Module):
    def __init__(self, latent_t_size, latent_f_size, stats_json="", pitch_feature_level="frame_level", energy_feature_level="frame_level"):
        super().__init__()
        
        if stats_json:
            with open(stats_json, "r") as f:
                stats = json.load(f)
                self.pitch_min, pitch_max = stats["pitch"][:2]
                self.energy_min, energy_max = stats["energy"][:2]
        else:
            self.pitch_min = self.energy_min = 0
        
        self.latent_t_size = latent_t_size
        self.latent_f_size = latent_f_size
        self.pitch_feature_level = pitch_feature_level
        self.energy_feature_level = energy_feature_level

        self.null_cond = nn.Parameter(torch.zeros(2, latent_t_size, latent_f_size))
        
        self.infer = False

    def get_unconditional_condition(self, batchsize):
        return self.null_cond.expand(batchsize, -1, -1, -1)
    
    def forward(self, inputs):
        pitch = inputs["pitch"]
        pitch_sketch = inputs["pitch_sketch"]
        pitch_length = inputs["pitch_length"]
        energy = inputs["energy"]
        energy_sketch = inputs["energy_sketch"]
        energy_length = inputs["energy_length"]
        pitch_prediction, pitch_mask = inputs["noncond_predicted_pitch"]
        energy_prediction, energy_mask = inputs["noncond_predicted_energy"]
        mel_mask = inputs["noncond_mel_mask"]
        
        if self.training:
            con_pitch = pitch_sketch
            con_energy = energy_sketch
        elif self.infer:
            if isinstance(pitch_sketch, torch.Tensor):
                # get the curve from user input
                con_pitch = pitch_sketch
            elif isinstance(pitch, torch.Tensor):
                con_pitch = pitch
                con_pitch = sketch_extractor(pitch_prediction)
                con_pitch = min_max_normalize(con_pitch)
            else:
                con_pitch = sketch_extractor(pitch_prediction)
                con_pitch = min_max_normalize(con_pitch)
                pitch_length = pitch_mask.sum(dim=1).long()
            
            if isinstance(energy_sketch, torch.Tensor):
                con_energy = energy_sketch
            elif isinstance(energy, torch.Tensor):
                con_energy = energy
                con_energy = sketch_extractor(energy_prediction)
                con_energy = min_max_normalize(con_energy)
            else:
                con_energy = sketch_extractor(energy_prediction)
                con_energy = min_max_normalize(con_energy)
                energy_length = energy_mask.sum(dim=1).long()
        else:
            con_pitch = sketch_extractor(pitch_prediction)
            con_energy = sketch_extractor(energy_prediction)
            con_pitch = min_max_normalize(con_pitch)
            con_energy = min_max_normalize(con_energy)

            pitch_length = pitch_mask.sum(dim=1).long()
            energy_length = energy_mask.sum(dim=1).long()
                
        latent_mask = wav_mask_to_latent_mask(mel_mask, self.latent_t_size)
        con_pitch = modify_curve_length(con_pitch, pitch_length, latent_mask, pad_is_1=False)
        con_energy = modify_curve_length(con_energy, energy_length, latent_mask, pad_is_1=False)
        con_pitch_energy = torch.stack([con_pitch, con_energy], dim=1).unsqueeze(-1).expand(-1, -1, -1, self.latent_f_size)

        return con_pitch_energy



