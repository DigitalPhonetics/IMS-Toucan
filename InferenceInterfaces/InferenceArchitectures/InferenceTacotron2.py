import torch
import torch.nn.functional as F

from Layers.Attention import GuidedAttentionLoss
from Layers.RNNAttention import AttForward
from Layers.RNNAttention import AttForwardTA
from Layers.RNNAttention import AttLoc
from Layers.TacotronDecoder import Decoder
from Layers.TacotronEncoder import Encoder
from Utility.SoftDTW.sdtw_cuda_loss import SoftDTW
from Utility.utils import make_non_pad_mask


class Tacotron2(torch.nn.Module):

    def __init__(
            self,
            # network structure related
            path_to_weights,
            idim=66,
            odim=80,
            embed_dim=512,
            elayers=1,
            eunits=512,
            econv_layers=3,
            econv_chans=512,
            econv_filts=5,
            atype="forward_ta",
            adim=512,
            aconv_chans=32,
            aconv_filts=15,
            cumulate_att_w=True,
            dlayers=2,
            dunits=1024,
            prenet_layers=2,
            prenet_units=256,  # 64 or 256 are defaults
            postnet_layers=5,
            postnet_chans=512,
            postnet_filts=5,
            output_activation=None,
            use_batch_norm=True,
            use_concate=True,
            use_residual=False,
            reduction_factor=1,
            spk_embed_dim=None,
            spk_embed_integration_type="add",
            # training related
            dropout_rate=0.5,
            zoneout_rate=0.1,
            use_masking=False,
            use_weighted_masking=True,
            bce_pos_weight=5.0,
            loss_type="L1+L2",
            use_guided_attn_loss=True,
            guided_attn_loss_sigma=0.4,
            guided_attn_loss_lambda=1.0,
            use_dtw_loss=False):
        super().__init__()

        # store hyperparameters
        self.use_dtw_loss = use_dtw_loss
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.spk_embed_dim = spk_embed_dim
        self.cumulate_att_w = cumulate_att_w
        self.reduction_factor = reduction_factor
        self.use_guided_attn_loss = use_guided_attn_loss
        self.loss_type = loss_type
        if self.spk_embed_dim is not None:
            self.spk_embed_integration_type = spk_embed_integration_type

        # define activation function for the final output
        if output_activation is None:
            self.output_activation_fn = None
        elif hasattr(F, output_activation):
            self.output_activation_fn = getattr(F, output_activation)
        else:
            raise ValueError(f"there is no such an activation function. " f"({output_activation})")

        # set padding idx
        padding_idx = 0
        self.padding_idx = padding_idx

        # define network modules
        self.enc = Encoder(idim=idim,
                           input_layer="linear",
                           embed_dim=embed_dim,
                           elayers=elayers,
                           eunits=eunits,
                           econv_layers=econv_layers,
                           econv_chans=econv_chans,
                           econv_filts=econv_filts,
                           use_batch_norm=use_batch_norm,
                           use_residual=use_residual,
                           dropout_rate=dropout_rate,
                           padding_idx=padding_idx, )

        if spk_embed_dim is None:
            dec_idim = eunits
        elif spk_embed_integration_type == "concat":
            dec_idim = eunits + spk_embed_dim
        elif spk_embed_integration_type == "add":
            dec_idim = eunits
            self.projection = torch.nn.Linear(self.spk_embed_dim, eunits)
        else:
            raise ValueError(f"{spk_embed_integration_type} is not supported.")

        if atype == "location":
            att = AttLoc(dec_idim, dunits, adim, aconv_chans, aconv_filts)
        elif atype == "forward":
            att = AttForward(dec_idim, dunits, adim, aconv_chans, aconv_filts)
            if self.cumulate_att_w:
                self.cumulate_att_w = False
        elif atype == "forward_ta":
            att = AttForwardTA(dec_idim, dunits, adim, aconv_chans, aconv_filts, odim)
            if self.cumulate_att_w:
                self.cumulate_att_w = False
        else:
            raise NotImplementedError("Support only location or forward")
        self.dec = Decoder(idim=dec_idim,
                           odim=odim,
                           att=att,
                           dlayers=dlayers,
                           dunits=dunits,
                           prenet_layers=prenet_layers,
                           prenet_units=prenet_units,
                           postnet_layers=postnet_layers,
                           postnet_chans=postnet_chans,
                           postnet_filts=postnet_filts,
                           output_activation_fn=self.output_activation_fn,
                           cumulate_att_w=self.cumulate_att_w,
                           use_batch_norm=use_batch_norm,
                           use_concate=use_concate,
                           dropout_rate=dropout_rate,
                           zoneout_rate=zoneout_rate,
                           reduction_factor=reduction_factor, )
        self.taco2_loss = Tacotron2Loss(use_masking=use_masking,
                                        use_weighted_masking=use_weighted_masking,
                                        bce_pos_weight=bce_pos_weight, )
        if self.use_guided_attn_loss:
            self.attn_loss = GuidedAttentionLoss(sigma=guided_attn_loss_sigma,
                                                 alpha=guided_attn_loss_lambda, )
        if self.use_dtw_loss:
            self.dtw_criterion = SoftDTW(use_cuda=True, gamma=0.1)

        self.load_state_dict(torch.load(path_to_weights, map_location='cpu')["model"])

    def forward(self, text, speaker_embedding=None, return_atts=False,
                threshold=0.5,
                minlenratio=0.0,
                maxlenratio=10.0,
                use_att_constraint=False,
                backward_window=1,
                forward_window=3):
        spemb = speaker_embedding
        h = self.enc.inference(text)
        if self.spk_embed_dim is not None:
            hs, spembs = h.unsqueeze(0), spemb.unsqueeze(0)
            h = self._integrate_with_spk_embed(hs, spembs)[0]
        outs, probs, att_ws = self.dec.inference(h,
                                                 threshold=threshold,
                                                 minlenratio=minlenratio,
                                                 maxlenratio=maxlenratio,
                                                 use_att_constraint=use_att_constraint,
                                                 backward_window=backward_window,
                                                 forward_window=forward_window, )
        if return_atts:
            return att_ws
        else:
            return outs

    def _forward(self,
                 xs: torch.Tensor,
                 ilens: torch.Tensor,
                 ys: torch.Tensor,
                 olens: torch.Tensor,
                 spembs: torch.Tensor, ):
        hs, hlens = self.enc(xs, ilens)
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)
        return self.dec(hs, hlens, ys)

    def _integrate_with_spk_embed(self, hs: torch.Tensor,
                                  spembs: torch.Tensor):
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = torch.cat([hs, spembs], dim=-1)
        else:
            raise NotImplementedError("support only add or concat.")

        return hs


class Tacotron2Loss(torch.nn.Module):

    def __init__(self, use_masking=False, use_weighted_masking=True, bce_pos_weight=20.0):
        super(Tacotron2Loss, self).__init__()
        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.bce_criterion = torch.nn.BCEWithLogitsLoss(
            reduction=reduction, pos_weight=torch.tensor(bce_pos_weight)
            )

        self._register_load_state_dict_pre_hook(self._load_state_dict_pre_hook)

    def forward(self, after_outs, before_outs, logits, ys, labels, olens):
        # make mask and apply it
        if self.use_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            ys = ys.masked_select(masks)
            after_outs = after_outs.masked_select(masks)
            before_outs = before_outs.masked_select(masks)
            labels = labels.masked_select(masks[:, :, 0])
            logits = logits.masked_select(masks[:, :, 0])

        # calculate loss
        l1_loss = self.l1_criterion(after_outs, ys) + self.l1_criterion(before_outs, ys)
        mse_loss = self.mse_criterion(after_outs, ys) + self.mse_criterion(
            before_outs, ys
            )
        bce_loss = self.bce_criterion(logits, labels)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            masks = make_non_pad_mask(olens).unsqueeze(-1).to(ys.device)
            weights = masks.float() / masks.sum(dim=1, keepdim=True).float()
            out_weights = weights.div(ys.size(0) * ys.size(2))
            logit_weights = weights.div(ys.size(0))

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(masks).sum()
            mse_loss = mse_loss.mul(out_weights).masked_select(masks).sum()
            bce_loss = (
                bce_loss.mul(logit_weights.squeeze(-1)).masked_select(masks.squeeze(-1)).sum())

        return l1_loss, mse_loss, bce_loss

    def _load_state_dict_pre_hook(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
            ):
        """Apply pre hook fucntion before loading state dict.
        From v.0.6.1 `bce_criterion.pos_weight` param is registered as a parameter but
        old models do not include it and as a result, it causes missing key error when
        loading old model parameter. This function solve the issue by adding param in
        state dict before loading as a pre hook function
        of the `load_state_dict` method.
        """
        key = prefix + "bce_criterion.pos_weight"
        if key not in state_dict:
            state_dict[key] = self.bce_criterion.pos_weight
