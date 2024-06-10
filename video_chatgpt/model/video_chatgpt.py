from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_TOKEN_1 = "<video1>"
DEFAULT_VIDEO_TOKEN_2 = "<video2>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VIDEO_PATCH_TOKEN_1 = "<vid_patch1>"
DEFAULT_VIDEO_PATCH_TOKEN_2 = "<vid_patch2>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"

DEFAULT_IMG_TOKEN = "<img>"
DEFAULT_IMG_TOKEN1 = "<img1>"
DEFAULT_IMG_TOKEN2 = "<img2>"
DEFAULT_IMG_PATCH_TOKEN = "<img_patch>"
DEFAULT_IMG_PATCH_TOKEN_1 = "<img_patch1>"
DEFAULT_IMG_PATCH_TOKEN_2 = "<img_patch2>"
DEFAULT_IMG_START_TOKEN = "<img_start>"
DEFAULT_IMG_END_TOKEN = "<img_end>"


class VisionConfig:
    def __init__(self):
        self.frame_size = 224
        self.patch_size = 14
        self.hidden_size = 1024
        self.use_vid_start_end = None
        self.vid_start_token = None
        self.vid_end_token = None
        self.vid_patch_token = None


class VideoChatGPTConfig(LlamaConfig):
    model_type = "VideoChatGPT"


class VideoChatGPTLlamaModel(LlamaModel):
    config_class = VideoChatGPTConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):  # TODO: Remove unused params
        super(VideoChatGPTLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_config = VisionConfig()

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def initialize_vision_modules(self, pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        vision_config = self.vision_config
        num_patches = (vision_config.frame_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            video_token_len=num_patches,
            vision_config=vision_config
        )
    
    def forward_single_view(
            self,
            input_ids: torch.LongTensor = None,
            media_type: str = "img",
            inputs_embeds: Optional[torch.FloatTensor] = None,
            orig_embeds_params:  torch.LongTensor = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            img_spatio_temporal_features: Optional[torch.FloatTensor] = None,
         ):
        
        use_only_video = video_spatio_temporal_features is not None and img_spatio_temporal_features is None
        use_only_img = img_spatio_temporal_features is not None and video_spatio_temporal_features is None
        use_one_media_type = not (use_only_video and use_only_img)

        assert use_one_media_type, "Use only either images or videos"
        
        if video_spatio_temporal_features is not None:
            spatio_temporal_features = video_spatio_temporal_features
        elif img_spatio_temporal_features is not None:
            spatio_temporal_features = img_spatio_temporal_features
        else:
            raise ValueError("No input features found")

        if (input_ids.shape[1] != 1 or self.training):

            features = self.mm_projector(spatio_temporal_features)
            dummy_features = torch.zeros(features.shape[1], 1024, device=inputs_embeds.device,
                                            dtype=inputs_embeds.dtype)
            dummy_features = self.mm_projector(dummy_features)

            new_input_embeds = []
            cur_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.vid_patch_token).sum() == 0:
                    # Multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_idx += 1
                    continue
                if self.vision_config.use_vid_start_end:
                    if (cur_input_ids == self.vision_config.vid_start_token).sum() != (
                            cur_input_ids == self.vision_config.vid_end_token).sum():
                        raise ValueError("The number of video start tokens and video end tokens should be the same.")
                    start_tokens = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0]
                    for video_start_token_pos in start_tokens:
                        cur_video_features = features[cur_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_video_features.shape[0]
                        if cur_input_ids[video_start_token_pos + num_patches + 1] != self.vision_config.vid_end_token:
                            raise ValueError("The video end token should follow the video start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos].detach(),
                                                            cur_input_embeds[
                                                            video_start_token_pos:video_start_token_pos + 1],
                                                            cur_video_features, cur_input_embeds[
                                                                                video_start_token_pos + num_patches
                                                                                + 1:video_start_token_pos
                                                                                + num_patches + 2],
                                                            cur_input_embeds[
                                                            video_start_token_pos + num_patches + 2:].detach()),
                                                            dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos + 1],
                                                            cur_video_features,
                                                            cur_input_embeds[video_start_token_pos
                                                                            + num_patches + 1:]), dim=0)
                        cur_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_video_features = features[cur_idx]
                    num_patches = cur_video_features.shape[0]
                    if (cur_input_ids == self.vision_config.vid_patch_token).sum() != num_patches:
                        raise ValueError(
                            "The number of video patch tokens should be the same as the number of video patches.")
                    masked_indices = torch.where(cur_input_ids == self.vision_config.vid_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches,
                                                    device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The video patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(),
                                                        cur_video_features,
                                                        cur_input_embeds[mask_index_start + num_patches:].detach()),
                                                        dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_video_features,
                                                        cur_input_embeds[mask_index_start + num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return inputs_embeds
    
    def forward_two_views(
            self,
            input_ids: torch.LongTensor = None,
            media_type: str = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            orig_embeds_params: Optional[torch.FloatTensor] = None,
            video_spatio_temporal_features1: Optional[torch.FloatTensor] = None,
            video_spatio_temporal_features2: Optional[torch.FloatTensor] = None,
            img_spatio_temporal_features1: Optional[torch.FloatTensor] = None,
            img_spatio_temporal_features2: Optional[torch.FloatTensor] = None,
    ):
        
        use_only_video = (video_spatio_temporal_features1, video_spatio_temporal_features2) != (None, None) \
                and (img_spatio_temporal_features1, img_spatio_temporal_features2) == (None, None)
        use_only_img = (img_spatio_temporal_features1, img_spatio_temporal_features2) != (None, None) \
                and (video_spatio_temporal_features1, video_spatio_temporal_features2) == (None, None)
        
        use_one_media_type = not (use_only_video and use_only_img)

        assert use_one_media_type, "Use only either images or videos"

        
        if (video_spatio_temporal_features1, video_spatio_temporal_features2) != (None, None):
            spatial_temporal_features1, spatial_temporal_features2 = video_spatio_temporal_features1, video_spatio_temporal_features2
        elif (img_spatio_temporal_features1, img_spatio_temporal_features2) != (None, None):
            spatial_temporal_features1, spatial_temporal_features2 = img_spatio_temporal_features1, img_spatio_temporal_features2
        else:
            raise ValueError("No input features found")

        # if (input_ids.shape[1] != 1 or self.training) and video_spatio_temporal_features is not None:
        if (input_ids.shape[1] != 1 or self.training):

            features1 = self.mm_projector(spatial_temporal_features1)
            features2 = self.mm_projector(spatial_temporal_features2)
            dummy_features1 = torch.zeros(features1.shape[1], 1024, device=inputs_embeds.device,
                                            dtype=inputs_embeds.dtype)
            dummy_features2 = torch.zeros(features2.shape[1], 1024, device=inputs_embeds.device,
                                            dtype=inputs_embeds.dtype)
            dummy_features1 = self.mm_projector(dummy_features1)
            dummy_features2 = self.mm_projector(dummy_features2)

            new_input_embeds = []
            cur_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == self.vision_config.vid_patch_token_1).sum() == 0 and \
                    (cur_input_ids == self.vision_config.vid_patch_token_2).sum() == 0:
                    # Multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_idx += 1
                    continue
                if self.vision_config.use_vid_start_end:
                    if (cur_input_ids == self.vision_config.vid_start_token).sum() != (
                            cur_input_ids == self.vision_config.vid_end_token).sum():
                        raise ValueError("The number of video start tokens and video end tokens should be the same.")
                    start_tokens = torch.where(cur_input_ids == self.vision_config.vid_start_token)[0]
                    for video_start_token_pos in start_tokens:
                        cur_video_features = features[cur_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_video_features.shape[0]
                        if cur_input_ids[video_start_token_pos + num_patches + 1] != self.vision_config.vid_end_token:
                            raise ValueError("The video end token should follow the video start token.")
                        if orig_embeds_params is not None:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos].detach(),
                                                            cur_input_embeds[
                                                            video_start_token_pos:video_start_token_pos + 1],
                                                            cur_video_features1,
                                                            cur_input_embeds[
                                                            video_start_token_pos + 2:video_start_token_pos + 3],
                                                            cur_video_features2, cur_input_embeds[
                                                                                video_start_token_pos + num_patches
                                                                                + 1:video_start_token_pos
                                                                                + num_patches + 2],
                                                            cur_input_embeds[
                                                            video_start_token_pos + num_patches + 2:].detach()),
                                                            dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:video_start_token_pos + 1],
                                                            cur_video_features,
                                                            cur_input_embeds[video_start_token_pos
                                                                            + num_patches + 1:]), dim=0)
                        cur_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_video_features1 = features1[cur_idx]

                    cur_video_features2 = features2[cur_idx]
                    num_patches = cur_video_features2.shape[0]  if cur_video_features1.shape[0] == cur_video_features2.shape[0] \
                                    else None

                    if (cur_input_ids == self.vision_config.vid_patch_token_1).sum() != num_patches and \
                        (cur_input_ids == self.vision_config.vid_patch_token_2).sum() != num_patches:
                        raise ValueError(
                            "The number of video patch tokens should be the same as the number of video patches.")
                    masked_indices = torch.where(cur_input_ids == self.vision_config.vid_patch_token_1)[0]
                    # masked_indices_2 = torch.where(cur_input_ids == self.vision_config.vid_patch_token_2)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start + num_patches,
                                                    device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The video patch tokens should be consecutive.")
                    
                    mask_index_end = num_patches * 2
                    if orig_embeds_params is not None:

                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(),
                                                        cur_video_features1,
                                                        cur_video_features2,
                                                        cur_input_embeds[mask_index_start + mask_index_end:].detach()),
                                                        dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(),
                                                        cur_video_features1,
                                                        cur_video_features2,
                                                        cur_input_embeds[mask_index_start + mask_index_end:].detach()),
                                                        dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                    cur_idx += 1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return inputs_embeds

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            media_type: str = "img",
            multi_view: Optional[bool] = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            video_spatio_temporal_features1: Optional[torch.FloatTensor] = None,
            video_spatio_temporal_features2: Optional[torch.FloatTensor] = None,
            img_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            img_spatio_temporal_features1: Optional[torch.FloatTensor] = None,
            img_spatio_temporal_features2: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        if multi_view:
            inputs_embeds = self.forward_two_views(
                    input_ids=input_ids,
                    media_type=media_type,
                    inputs_embeds=inputs_embeds,
                    orig_embeds_params=orig_embeds_params,
                    video_spatio_temporal_features1=video_spatio_temporal_features1,
                    video_spatio_temporal_features2=video_spatio_temporal_features2,
                    img_spatio_temporal_features1=img_spatio_temporal_features1,
                    img_spatio_temporal_features2=img_spatio_temporal_features2,
                )
        else:
            inputs_embeds = self.forward_single_view(
                input_ids=input_ids,
                media_type=media_type,
                inputs_embeds=inputs_embeds,
                orig_embeds_params=orig_embeds_params,
                video_spatio_temporal_features=video_spatio_temporal_features,
                img_spatio_temporal_features=img_spatio_temporal_features,
            )

        return super(VideoChatGPTLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class VideoChatGPTLlamaForCausalLM(LlamaForCausalLM):
    config_class = VideoChatGPTConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = VideoChatGPTLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            video_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            video_spatio_temporal_features1: Optional[torch.FloatTensor] = None,
            video_spatio_temporal_features2: Optional[torch.FloatTensor] = None,
            img_spatio_temporal_features: Optional[torch.FloatTensor] = None,
            img_spatio_temporal_features1: Optional[torch.FloatTensor] = None,
            img_spatio_temporal_features2: Optional[torch.FloatTensor] = None,
            return_dict: Optional[bool] = None,
            multi_view: Optional[bool] = False,
            media_type: str = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            media_type=media_type,
            multi_view=multi_view,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            video_spatio_temporal_features=video_spatio_temporal_features,
            video_spatio_temporal_features1=video_spatio_temporal_features1,
            video_spatio_temporal_features2=video_spatio_temporal_features2,
            img_spatio_temporal_features=img_spatio_temporal_features,
            img_spatio_temporal_features1=img_spatio_temporal_features1,
            img_spatio_temporal_features2=img_spatio_temporal_features2,
            return_dict=return_dict,

        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "video_spatio_temporal_features": kwargs.get("video_spatio_temporal_features", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_vid_start_end, tokenizer, device, media_type, multi_view,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.get_model().vision_config
        vision_config.use_vid_start_end = mm_use_vid_start_end

        if not multi_view:
            if media_type == "video":
                default_patch_token = DEFAULT_VIDEO_PATCH_TOKEN
                default_start_token = DEFAULT_VID_START_TOKEN
                default_end_token = DEFAULT_VID_END_TOKEN
            else:
                default_patch_token = DEFAULT_IMG_PATCH_TOKEN
                default_start_token = DEFAULT_IMG_START_TOKEN
                default_end_token = DEFAULT_IMG_END_TOKEN
            tokenizer.add_tokens([default_patch_token], special_tokens=True)

        else:
            if media_type == "video":
                default_patch_token_1 = DEFAULT_VIDEO_PATCH_TOKEN_1
                default_patch_token_2 = DEFAULT_VIDEO_PATCH_TOKEN_2
                default_start_token = DEFAULT_VID_START_TOKEN
                default_end_token = DEFAULT_VID_END_TOKEN
            else:
                default_patch_token_1 = DEFAULT_IMG_PATCH_TOKEN_1
                default_patch_token_2 = DEFAULT_IMG_PATCH_TOKEN_2
                default_start_token = DEFAULT_IMG_START_TOKEN
                default_end_token = DEFAULT_IMG_END_TOKEN

        
            tokenizer.add_tokens([default_patch_token_1], special_tokens=True)
            tokenizer.add_tokens([default_patch_token_2], special_tokens=True)

        self.resize_token_embeddings(len(tokenizer))

        if mm_use_vid_start_end:
            num_new_tokens = tokenizer.add_tokens([default_start_token, default_end_token], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.vid_start_token, vision_config.vid_end_token = tokenizer.convert_tokens_to_ids(
                [default_start_token, default_end_token])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [
                    self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(
                        f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. "
                        f"Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        if not multi_view:
            vision_config.vid_patch_token = tokenizer.convert_tokens_to_ids([default_patch_token])[0]
        else:
            vision_config.vid_patch_token_1 = tokenizer.convert_tokens_to_ids([default_patch_token_1])[0]
            vision_config.vid_patch_token_2 = tokenizer.convert_tokens_to_ids([default_patch_token_2])[0]


AutoConfig.register("VideoChatGPT", VideoChatGPTConfig)
AutoModelForCausalLM.register(VideoChatGPTConfig, VideoChatGPTLlamaForCausalLM)
