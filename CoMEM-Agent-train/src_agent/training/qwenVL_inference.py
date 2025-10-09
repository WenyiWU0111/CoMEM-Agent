from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel, Qwen2_5_VLCausalLMOutputWithPast
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from torch.nn import CrossEntropyLoss
from transformers.utils import logging
import numpy as np
from qformer import QFormer
from transformers import AutoTokenizer
logger = logging.get_logger(__name__)

# INFERENCE: Main structure of Inference part
class Qwen2_5_VLForConditionalGeneration_new(Qwen2_5_VLForConditionalGeneration):

    def __init__(self, config, max_memory=None, device_map=None):
        super().__init__(config)
        self.model_inf = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True,
            max_memory=max_memory,
            device_map=device_map
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
        )
        self.knowledge_processor = QFormer()
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here
        self.knowledge_rope_deltas = None
        # Initialize weights and apply final processing
        self.post_init()
        
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image

                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        if second_per_grid_ts is not None:
                            second_per_grid_t = second_per_grid_ts[video_index]
                        else:
                            second_per_grid_t = 1.0
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                    expanded_range = range_tensor.expand(-1, llm_grid_h * llm_grid_w)

                    time_tensor = expanded_range * second_per_grid_t * self.config.vision_config.tokens_per_second

                    time_tensor_long = time_tensor.long()
                    t_index = time_tensor_long.flatten()

                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    # EDIT: forward function is modified to support history and experience inputs
    # @add_start_docstrings_to_model_forward(QWEN2_5_VL_INPUTS_DOCSTRING)
    # @replace_return_docstrings(output_type=Qwen2_5_VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        # Added for History 
        history_input_ids: torch.LongTensor = None,
        history_inputs_embeds: Optional[torch.FloatTensor] = None,
        history_attention_mask: Optional[torch.Tensor] = None,
        history_position_ids: Optional[torch.LongTensor] = None,
        history_pixel_values: Optional[torch.Tensor] = None,
        history_image_grid_thw: Optional[torch.LongTensor] = None,
        history_cache_position: Optional[torch.LongTensor] = None,
        history_past_key_values: Optional[List[torch.FloatTensor]] = None,
        # Added for Experience
        experience_input_ids: torch.LongTensor = None,
        experience_inputs_embeds: Optional[torch.FloatTensor] = None,
        experience_attention_mask: Optional[torch.Tensor] = None,
        experience_position_ids: Optional[torch.LongTensor] = None,
        experience_pixel_values: Optional[torch.Tensor] = None,
        experience_image_grid_thw: Optional[torch.LongTensor] = None,
        experience_cache_position: Optional[torch.LongTensor] = None,
        experience_past_key_values: Optional[List[torch.FloatTensor]] = None,
    # ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
    ):
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        >>> model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""
        # print(self.tokenizer.decode(input_ids[0], skip_special_tokens=False))
        # Batch size is always 1
        attention_mask = None
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.dtype)
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
                video_embeds = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        
        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    video_grid_thw,
                    second_per_grid_ts,
                    attention_mask,
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        # INFERENCE: Prepare experience and history inputs if it's the 1st token generation
        if input_ids.shape[1] != 1:
            concatenated_embeddings, final_position_ids = self.get_compress_history_and_experience(inputs_embeds, position_ids,
                                history_input_ids, history_inputs_embeds, history_attention_mask, history_position_ids, history_pixel_values, history_image_grid_thw, history_cache_position, history_past_key_values,
                                experience_input_ids, experience_inputs_embeds, experience_attention_mask, experience_position_ids, experience_pixel_values, experience_image_grid_thw, experience_cache_position, experience_past_key_values)
            self.shift_length = concatenated_embeddings.shape[1] - inputs_embeds.shape[1]
            print('shift_length', self.shift_length)
        else:
            concatenated_embeddings = inputs_embeds
            final_position_ids = position_ids + self.shift_length  ### remember to add knowledege length to the position ids
        concatenated_embeddings = concatenated_embeddings.to(self.model_inf.device)
        final_position_ids = final_position_ids.to(self.model_inf.device)
        # print('run inference model')
        outputs = self.model_inf(
            input_ids=None,
            position_ids=final_position_ids,
            # attention_mask=attention_mask,
            attention_mask=None,
            past_key_values=past_key_values,
            inputs_embeds=concatenated_embeddings,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            cache_position=None,
        )
        hidden_states = outputs.hidden_states[-1]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits_sequence_length = logits.shape[1]
            labels_sequence_length = labels.shape[1]
            remove_length = logits_sequence_length - labels_sequence_length
            logits = logits.float()
            logits = logits[:, remove_length:, :] #QEDIT: remove the first 24 q-former tokens ##NOTE: change to 80 if 10
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2_5_VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )

    def get_compress_history_and_experience(self, 
                               inputs_embeds, position_ids,
                               history_input_ids, history_inputs_embeds, history_attention_mask, history_position_ids, history_pixel_values, history_image_grid_thw, history_cache_position, history_past_key_values,
                               experience_input_ids, experience_inputs_embeds, experience_attention_mask, experience_position_ids, experience_pixel_values, experience_image_grid_thw, experience_cache_position, experience_past_key_values):        
        
        # Initialize compressed embeddings list
        compressed_embeddings_list = []
        
        # Process History inputs if provided
        if history_input_ids is not None and len(history_input_ids) > 0:
            print("Processing history inputs...")
            compressed_history = self._process_and_compress_inputs(
                history_input_ids, history_inputs_embeds, history_attention_mask, 
                history_position_ids, history_pixel_values, history_image_grid_thw, 
                history_cache_position, history_past_key_values, "history"
            )
            if compressed_history is not None:
                compressed_embeddings_list.append(compressed_history)
        
        # Process Experience inputs if provided
        if experience_input_ids is not None and len(experience_input_ids) > 0:
            print("Processing experience inputs...")
            compressed_experience = self._process_and_compress_inputs(
                experience_input_ids, experience_inputs_embeds, experience_attention_mask, 
                experience_position_ids, experience_pixel_values, experience_image_grid_thw, 
                experience_cache_position, experience_past_key_values, "experience"
            )
            if compressed_experience is not None:
                compressed_embeddings_list.append(compressed_experience)
        
        # If no compressed embeddings, return original inputs
        if not compressed_embeddings_list:
            return inputs_embeds, position_ids
        
        # Concatenate all compressed embeddings
        all_compressed_embeddings = torch.cat(compressed_embeddings_list, dim=1)
        print(f"All compressed embeddings shape: {all_compressed_embeddings.shape}")
        
        concatenated_embeddings, final_position_ids = get_qformer_position_id(all_compressed_embeddings, inputs_embeds, position_ids)
        return concatenated_embeddings, final_position_ids
    
    def _process_and_compress_inputs(self, input_ids, inputs_embeds, attention_mask, 
                                   position_ids, pixel_values, image_grid_thws, 
                                   cache_position, past_key_values, input_type):
        """Helper function to process and compress a specific type of inputs (history/experience)"""
        
        compressed_list = []
        # batch size is always 1
        for k_input_id, k_pixel_value, k_image_grid_thw in zip(input_ids, pixel_values, image_grid_thws):
            k_input_id = k_input_id.unsqueeze(0).to(self.model.device)#
            k_pixel_value = k_pixel_value.unsqueeze(0).to(self.model.device)#
            k_image_grid_thw = k_image_grid_thw.to(self.model.device)#.unsqueeze(0)
            # k_attention_mask = attention_mask[0]
            k_attention_mask = torch.ones(k_input_id.shape[0], k_input_id.shape[1]).to(k_input_id.device)
            # Get embeddings
            k_inputs_embed = self.model.embed_tokens(k_input_id)
            
            # Process pixel values if provided
            if k_pixel_value is not None:
                k_pixel_value = k_pixel_value.type(self.visual.dtype)
                k_image_embeds = self.visual(k_pixel_value, grid_thw=k_image_grid_thw)
                n_image_tokens = (k_input_id == self.config.image_token_id).sum().item()
                n_image_features = k_image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = k_input_id == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(k_inputs_embed)
                image_mask = mask_expanded.to(k_inputs_embed.device)

                k_image_embeds = k_image_embeds.to(k_inputs_embed.device, k_inputs_embed.dtype)
                k_inputs_embed = k_inputs_embed.masked_scatter(image_mask, k_image_embeds)
                
            # Get Position IDs
            k_position_ids = None
            if position_ids is None and (k_attention_mask is None or k_attention_mask.ndim == 2):
                # calculate RoPE index once per generation in the pre-fill stage only
                if (
                    (cache_position is not None and cache_position[0] == 0)
                    or self.knowledge_rope_deltas is None
                    or (past_key_values is None or past_key_values.get_seq_length() == 0)
                ):
                    k_position_ids, knowledge_rope_deltas = self.get_rope_index(
                        k_input_id,
                        k_image_grid_thw,
                        None,
                        None,
                        k_attention_mask,
                    )
                    self.knowledge_rope_deltas = knowledge_rope_deltas
                # then use the prev pre-calculated rope-deltas to get the correct position ids
                else:
                    batch_size, seq_length, _ = k_inputs_embed.shape
                    delta = (
                        (cache_position[0] + self.knowledge_rope_deltas).to(k_inputs_embed.device)
                        if cache_position is not None
                        else 0
                    )
                    k_position_ids = torch.arange(seq_length, device=k_inputs_embed.device)
                    k_position_ids = k_position_ids.view(1, -1).expand(batch_size, -1)
                    if cache_position is not None:  # otherwise `deltas` is an int `0`
                        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    k_position_ids = k_position_ids.add(delta)
                    k_position_ids = k_position_ids.unsqueeze(0).expand(3, -1, -1)   
            
            # Run through model to get hidden states
            self.model.config.use_cache = True
            k_output = self.model(
                input_ids=None,
                position_ids=k_position_ids,
                attention_mask=k_attention_mask,
                inputs_embeds=k_inputs_embed,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                cache_position=None,
            )
            single_hidden_state = k_output.hidden_states[-1] # Shape: [1, seq_len, hidden_dim]
            single_attention_mask = k_attention_mask # Shape: [1, seq_len]
            compressed = self.knowledge_processor(single_hidden_state, single_attention_mask)
            compressed_list.append(compressed.to(k_input_id.device))
            
        # Concatenate along sequence length dimension (dim=1)
        compressed_inputs_embeds = torch.cat(compressed_list, dim=1)
        print(f"compressed_{input_type}_inputs_embeds: {compressed_inputs_embeds.shape}")
        
        
        
        
        return compressed_inputs_embeds

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )
        # Additional inputs for history
        model_inputs["history_input_ids"] = kwargs.get("history_input_ids", None)
        model_inputs["history_inputs_embeds"] = kwargs.get("history_inputs_embeds", None)
        model_inputs["history_attention_mask"] = kwargs.get("history_attention_mask", None)
        model_inputs["history_pixel_values"] = kwargs.get("history_pixel_values", None)
        model_inputs["history_image_grid_thw"] = kwargs.get("history_image_grid_thw", None)
        model_inputs["history_position_ids"] = kwargs.get("history_position_ids", None)
        model_inputs["history_cache_position"] = kwargs.get("history_cache_position", None)
        model_inputs["history_past_key_values"] = kwargs.get("history_past_key_values", None)
        
        # Additional inputs for experience
        model_inputs["experience_input_ids"] = kwargs.get("experience_input_ids", None)
        model_inputs["experience_inputs_embeds"] = kwargs.get("experience_inputs_embeds", None)
        model_inputs["experience_attention_mask"] = kwargs.get("experience_attention_mask", None)
        model_inputs["experience_pixel_values"] = kwargs.get("experience_pixel_values", None)
        model_inputs["experience_image_grid_thw"] = kwargs.get("experience_image_grid_thw", None)
        model_inputs["experience_position_ids"] = kwargs.get("experience_position_ids", None)
        model_inputs["experience_cache_position"] = kwargs.get("experience_cache_position", None)
        model_inputs["experience_past_key_values"] = kwargs.get("experience_past_key_values", None)
        
        # Qwen2-5-VL position_ids are prepareed with rope_deltas in forward
        model_inputs["position_ids"] = None

        if cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None

        return model_inputs

    def _get_image_nums_and_video_nums(
        self,
        input_ids: Optional[torch.LongTensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the number of images and videos for each sample to calculate the separation length of the sample tensor.
        These parameters are not passed through the processor to avoid unpredictable impacts from interface modifications.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary.

        Returns:
            image_nums (`torch.LongTensor` of shape `(batch_size, num_images_sample)`)
            video_nums (`torch.LongTensor` of shape `(batch_size, num_videos_sample)`)
        """
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id

        vision_start_mask = input_ids == vision_start_token_id
        vision_first_mask = torch.roll(vision_start_mask, shifts=1, dims=1)
        image_mask = input_ids == image_token_id
        video_mask = input_ids == video_token_id
        image_nums = torch.sum(vision_first_mask & image_mask, dim=1)
        video_nums = torch.sum(vision_first_mask & video_mask, dim=1)

        return image_nums, video_nums

    def _expand_inputs_for_generation(
        self,
        expand_size: int = 1,
        is_encoder_decoder: bool = False,
        input_ids: Optional[torch.LongTensor] = None,
        **model_kwargs,
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        # Overwritten -- Support for expanding tensors without a batch size dimension
        # e.g., pixel_values, image_grid_thw, pixel_values_videos, video_grid_thw, second_per_grid_t
        # pixel_values.shape[0] is sum(seqlen_images for samples)
        # image_grid_thw.shape[0] is sum(num_images for samples)

        if expand_size == 1:
            return input_ids, model_kwargs

        visual_keys = ["pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw", "second_per_grid_ts"]

        def _expand_dict_for_generation_visual(dict_to_expand):
            image_grid_thw = model_kwargs.get("image_grid_thw", None)
            video_grid_thw = model_kwargs.get("video_grid_thw", None)
            image_nums, video_nums = self._get_image_nums_and_video_nums(input_ids)

            def _repeat_interleave_samples(x, lengths, repeat_times):
                samples = torch.split(x, lengths)
                repeat_args = [repeat_times] + [1] * (x.dim() - 1)
                result = torch.cat([sample.repeat(*repeat_args) for sample in samples], dim=0)
                return result

            for key in dict_to_expand:
                if key == "pixel_values":
                    # split images into samples
                    samples = torch.split(image_grid_thw, list(image_nums))
                    # compute the sequence length of images for each sample
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "image_grid_thw":
                    # get the num of images for each sample
                    lengths = list(image_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "pixel_values_videos":
                    samples = torch.split(video_grid_thw, list(video_nums))
                    lengths = [torch.prod(sample, dim=1).sum() for sample in samples]
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "video_grid_thw":
                    lengths = list(video_nums)
                    dict_to_expand[key] = _repeat_interleave_samples(
                        dict_to_expand[key], lengths=lengths, repeat_times=expand_size
                    )
                elif key == "second_per_grid_ts":
                    if not isinstance(dict_to_expand[key], list):
                        raise TypeError(
                            f"Expected value for key '{key}' to be a list, but got {type(dict_to_expand[key])} instead."
                        )
                    tensor = torch.tensor(dict_to_expand[key])
                    lengths = list(video_nums)
                    tensor = _repeat_interleave_samples(tensor, lengths=lengths, repeat_times=expand_size)
                    dict_to_expand[key] = tensor.tolist()
            return dict_to_expand

        def _expand_dict_for_generation(dict_to_expand):
            for key in dict_to_expand:
                if (
                    key != "cache_position"
                    and dict_to_expand[key] is not None
                    and isinstance(dict_to_expand[key], torch.Tensor)
                    and key not in visual_keys
                ):
                    dict_to_expand[key] = dict_to_expand[key].repeat_interleave(expand_size, dim=0)
            return dict_to_expand

        # input_ids is required for expanding visual inputs
        # If input_ids is unavailable, visual inputs will not be used; therefore, there is no need to expand visual inputs.
        if input_ids is not None and input_ids.numel() != 0:
            model_kwargs = _expand_dict_for_generation_visual(model_kwargs)

        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)

        model_kwargs = _expand_dict_for_generation(model_kwargs)

        if is_encoder_decoder:
            if model_kwargs.get("encoder_outputs") is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

        return input_ids, model_kwargs




def get_qformer_position_id(q_former_list, raw_embeddings, raw_position_ids):
    q_former_outputs = q_former_list
    # 1. Get dimensions
    batch_size = q_former_outputs.shape[0]
    q_former_total_len = q_former_outputs.shape[1]
    raw_len = raw_embeddings.shape[1]
    total_len = q_former_total_len + raw_len
    device = q_former_outputs.device
    dtype = raw_embeddings.dtype
    # 2. Create sequential position IDs for Q-former outputs
    q_former_position_ids = torch.zeros(
        3,  # 3 dimensions for Qwen2.5
        batch_size,
        q_former_total_len,
        dtype=raw_position_ids.dtype,  # Use same dtype as raw_position_ids
        device=raw_embeddings.device
    )

    # Fill sequentially (all 3 dimensions get same values for Q-former outputs)
    for dim in range(3):
        q_former_position_ids[dim] = torch.arange(
            q_former_total_len, 
            device=device,
            dtype=raw_position_ids.dtype  # Use same dtype as raw_position_ids
        ).unsqueeze(0).expand(batch_size, -1)
    # 3. Ensure position_ids_4 starts after Q-former position IDs
    raw_position_ids = raw_position_ids.to(raw_embeddings.device)
    max_q_former_pos = q_former_position_ids.max().to(raw_position_ids.device)
    min_raw_pos = raw_position_ids.min()

    # Apply offset if needed
    raw_position_ids = raw_position_ids.clone().to(raw_embeddings.device)
    if min_raw_pos <= max_q_former_pos:
        offset = max_q_former_pos + 1 - min_raw_pos
        offset = offset.to(dtype=raw_position_ids.dtype)
        raw_position_ids += offset
    # 4. Concatenate position IDs
    final_position_ids = torch.cat([q_former_position_ids, raw_position_ids], dim=2)
    
    # 5. Concatenate embeddings
    q_former_outputs = q_former_outputs.to(raw_embeddings.device)
    concatenated_embeddings = torch.cat([
        q_former_outputs,
        raw_embeddings
    ], dim=1)

    # 6. Calculate final mrope_position_deltas
    # This is typically the max position ID + 1 - total sequence length
    max_position = final_position_ids.max()
    final_mrope_deltas = (max_position + 1 - total_len).unsqueeze(0).expand(batch_size, 1).to(q_former_outputs.device)

    return concatenated_embeddings, final_position_ids