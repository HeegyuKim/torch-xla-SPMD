import torch
from typing import Optional, List



def make_forward_inputs(
    model, 
    input_ids: torch.LongTensor = None,
    pixel_values: torch.FloatTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    vision_feature_layer: Optional[int] = None,
    vision_feature_select_strategy: Optional[str] = None,
    ):
    inputs_embeds = model.get_input_embeddings()(input_ids)

    # 2. Merge text and images
    if pixel_values is not None and input_ids.shape[1] != 1:
        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
        selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

        if vision_feature_select_strategy == "default":
            selected_image_feature = selected_image_feature[:, 1:]
        elif vision_feature_select_strategy == "full":
            selected_image_feature = selected_image_feature
        else:
            raise ValueError(
                f"Unexpected select feature strategy: {model.config.vision_feature_select_strategy}"
            )

        image_features = self.multi_modal_projector(selected_image_feature)
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, position_ids
        )
        if labels is None:
            labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)