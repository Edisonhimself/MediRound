from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM,\
                         MistralConfig, MistralModel, MistralForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaMistralConfig(MistralConfig):
    model_type = "llava_mistral"


class LlavaMistralModel(LlavaMetaModel, MistralModel):
    config_class = LlavaMistralConfig

    def __init__(self, config: MistralConfig):
        super(LlavaMistralModel, self).__init__(config)


class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = LlavaMistralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

                                                       
        self.post_init()

    def get_model(self):
        return self.model


                 
    def forward(
        self,

                   
        processed_cropped_images_encode = None,                                   
        processed_encode_bboxes = None,
        mask4cropped_image = None,                         
        mask4input_bbox = None,                            

        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels,
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids, attention_mask, past_key_values, labels, images
        )

                                           
                                                                                                                                                                          
        if processed_cropped_images_encode is not None or processed_encode_bboxes is not None:                
            inputs_embeds = self.replace_with_cropped_and_bbox(
                inputs_embeds,
                mask4cropped_image,
                processed_cropped_images_encode,
                mask4input_bbox,
                processed_encode_bboxes,
            )


                                                                                       
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
                                                
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
                                
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
                                               
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        if self.training:
            output_hidden_states = outputs.hidden_states
        else:
            output_hidden_states = hidden_states


                                                                               
                    
                            
                                  
                           
                     
                               
                                               
                                              
                                     
                                                         

                                                   
                         
                                                               
                                                                                                         

                                               
                                             
                                                              
                                                                
                                          
                                                    
                                           
                       

                                                 
                                                                                                     
                                                                                              

                                                     
                                                                                                  
                                                                                                     

                                         
                                                            
                                                                                                                          
                                                                                                       

                                          
                                                                                 

                                       
                                                                      
                                                
                                                                               


        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=output_hidden_states,                          
            attentions=outputs.attentions,
        )






    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):                                        
        if past_key_values and self.training==True:
            input_ids = input_ids[:, -1:]

                                                                                            
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images,
            }
        )
        return model_inputs
    
               
    def replace_with_cropped_and_bbox(
        self,
        inputs_embeds,
        mask4cropped_image,
        processed_cropped_images_encode,
        mask4input_bbox,
        processed_encode_bboxes,
    ):
        hidden_dim = inputs_embeds.size(-1)

                   
        inputs_flat = inputs_embeds.view(-1, hidden_dim)

        mask_img_flat = mask4cropped_image.view(-1)

        num_masked_imgs = mask_img_flat.sum().item()
        num_imgs = processed_cropped_images_encode.size(0)
        assert num_masked_imgs == num_imgs,\
            f"[Cropped] Mismatch: mask count={num_masked_imgs}, image count={num_imgs}"

        inputs_flat[mask_img_flat] = processed_cropped_images_encode

                                  
        mask_bbox_flat = mask4input_bbox.view(-1)

        num_masked_bboxes = mask_bbox_flat.sum().item()
        num_bboxes = processed_encode_bboxes.size(0)
        assert num_masked_bboxes == num_bboxes,\
            f"[BBox] Mismatch: mask count={num_masked_bboxes}, bbox count={num_bboxes}"

        inputs_flat[mask_bbox_flat] = processed_encode_bboxes

                     
        inputs_embeds = inputs_flat.view_as(inputs_embeds)

        return inputs_embeds
    


    def sample_from_logits(self, logits, temperature=0.7, top_p=0.9):
        """
        logits: (batch, vocab)
        return: sampled token ids (batch,)
        """
        logits = logits / temperature
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        mask = cum_probs > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits[mask] = -float('Inf')

        probs = F.softmax(sorted_logits, dim=-1)
        sampled_idx_in_sorted = torch.multinomial(probs, num_samples=1)
        sampled_idx = sorted_idx.gather(-1, sampled_idx_in_sorted)
        return sampled_idx.squeeze(-1)
    





AutoConfig.register("llava_mistral", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)
