from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BitsAndBytesConfig, CLIPVisionModel

from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         DEFAULT_IMAGE_PATCH_TOKEN)

                                                                             
                                                                        

             
from .llava.model.language_model.llava_mistral import (LlavaMistralForCausalLM,
                                                       LlavaMistralModel)

from .segment_anything import build_sam_vit_h,build_sam_vit_b

from .bbox_head.init_bbox_decoder import init_bbox_head, init_router, MaskQualityMLP, RefineMLP
from .bbox_head.bbox_utils import bbox_giou, xywh2xyxy

def dice_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
    scale=1000,             
    eps=1e-6,
):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1, 2)
    targets = targets.flatten(1, 2)
    numerator = 2 * (inputs / scale * targets).sum(-1)
    denominator = (inputs / scale).sum(-1) + (targets / scale).sum(-1)
    loss = 1 - (numerator + eps) / (denominator + eps)
    loss = loss.sum() / (num_masks + 1e-8)
    return loss


def compute_iou(pred_mask, gt_mask, threshold=0):
                               
    pred_mask = (pred_mask > threshold).to(torch.bool)           
    gt_mask = (gt_mask > threshold).to(torch.bool)           
    
                  
    intersection = (pred_mask & gt_mask).sum(dim=(1, 2))                        
    union = (pred_mask | gt_mask).sum(dim=(1, 2))                        
    
                        
    iou = intersection / (union + 1e-6)         
    iou = iou.unsqueeze(1)                        

    return iou


def sigmoid_ce_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    num_masks: float,
):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    loss = loss.flatten(1, 2).mean(1).sum() / (num_masks + 1e-8)
    return loss



def box_loss(pred:torch.Tensor,
             gt:torch.Tensor):
    l1_loss = nn.L1Loss(reduction="mean")(xywh2xyxy(pred), gt)
    giou_loss = bbox_giou(pred, gt)
    return l1_loss + giou_loss





class MediRoundMetaModel:
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(MediRoundMetaModel, self).__init__(config)

        self.config = config
        if not hasattr(self.config, "train_mask_decoder"):                          
            self.config.train_mask_decoder = kwargs["train_mask_decoder"]
            self.config.out_dim = kwargs["out_dim"]
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
        else:
            self.vision_pretrained = kwargs.get("vision_pretrained", None)
            self.initialize_mediround_modules(self.config)
                         
            raise SystemError("Model initialization error")

    def initialize_mediround_modules(self, config, stage2_flag):
             
                     
                                                                     
        self.visual_model = build_sam_vit_b(self.vision_pretrained)
        for param in self.visual_model.parameters():
            param.requires_grad = False
        if config.train_mask_decoder:
            self.visual_model.mask_decoder.train()
            for param in self.visual_model.mask_decoder.parameters():
                if not stage2_flag:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

                          
        in_dim = config.hidden_size
        out_dim = config.out_dim
        text_fc = [
            nn.Linear(in_dim, in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),           
            nn.Dropout(0.0),
        ]
        self.text_hidden_fcs = nn.ModuleList([nn.Sequential(*text_fc)])
        self.text_hidden_fcs.train()
        for param in self.text_hidden_fcs.parameters():
            if not stage2_flag:
                param.requires_grad = True
            else:
                param.requires_grad = False

                        
        self.bbox_encoder = nn.Linear(4,self.config.hidden_size)
        for param in self.bbox_encoder.parameters():
            if not stage2_flag:
                param.requires_grad = True
            else:
                param.requires_grad = False
            

        if stage2_flag:
            self.mask_quality_MLP = MaskQualityMLP()
            for param in self.mask_quality_MLP.parameters():
                param.requires_grad = True
                
            self.refine_MLP = RefineMLP()
            for param in self.refine_MLP.parameters():
                param.requires_grad = True

    def initialize_lisa_modules(self, config, stage2_flag):
        return self.initialize_mediround_modules(config, stage2_flag)



            




                           
                                                          
                                                    
                                        
                                                    
                                        


             
                                                  
class MediRoundModel(MediRoundMetaModel, LlavaMistralModel):
    def __init__(
        self,
        config,
        **kwargs,
    ):
        super(MediRoundModel, self).__init__(config, **kwargs)

        self.config.use_cache = False
        self.config.vision_tower = self.config.mm_vision_tower
        self.config.mm_vision_select_feature = "patch"
        self.config.image_aspect_ratio = "square"
        self.config.image_grid_pinpoints = None
        self.config.tune_mm_mlp_adapter = False
        self.config.freeze_mm_mlp_adapter = True
        self.config.pretrain_mm_mlp_adapter = None
        self.config.mm_use_im_patch_token = False


             
                                               
class MediRoundForCausalLM(LlavaMistralForCausalLM):
    def __init__(
        self,
        config,
        **kwargs,
    ):
                
        if not hasattr(config, "train_mask_decoder"):                                 
            config.mm_use_im_start_end = kwargs.pop("use_mm_start_end", True)
            config.mm_vision_tower = kwargs.get(
                "vision_tower", "openai/clip-vit-large-patch14"
            )
            self.ce_loss_weight = kwargs.pop("ce_loss_weight", None)
            self.dice_loss_weight = kwargs.pop("dice_loss_weight", None)
            self.bce_loss_weight = kwargs.pop("bce_loss_weight", None)

                                                    
            self.bbox_loss_weight = kwargs.pop("bbox_loss_weight", None)

        else:

            config.mm_vision_tower = config.vision_tower
                         
            raise SystemError("Model initialization error")
            
        self.seg_token_idx = kwargs.pop("seg_token_idx")
        self.box_token_idx = kwargs.pop("box_token_idx")
        self.croppedimage_token_idx = kwargs.pop("croppedimage_token_idx")               

        super().__init__(config)

        self.model = MediRoundModel(config, **kwargs)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)




                                                       
        self.post_init()

    def get_visual_embs(self, pixel_values: torch.FloatTensor):
        with torch.no_grad():
            image_embeddings_list = []
            for i in range(pixel_values.shape[0]):
                torch.cuda.empty_cache()
                image_embeddings = self.model.visual_model.image_encoder(
                    pixel_values[i].unsqueeze(0)
                )
                image_embeddings_list.append(image_embeddings)
            torch.cuda.empty_cache()
            image_embeddings = torch.cat(image_embeddings_list, 0)
        return image_embeddings

    def forward(self, **kwargs):
        if "past_key_values" in kwargs:
            return super().forward(**kwargs)
        return self.model_forward(**kwargs)

    def model_forward(
        self,
        images: torch.FloatTensor,                                      
        images_clip: torch.FloatTensor,                                            
        input_ids: torch.LongTensor,
        labels: torch.LongTensor,                                                        
        attention_masks: torch.LongTensor,
        offset: torch.LongTensor,
        masks_list: List[torch.FloatTensor],                        

        ref_masks_list: List[torch.FloatTensor],                                      
        cropped_images_encode: torch.FloatTensor,


        label_list: List[torch.Tensor],                 
        resize_list: List[tuple],
        encode_bboxes_list: List[torch.FloatTensor],       
        inference: bool = False,
        input_round_indices: bool = False,
        round_current_4_mt=None,
        **kwargs,
    ):  
        
        num_token_per_mask = 0

                                   
        image_embeddings = self.get_visual_embs(images)
        batch_size = image_embeddings.shape[0]
        assert batch_size == len(offset) - 1

        seg_token_mask = input_ids[:, 1:] == self.seg_token_idx
                                                                  
        seg_token_mask = torch.cat(
            [
                seg_token_mask,
                torch.zeros((seg_token_mask.shape[0], 1)).bool().cuda(),
            ],
            dim=1,
        )


                                                                                                                                                             
        seg_token_mask = torch.cat(
                         
                                                                                          
            [torch.zeros((seg_token_mask.shape[0], 575)).bool().cuda(), seg_token_mask],
            dim=1,
        )

                                 
        processed_cropped_images_encode, processed_encode_bboxes = self.process_extra_replacement_data(cropped_images_encode, encode_bboxes_list)

                                                                      
        mask4cropped_image = (input_ids == self.croppedimage_token_idx)
        mask4cropped_image = torch.cat([torch.zeros((mask4cropped_image.shape[0], 575)).bool().cuda(), mask4cropped_image], dim=1)
        mask4input_bbox = (input_ids == self.box_token_idx)
        mask4input_bbox = torch.cat([torch.zeros((mask4input_bbox.shape[0], 575)).bool().cuda(), mask4input_bbox], dim=1)
        

        if inference:
            n_batch = 1
            length = input_ids.shape[0]
            assert images_clip.shape[0] == 1
            images_clip_extend = images_clip.expand(length, -1, -1, -1).contiguous()

            output_hidden_states = []
            for i in range(n_batch):
                start_i, end_i = i * length, min((i + 1) * length, input_ids.shape[0])
                output_i = super().forward(
                               
                    processed_cropped_images_encode = processed_cropped_images_encode,
                    processed_encode_bboxes = processed_encode_bboxes,
                    mask4cropped_image = mask4cropped_image,
                    mask4input_bbox = mask4input_bbox,
                    
                    images=images_clip_extend[: end_i - start_i],
                    attention_mask=attention_masks[start_i:end_i],
                    input_ids=input_ids[start_i:end_i],
                    output_hidden_states=True,
                )
                output_hidden_states.append(output_i.hidden_states)
                torch.cuda.empty_cache()

            output_hidden_states_list = []
            output_hidden_states_level = torch.cat(output_hidden_states, dim=0)
            output_hidden_states_list.append(output_hidden_states_level)
            output_hidden_states = output_hidden_states_list
            output = None

        else:
            images_clip_list = []
            for i in range(len(offset) - 1):
                start_i, end_i = offset[i], offset[i + 1]
                images_clip_i = (
                    images_clip[i]
                    .unsqueeze(0)
                    .expand(end_i - start_i, -1, -1, -1)
                    .contiguous()
                )
                images_clip_list.append(images_clip_i)
            images_clip = torch.cat(images_clip_list, dim=0)

            output = super().forward(
                           
                processed_cropped_images_encode = processed_cropped_images_encode,
                processed_encode_bboxes = processed_encode_bboxes,
                mask4cropped_image = mask4cropped_image,
                mask4input_bbox = mask4input_bbox,

                images=images_clip,
                attention_mask=attention_masks,
                input_ids=input_ids,
                labels=labels,
                output_hidden_states=True,
            )
            output_hidden_states = output.hidden_states

        hidden_states = []
                                                 

        
        
        assert len(self.model.text_hidden_fcs) == 1
        hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states[-1]))
                                                        
                                                              

        last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
                                                                                        

        pred_embeddings = last_hidden_state[seg_token_mask]
                                                                          
        


                                                                                                                                        
        seg_token_counts = seg_token_mask.int().sum(-1)          
                                                                          

        seg_token_offset = seg_token_counts.cumsum(-1)
        seg_token_offset = torch.cat(
            [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
        )

        seg_token_offset = seg_token_offset[offset]

        assert seg_token_offset[-1] == len(pred_embeddings),\
            f"Incomplete embedding split: seg_token_offset[-1]={seg_token_offset[-1]}, pred_embeddings={len(pred_embeddings)}"

        pred_embeddings_ = []
                                     
        for i in range(len(seg_token_offset) - 1):
            start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
            pred_embeddings_.append(pred_embeddings[start_i:end_i])
                                                                                 
        pred_embeddings = pred_embeddings_                                      


        multimask_output = False
        mask_probs = []
        pred_masks = [] 
        scores_list = []
        gt_mask_ref_valid_list = []

        pred_masks_not_refine = []
        for i in range(len(pred_embeddings)):                                      

            mask_tokens = pred_embeddings[i]                 

            stage2_flag = getattr(self.config, "stage", 2) == 2
            wise_evaluate_flag = stage2_flag
            jcm_threshold = getattr(self.config, "jcm_threshold", 0.6)


            if inference and wise_evaluate_flag and input_round_indices:
                score_temp = self.model.mask_quality_MLP(mask_tokens[round_current_4_mt:round_current_4_mt+1])
                if score_temp.item() <= jcm_threshold:
                    refined_tokens = self.model.refine_MLP(mask_tokens[round_current_4_mt:round_current_4_mt+1])                        
                    mask_tokens[round_current_4_mt:round_current_4_mt+1] = refined_tokens
                    text_embeds = mask_tokens.unsqueeze(1)
                else:
                    text_embeds = mask_tokens.unsqueeze(1) 
            
            elif not inference and stage2_flag:

                refined_tokens = self.model.refine_MLP(mask_tokens)

                                     
                refined_tokens_unsqueezed = refined_tokens.unsqueeze(1)                             
                mask_tokens_unsqueezed = mask_tokens.unsqueeze(1)                             

                                                          
                text_embeds = torch.cat([refined_tokens_unsqueezed, mask_tokens_unsqueezed], dim=0)
                scores_list.append(self.model.mask_quality_MLP(mask_tokens))
            
            else:
                text_embeds = mask_tokens.unsqueeze(1)                       
            
                
            (
                sparse_embeddings,
                dense_embeddings,
            ) = self.model.visual_model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
                                                                                                                                                                                       
                text_embeds=text_embeds,
            )
            sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
            low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                image_embeddings=image_embeddings[i].unsqueeze(0),                                     
                image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
            )
            pred_mask = self.model.visual_model.postprocess_masks(
                low_res_masks,
                input_size=resize_list[i],
                original_size=label_list[i].shape,                             
            )
            if stage2_flag and not inference:
                pred_masks.append(pred_mask[:(pred_mask.shape[0])//2, :][:,0])                          

                       
                pred_masks_not_refine.append(pred_mask[pred_mask.shape[0]//2:, :][:,0])                                     

            else:
                pred_masks.append(pred_mask[:, 0])
                                                         



        model_output = output
        gt_masks = masks_list
        gt_masks_ref = gt_mask_ref_valid_list              

        if inference:
                                  
            return {
                                           
                "pred_masks": pred_masks,
                "gt_masks": gt_masks,
            }

        output = model_output.logits

        if not inference and not stage2_flag:
            ce_loss = model_output.loss
            ce_loss = ce_loss * self.ce_loss_weight
        else:
            ce_loss = 0

        mask_bce_loss = 0
        mask_dice_loss = 0
                       
        num_masks = 0

        mask_quality_bce_loss = 0
 
        mask_ref_bce_loss = 0
        mask_ref_dice_loss = 0
        num_masks_ref = 0

        for batch_idx in range(len(pred_masks)):
            gt_mask = gt_masks[batch_idx]
                                                                                                                                                           
            pred_mask = pred_masks[batch_idx]

            assert (
                gt_mask.shape[0] == pred_mask.shape[0]
            ), "gt_mask.shape: {}, pred_mask.shape: {}".format(
                gt_mask.shape, pred_mask.shape
            )
            mask_bce_loss += (
                sigmoid_ce_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )
            mask_dice_loss += (
                dice_loss(pred_mask, gt_mask, num_masks=gt_mask.shape[0])
                * gt_mask.shape[0]
            )


            if not inference and stage2_flag:
                pred_mask_not_refine_item = pred_masks_not_refine[batch_idx]
                                                                                 
                scores_gt = compute_iou(pred_mask_not_refine_item, gt_mask)
                mask_quality_bce_loss += (
                    F.binary_cross_entropy(scores_list[batch_idx].to(torch.bfloat16), scores_gt.to(torch.bfloat16),  reduction="sum")
                )

            num_masks += gt_mask.shape[0]

            mask_ref_bce_loss += 0
            mask_ref_dice_loss += 0
            num_masks_ref += 0

        mask_bce_loss = self.bce_loss_weight * mask_bce_loss / (num_masks + 1e-8)
        mask_dice_loss = self.dice_loss_weight * mask_dice_loss / (num_masks + 1e-8)
        mask_loss = mask_bce_loss + mask_dice_loss

                   
        mask_ref_bce_loss = self.bce_loss_weight * mask_ref_bce_loss / (num_masks_ref + 1e-8)
        mask_ref_dice_loss = self.dice_loss_weight * mask_ref_dice_loss / (num_masks_ref + 1e-8)
        mask_ref_loss = mask_ref_bce_loss + mask_ref_dice_loss

                                                
        if not inference and not stage2_flag:
            loss = ce_loss + mask_loss + mask_ref_loss
        else:
            loss = mask_loss 

        mask_ref_loss = torch.tensor(mask_ref_loss)


        if not inference and stage2_flag:
            mask_quality_bce_loss = 2 * mask_quality_bce_loss / (num_masks + 1e-8)
            loss = loss + mask_quality_bce_loss



        if not inference and stage2_flag:
            return {
                "loss": loss,
                "ce_loss": mask_quality_bce_loss,              
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
                "mask_ref_loss": mask_ref_loss 
            }

        else:
            return {
                "loss": loss,
                "ce_loss": ce_loss,
                "mask_bce_loss": mask_bce_loss,
                "mask_dice_loss": mask_dice_loss,
                "mask_loss": mask_loss,
                "mask_ref_loss": mask_ref_loss 
            }






               
    def process_extra_replacement_data(self,cropped_images_encode, encode_bboxes_list):
                                                                                                                                                                                         

        cropped_images_encode = [x for x in cropped_images_encode if isinstance(x, torch.Tensor) and x.numel() > 0]
        encode_bboxes_list = [x for x in encode_bboxes_list if isinstance(x, torch.Tensor) and x.numel() > 0]

                             
        if len(cropped_images_encode) == 0 or len(encode_bboxes_list) == 0:
            return None, None
    
        all_crops = torch.cat(cropped_images_encode, dim=0)                                  
                    
        all_features = self.encode_images(all_crops, features='cls') 
        all_cropped_image_features = all_features.squeeze(1)                              
        
        device = all_cropped_image_features.device
        all_bboxes = torch.cat(encode_bboxes_list, dim=0).to(device)                        
        all_bbox_features = self.model.bbox_encoder(all_bboxes)
        
        return all_cropped_image_features, all_bbox_features

              

    def evaluate(
        self,
        images_clip,
        images,
        input_ids,
        resize_list,
        label_list,
        original_size_list=None,
        max_new_tokens=32,
        tokenizer=None,
    ):
        with torch.no_grad():
            outputs = self.generate(
                images=images_clip,
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                num_beams=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            output_hidden_states = outputs.hidden_states[-1]
            output_ids = outputs.sequences

            seg_token_mask = output_ids[:, 1:] == self.seg_token_idx
                                                                                                          
            seg_token_mask = torch.cat(
                [
                                 
                                                                                
                    torch.zeros((seg_token_mask.shape[0], 575)).bool().cuda(),
                    seg_token_mask,
                ],
                dim=1,
            )

            hidden_states = []
            hidden_states_4bbox = []

            assert len(self.model.text_hidden_fcs) == 1
            hidden_states.append(self.model.text_hidden_fcs[0](output_hidden_states))
            hidden_states_4bbox.append(output_hidden_states)

            last_hidden_state = torch.stack(hidden_states, dim=-1).sum(dim=-1)
            last_hidden_state_4bbox = torch.stack(hidden_states_4bbox, dim=-1).sum(dim=-1)

            pred_embeddings = last_hidden_state[seg_token_mask]
            pred_embeddings_4bbox = last_hidden_state_4bbox[seg_token_mask]

            seg_token_counts = seg_token_mask.int().sum(-1)          
            seg_token_offset = seg_token_counts.cumsum(-1)
            seg_token_offset = torch.cat(
                [torch.zeros(1).long().cuda(), seg_token_offset], dim=0
            )

            pred_embeddings_ = []
            pred_embeddings_4bbox_ = []
            for i in range(len(seg_token_offset) - 1):
                start_i, end_i = seg_token_offset[i], seg_token_offset[i + 1]
                pred_embeddings_.append(pred_embeddings[start_i:end_i])
                pred_embeddings_4bbox_.append(pred_embeddings_4bbox[start_i:end_i])
            pred_embeddings = pred_embeddings_                                      
            pred_embeddings_4bbox = pred_embeddings_4bbox_
        
                    
            pred_bboxes = []
            for i in range(len(pred_embeddings_4bbox)):
                pred_bbox = self.model.bbox_decoder(pred_embeddings_4bbox[i])                                                 
                pred_bboxes.append(pred_bbox)
            if pred_bboxes[0].shape[0] !=1:
                print("ATTENTION")

            image_embeddings = self.get_visual_embs(images)

            multimask_output = False
            pred_masks = []
            for i in range(len(pred_embeddings)):
                (
                    sparse_embeddings,
                    dense_embeddings,
                ) = self.model.visual_model.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                    text_embeds=pred_embeddings[i].unsqueeze(1),
                )

                sparse_embeddings = sparse_embeddings.to(pred_embeddings[i].dtype)
                low_res_masks, iou_predictions = self.model.visual_model.mask_decoder(
                    image_embeddings=image_embeddings[i].unsqueeze(0),
                    image_pe=self.model.visual_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=multimask_output,
                )
                pred_mask = self.model.visual_model.postprocess_masks(
                    low_res_masks,
                    input_size=resize_list[i],
                    original_size=label_list[i].shape,
                )
                pred_masks.append(pred_mask[:, 0])


        return output_ids, pred_masks, pred_bboxes


# Backward-compatible public names for checkpoints and downstream code that
# imported the original LISA classes.
LisaMetaModel = MediRoundMetaModel
LisaModel = MediRoundModel
LISAForCausalLM = MediRoundForCausalLM
