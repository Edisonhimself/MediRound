import argparse
import datetime
import os
import random
import shutil
import sys
import time
from functools import partial

import cv2
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from transformers import CLIPImageProcessor

from model.mediround import MediRoundForCausalLM
from model.llava import conversation as conversation_lib
from utils.dataset import HybridDataset, ValDataset_Multi_Turn_Med_Seg, collate_fn
from utils.utils import (
    AverageMeter,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    ProgressMeter,
    Summary,
    diceCoefficientGPU,
    dict_to_cuda,
    intersectionAndUnionGPU,
)



def parse_args(args):
    parser = argparse.ArgumentParser(description="MediRound multi-turn training and evaluation")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--version", default=None, type=str)
    parser.add_argument(
        "--precision",
        default="bf16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for training or evaluation",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=1024, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument("--vision_tower", "--vision-tower", dest="vision_tower", required=True, type=str)
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--dataset_dir", required=True, type=str)
    parser.add_argument("--train_json", default=None, type=str)
    parser.add_argument("--eval_json", default=None, type=str)
    parser.add_argument("--output_dir", default="./outputs", type=str)
    parser.add_argument("--exp_name", default="mediround", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument("--batch_size", default=2, type=int, help="batch size per device per step")
    parser.add_argument("--grad_accumulation_steps", default=10, type=int)
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.0003, type=float)
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=0.5, type=float)
    parser.add_argument("--bce_loss_weight", default=2.0, type=float)
    parser.add_argument("--bbox_loss_weight", default=2.0, type=float)
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--weight", default=None, type=str)
    parser.add_argument("--stage", default=2, type=int, choices=[1, 2])
    parser.add_argument("--jcm_threshold", default=0.6, type=float)
    parser.add_argument("--vision_pretrained", required=True, type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=False)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2", "mistral_instruct"],
    )
    parsed_args = parser.parse_args(args)
    if parsed_args.model_name_or_path is None:
        parsed_args.model_name_or_path = parsed_args.version
    if parsed_args.model_name_or_path is None:
        parser.error("--model_name_or_path is required")
    if parsed_args.eval_only:
        if parsed_args.eval_json is None:
            parser.error("--eval_only requires --eval_json")
        if parsed_args.weight is None:
            parser.error("--eval_only requires --weight")
    else:
        if parsed_args.train_json is None:
            parser.error("training requires --train_json")
        if not parsed_args.no_eval and parsed_args.eval_json is None:
            parser.error("training with validation requires --eval_json or --no_eval")
        if parsed_args.stage == 2 and parsed_args.weight is None:
            parser.error("--stage 2 training requires --weight from a stage 1 checkpoint")
    parsed_args.version = parsed_args.model_name_or_path
    parsed_args.log_base_dir = parsed_args.output_dir
    return parsed_args





def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode():
                                   
                                             
                
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        gpu = rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        distributed = False
        return

    distributed = True
    dist_url = "env://"

    torch.cuda.set_device(gpu)
    dist_backend = "nccl"
    print(
        "| distributed init (rank {}, world {}): {}".format(
            rank, world_size, dist_url
        ),
        flush=True,
    )
    torch.distributed.init_process_group(
        backend=dist_backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        timeout=datetime.timedelta(
            days=365
        ),                                             
    )
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)






def main(args):
    init_distributed_mode()

    args = parse_args(args)
    stage2_flag = args.stage == 2
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
    else:
        writer = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.add_tokens("[SEG]")
    tokenizer.add_tokens("[BBOXINPUT]")
    tokenizer.add_tokens("[CROPPEDIMAGE]")
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.box_token_idx = tokenizer("[BBOXINPUT]", add_special_tokens=False).input_ids[0]
    args.croppedimage_token_idx = tokenizer("[CROPPEDIMAGE]", add_special_tokens=False).input_ids[0]

    if args.use_mm_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "seg_token_idx": args.seg_token_idx,
        "box_token_idx": args.box_token_idx,
        "croppedimage_token_idx": args.croppedimage_token_idx,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "bbox_loss_weight": args.bbox_loss_weight,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half

    model = MediRoundForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )
    model.config.stage = args.stage
    model.config.jcm_threshold = args.jcm_threshold
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    model.get_model().config.stage = args.stage
    model.get_model().config.jcm_threshold = args.jcm_threshold
    model.get_model().initialize_mediround_modules(model.get_model().config, stage2_flag)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    if args.lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            excluded_modules = [
                "visual_model",
                "vision_tower",
                "mm_projector",
                "text_hidden_fcs",
                "bbox_encoder",
                "mask_quality_MLP",
                "refine_MLP",
            ]
            for name, module in model.named_modules():
                if isinstance(module, cls) and all(x not in name for x in excluded_modules) and any(
                    x in name for x in lora_target_modules
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        lora_target_modules = find_linear_layers(model, args.lora_target_modules.split(","))
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.resize_token_embeddings(len(tokenizer))

    if stage2_flag:
        for n, p in model.named_parameters():
            if any(x in n for x in ["mask_quality_MLP", "refine_MLP"]):
                print("n: ", n, "p.shape: ", p.shape)
                p.requires_grad = True
            else:
                p.requires_grad = False
    else:
        trainable_names = ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs", "bbox_encoder"]
        for n, p in model.named_parameters():
            if any(x in n for x in trainable_names):
                print("n: ", n, "p.shape: ", p.shape)
                p.requires_grad = True

    if args.weight:
        all_state_dict = {}
        for filename in os.listdir(args.weight):
            if filename.endswith(".bin"):
                state_dict = torch.load(os.path.join(args.weight, filename), map_location="cpu")
                for key, value in state_dict.items():
                    if key in all_state_dict and all_state_dict[key].shape != value.shape:
                        print(f"Skipped checkpoint key with mismatched shape: {key}")
                        continue
                    all_state_dict[key] = value
        missing_keys, unexpected_keys = model.load_state_dict(all_state_dict, strict=False)
        if missing_keys:
            print("Missing checkpoint keys:")
            for key in missing_keys:
                print(f"  {key}")
        if unexpected_keys:
            raise ValueError(f"Unexpected checkpoint keys: {unexpected_keys}")

    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1
    if args.eval_only:
        train_dataset = None
    else:
        train_dataset = HybridDataset(
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            samples_per_epoch=args.batch_size * args.grad_accumulation_steps * args.steps_per_epoch * world_size,
            precision=args.precision,
            image_size=args.image_size,
            train_json=args.train_json,
        )

    if args.eval_only or not args.no_eval:
        val_dataset_multi_turn_med_seg = ValDataset_Multi_Turn_Med_Seg(
            args.eval_json,
            args.dataset_dir,
            tokenizer,
            args.vision_tower,
            args.image_size,
        )
        if train_dataset is None:
            print(f"Evaluating with {len(val_dataset_multi_turn_med_seg)} examples.")
        else:
            print(
                f"Training with {len(train_dataset)} examples and validating with {len(val_dataset_multi_turn_med_seg)} examples."
            )
    else:
        val_dataset_multi_turn_med_seg = None
        print(f"Training with {len(train_dataset)} examples.")

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {"enabled": args.precision == "fp16"},
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    if args.eval_only:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
    else:
        model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            training_data=train_dataset,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
            config=ds_config,
        )

    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("ckpt_of_epoch_", ""))
        print("resume training from {}, start from epoch {}".format(args.resume, args.start_epoch))

    if val_dataset_multi_turn_med_seg is not None:
        assert args.val_batch_size == 1
        val_sampler_multi_turn_med_seg = torch.utils.data.distributed.DistributedSampler(
            val_dataset_multi_turn_med_seg, shuffle=False, drop_last=False
        )
        val_loader_med_turn_med_seg = torch.utils.data.DataLoader(
            val_dataset_multi_turn_med_seg,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler_multi_turn_med_seg,
            collate_fn=partial(
                collate_fn,
                tokenizer=tokenizer,
                conv_type=args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                local_rank=args.local_rank,
            ),
        )
    else:
        val_loader_med_turn_med_seg = None

    train_iter = None if args.eval_only else iter(train_loader)
    best_score, cur_ciou = 0.0, 0.0

    if args.eval_only:
        if val_loader_med_turn_med_seg is None:
            raise ValueError("--eval_only requires evaluation data")
        validate_multi_turn_med_seg(
            val_loader_med_turn_med_seg,
            model_engine,
            0,
            writer,
            args,
            wise_evaluate=stage2_flag,
        )
        return

    for epoch in range(args.start_epoch, args.epochs):
        train_iter = train(train_loader, model_engine, epoch, scheduler, writer, train_iter, args)
        is_best = args.no_eval
        if not args.no_eval:
            giou_mt, ciou_mt = validate_multi_turn_med_seg(
                val_loader_med_turn_med_seg, model_engine, epoch, writer, args, wise_evaluate=stage2_flag
            )
            if epoch + 1 >= 10:
                is_best = True
            else:
                is_best = giou_mt > best_score
                best_score = max(giou_mt, best_score)
                cur_ciou = ciou_mt if is_best else cur_ciou

        if is_best:
            if epoch + 1 >= 10:
                save_dir = os.path.join(args.log_dir, "ckpt_model_epoch_{}".format(epoch + 1))
            else:
                save_dir = os.path.join(args.log_dir, "ckpt_model")

            if args.local_rank == 0:
                torch.save(
                    {"epoch": epoch},
                    os.path.join(args.log_dir, "meta_log_giou{:.3f}_ciou{:.3f}.pth".format(best_score, cur_ciou)),
                )
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir, "ckpt_of_epoch_{}".format(epoch + 1))

def train(
    train_loader,
    model,                                                 
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    ce_losses = AverageMeter("CeLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    mask_ref_losses = AverageMeter("MaskRefLoss", ":.4f")

    progress = ProgressMeter(
        args.steps_per_epoch,
        [
            batch_time,
            losses,
            ce_losses,
            mask_losses,
            mask_bce_losses,
            mask_dice_losses,
            mask_ref_losses,
        ],
        prefix="Epoch: [{}]".format(epoch),
    )

                          
    model.train()                                                     
    end = time.time()
    for global_step in range(args.steps_per_epoch):
        for i in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            data_time.update(time.time() - end)
            input_dict = dict_to_cuda(input_dict)

            if args.precision == "fp16":
                input_dict["images"] = input_dict["images"].half()
                input_dict["images_clip"] = input_dict["images_clip"].half()
            elif args.precision == "bf16":
                input_dict["images"] = input_dict["images"].bfloat16()
                input_dict["images_clip"] = input_dict["images_clip"].bfloat16()

                           
                                                         
                                                                               
                   
                                                      
                                                                            
                   
                if "cropped_images_encode" in input_dict and input_dict["cropped_images_encode"]:
                    input_dict["cropped_images_encode"] = [
                        x.bfloat16() if x is not None else None for x in input_dict["cropped_images_encode"]
                    ]
                if "encode_bboxes_list" in input_dict and input_dict["encode_bboxes_list"]:
                    input_dict["encode_bboxes_list"] = [
                        x.bfloat16() if x is not None else None for x in input_dict["encode_bboxes_list"]
                    ]
            else:
                input_dict["images"] = input_dict["images"].float()
                input_dict["images_clip"] = input_dict["images_clip"].float()


            output_dict = model(**input_dict)                                                    

            loss = output_dict["loss"]
            ce_loss = output_dict["ce_loss"]
            mask_bce_loss = output_dict["mask_bce_loss"]
            mask_dice_loss = output_dict["mask_dice_loss"]
            mask_loss = output_dict["mask_loss"]
            mask_ref_loss = output_dict["mask_ref_loss"]                             

            losses.update(loss.item(), input_dict["images"].size(0))
            ce_losses.update(ce_loss.item(), input_dict["images"].size(0))
            mask_bce_losses.update(mask_bce_loss.item(), input_dict["images"].size(0))
            mask_dice_losses.update(mask_dice_loss.item(), input_dict["images"].size(0))
            mask_losses.update(mask_loss.item(), input_dict["images"].size(0))
            mask_ref_losses.update(mask_ref_loss.item(), input_dict["images"].size(0))
            model.backward(loss)
            model.step()

                              
        batch_time.update(time.time() - end)
        end = time.time()

        if global_step % args.print_freq == 0:                            
            if args.distributed:
                batch_time.all_reduce()
                data_time.all_reduce()

                losses.all_reduce()
                ce_losses.all_reduce()
                mask_bce_losses.all_reduce()
                mask_dice_losses.all_reduce()
                mask_losses.all_reduce()
                mask_ref_losses.all_reduce()

            if args.local_rank == 0:
                progress.display(global_step + 1)
                writer.add_scalar("train/loss", losses.avg, global_step+epoch*args.steps_per_epoch)
                writer.add_scalar("train/ce_loss", ce_losses.avg, global_step+epoch*args.steps_per_epoch)
                writer.add_scalar(
                    "train/mask_bce_loss", mask_bce_losses.avg, global_step+epoch*args.steps_per_epoch
                )
                writer.add_scalar(
                    "train/mask_dice_loss", mask_dice_losses.avg, global_step+epoch*args.steps_per_epoch
                )
                writer.add_scalar("train/mask_loss", mask_losses.avg, global_step+epoch*args.steps_per_epoch)
                      
                writer.add_scalar("train/mask_ref_loss", mask_ref_losses.avg, global_step+epoch*args.steps_per_epoch)
                writer.add_scalar(
                    "metrics/total_secs_per_batch", batch_time.avg, global_step+epoch*args.steps_per_epoch
                )
                writer.add_scalar(
                    "metrics/data_secs_per_batch", data_time.avg, global_step+epoch*args.steps_per_epoch
                )

            batch_time.reset()
            data_time.reset()
            losses.reset()
            ce_losses.reset()
            mask_bce_losses.reset()
            mask_dice_losses.reset()
            mask_losses.reset()
            mask_ref_losses.reset()

        if global_step != 0:
            curr_lr = scheduler.get_last_lr()
            if args.local_rank == 0:
                writer.add_scalar("train/lr", curr_lr[0], global_step)

                     
                        

    return train_iter


def get_bbox(
    mask
):
    h, w = mask.shape[:2]

    x = mask.any(1).nonzero()[0]
    y = mask.any(0).nonzero()[0]
    box = [x[0], y[0], x[-1] + 1, y[-1] + 1]               

    return box

def get_bounding_box_from_mask(mask):

    if isinstance(mask, torch.Tensor):
        if mask.is_cuda:
            mask = mask.cpu()

    mask = np.array(mask)                   
    height, width = mask.shape

                          
    y_indices, x_indices = np.where(mask == 255)
    if len(x_indices) == 0 or len(y_indices) == 0:
        raise ValueError("Mask does not contain any white regions.")

    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)

         
    x_min_norm, x_max_norm = x_min / width, x_max / width
    y_min_norm, y_max_norm = y_min / height, y_max / height

    return [x_min_norm, y_min_norm, x_max_norm, y_max_norm]



def validate_multi_turn_med_seg(val_loader, model_engine, epoch, writer, args, teacher_forcing = False, wise_evaluate = False):
                         
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)


                   
                                                                    
                                                                    
    
              
    acc_dice_meter = AverageMeter("Dice", ":6.3f", Summary.SUM)


                                           
    round_n_masks_list = [[] for _ in range(12)]                                                      
    round_n_gt_masks_list = [[] for _ in range(12)]

    all_masks_list = []
    all_gt_masks_list = []

    clip_image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower)



    model_engine.eval()
    for input_dict in tqdm.tqdm(val_loader):
        torch.cuda.empty_cache()

        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()

        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
                       
            input_dict["cropped_images_encode"] = [
                x.bfloat16() for x in input_dict["cropped_images_encode"]
            ]
            input_dict["encode_bboxes_list"] = [
                x.bfloat16() for x in input_dict["encode_bboxes_list"]
            ]

        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()


        if len(input_dict["masks_list"]) > 1 or len(input_dict["ref_masks_list"]) > 1:
             raise RuntimeError("size len error!")
        assert input_dict["masks_list"][0].shape[0] == input_dict["ref_masks_list"][0].shape[0], "masks_list and masks_ref_list must have the same length"
              
        round_num = input_dict["masks_list"][0].shape[0]

                                                                  
        image_path_current = input_dict["image_paths"][0]
        image_current4cropped = Image.open(os.path.join(args.dataset_dir, image_path_current))
        image_current4cropped = np.array(image_current4cropped.convert('RGB'))

                      
        for round_current in range(round_num):
            with torch.no_grad():

                               
                if wise_evaluate:
                    input_round_indices = [(i,b) for i, (a, b) in enumerate(input_dict["input_round_id_and_input_image_id_List"][0]) if a == round_current]                                                    
                    input_dict['input_round_indices'] = bool(input_round_indices)
                    input_dict['round_current_4_mt'] = round_current


                output_dict = model_engine(**input_dict)                  

                                                                              
                assert len(output_dict["pred_masks"]) == 1                    
                pred_masks_ori = (output_dict["pred_masks"][0] > 0).int()
                gt_masks_ori = output_dict["gt_masks"][0].int() 
                pred_mask_current = pred_masks_ori[round_current:round_current+1]                
                gt_mask_current = gt_masks_ori[round_current:round_current+1]

                                 
                round_n_masks_list[round_current].append(pred_mask_current)
                round_n_gt_masks_list[round_current].append(gt_mask_current)
                             
                all_masks_list.append(pred_mask_current)
                all_gt_masks_list.append(gt_mask_current)

                                                            
                input_round_indices = [(i,b) for i, (a, b) in enumerate(input_dict["input_round_id_and_input_image_id_List"][0]) if a == round_current]                                                    
                if input_round_indices and not teacher_forcing:                              
                                    
                    original_input_flag = 0
                                           
                    pred_mask_current_numpy = pred_mask_current.squeeze(0).cpu().numpy() 
                    pred_mask_current_numpy = (pred_mask_current_numpy * 255).astype(np.uint8)

                    image_masked_current = cv2.bitwise_and(image_current4cropped, image_current4cropped, mask=pred_mask_current_numpy)
                                   
                    if int((image_masked_current[..., 0] > 0).sum()) <= 9:
                        pred_mask_current_numpy[:] = 255
                        image_masked_current = cv2.bitwise_and(image_current4cropped, image_current4cropped, mask=pred_mask_current_numpy)
                        original_input_flag = 1

                    try:
                        x0, y0, x1, y1 = get_bbox(image_masked_current)
                    except Exception as e:
                        print(f"Warning: reference update failed: {e}")
                        continue           
                    max_width = max(x1-x0,y1-y0)
                    image_masked_cropped = image_masked_current[x0:x1,y0:y1]                        
                    image_masked_cropped_padded = np.zeros((max_width,max_width,image_masked_current.shape[-1]),dtype=image_masked_current.dtype)                   
                    image_masked_cropped_padded[:image_masked_cropped.shape[0],:image_masked_cropped.shape[1]] = image_masked_cropped

                                         
                    inputs_cropped_image_current = clip_image_processor.preprocess(Image.fromarray(image_masked_cropped_padded), return_tensors="pt")["pixel_values"][0]     

                                 
                    labe4box_current = (pred_mask_current.squeeze(0)).to(torch.long) * 255

                                                
                    if original_input_flag == 1:
                        labe4box_current[:] = 255

                    inputs_bbox_coords_current = get_bounding_box_from_mask(labe4box_current)
                    inputs_bbox_coords_current = torch.tensor(inputs_bbox_coords_current)

                            
                    for i, b in input_round_indices:                                                                    
                        input_dict["cropped_images_encode"][0][b] = inputs_cropped_image_current
                        input_dict["encode_bboxes_list"][0][b] = inputs_bbox_coords_current
                

    round_n_masks_list_tensor = []
    round_n_gt_masks_list_tensor = []
                              
    round_n_masks_list = [item for item in round_n_masks_list if item]
    round_n_gt_masks_list = [item for item in round_n_gt_masks_list if item]
                   
    round_n_masks_list_tensor = round_n_masks_list
    round_n_gt_masks_list_tensor = round_n_gt_masks_list

                                                                     
    for idx, (masks_list_round_i, output_list_round_i) in enumerate(zip(round_n_gt_masks_list_tensor, round_n_masks_list_tensor)):

        intersection, union, acc_iou, dice = 0.0, 0.0, 0.0, 0.0
        intersection_meter.reset()
        union_meter.reset()
        acc_iou_meter.reset()
        acc_dice_meter.reset()

        for mask_i, output_i in zip(masks_list_round_i, output_list_round_i):                
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0                    
                      
            dice += diceCoefficientGPU(output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255)
            dice[union_i == 0] += 1.0                    
        
        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / len(masks_list_round_i)
                  
        dice = dice.cpu().numpy() / len(masks_list_round_i)
        
        intersection_meter.update(intersection), union_meter.update(
            union
        ), acc_iou_meter.update(acc_iou, n=len(masks_list_round_i))
        
                  
        acc_dice_meter.update(dice, n=len(masks_list_round_i))

        
        intersection_meter.all_reduce()
        union_meter.all_reduce()
        acc_iou_meter.all_reduce()
        
                  
        acc_dice_meter.all_reduce()

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
        ciou = iou_class[1]
        giou = acc_iou_meter.avg[1]
        
                  
        dice = acc_dice_meter.avg[1]

        if args.local_rank == 0:
            writer.add_scalar(f"val/giou_round_{idx}", giou, epoch)
            writer.add_scalar(f"val/ciou_round_{idx}", ciou, epoch)
            writer.add_scalar(f"val/dice_round_{idx}", dice, epoch)
            print(f"giou_round_{idx}: {giou:.4f}, ciou_round_{idx}: {ciou:.4f}")
            print(f"dice_round_{idx}: {dice:.4f}")



         
    all_masks_list_tensor = all_masks_list
    all_gt_masks_list_tensor = all_gt_masks_list

              
    intersection, union, acc_iou, dice = 0.0, 0.0, 0.0, 0.0
    intersection_meter.reset()
    union_meter.reset()
    acc_iou_meter.reset()
    acc_dice_meter.reset()

    for mask_i, output_i in zip(all_gt_masks_list_tensor, all_masks_list_tensor):
        intersection_i, union_i, _ = intersectionAndUnionGPU(
            output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
        )
        intersection += intersection_i
        union += union_i
        acc_iou += intersection_i / (union_i + 1e-5)
        acc_iou[union_i == 0] += 1.0                    
                  
        dice += diceCoefficientGPU(output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255)
        dice[union_i == 0] += 1.0                    
    
    intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
    acc_iou = acc_iou.cpu().numpy() / len(all_gt_masks_list_tensor)
              
    dice = dice.cpu().numpy() / len(all_gt_masks_list_tensor)
    
    intersection_meter.update(intersection), union_meter.update(
        union
    ), acc_iou_meter.update(acc_iou, n=len(all_gt_masks_list_tensor))
    
              
    acc_dice_meter.update(dice, n=len(all_gt_masks_list_tensor))

    
    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()
    
              
    acc_dice_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    
              
    dice = acc_dice_meter.avg[1]

    if args.local_rank == 0:
        writer.add_scalar(f"val/giou_all", giou, epoch)
        writer.add_scalar(f"val/ciou_all", ciou, epoch)
        writer.add_scalar(f"val/dice_all", dice, epoch)
        print(f"giou_all: {giou:.4f}, ciou_all: {ciou:.4f}")
        print(f"dice_all: {dice:.4f}")



    return giou, ciou














if __name__ == "__main__":
    main(sys.argv[1:])
