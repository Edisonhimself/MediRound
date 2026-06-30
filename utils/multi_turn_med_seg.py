import copy
import json
import os
import random
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPImageProcessor

from model.bbox_head.bbox_utils import get_bounding_box_from_mask
from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

BRACKET_TYPES = ["IMAGE256", "MASK-ENCODE", "BOX-ENCODE", "MASK-DECODE", "REF-DECODE"]


def find_brackets(text):
    matches = re.compile(r"\[[^\]]+\]").findall(text)
    return [bracket for bracket in matches if any(kind in bracket for kind in BRACKET_TYPES)]


class MultiTurnMedSegDataset(torch.utils.data.Dataset):
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    img_size = 1024
    ignore_label = 255

    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        data_json,
        samples_per_epoch=None,
        precision="fp32",
        image_size=1024,
        random_sample=True,
    ):
        self.base_image_dir = base_image_dir
        self.tokenizer = tokenizer
        self.precision = precision
        self.image_size = image_size
        self.samples_per_epoch = samples_per_epoch
        self.random_sample = random_sample
        self.transform = ResizeLongestSide(image_size)
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        with open(data_json, "r") as f:
            self.list_data_dict = json.load(f)
        if random_sample:
            random.shuffle(self.list_data_dict)
        print(f"Loaded {len(self.list_data_dict)} multi-turn conversations from {os.path.basename(data_json)}.")

    def __len__(self):
        if self.samples_per_epoch is not None:
            return self.samples_per_epoch
        return len(self.list_data_dict)

    def preprocess(self, x):
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        return F.pad(x, (0, self.img_size - w, 0, self.img_size - h))

    def get_bitmask(self, mask_decode_path):
        label = Image.open(mask_decode_path)
        label = torch.from_numpy(np.array(label)).long()
        return label == 255

    def get_bbox(self, mask):
        x = mask.any(1).nonzero()[0]
        y = mask.any(0).nonzero()[0]
        return [x[0], y[0], x[-1] + 1, y[-1] + 1]

    def get_image_clip(self, image_path):
        img = cv2.imread(os.path.join(self.base_image_dir, image_path))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

    def get_image_sam(self, image_path):
        img = cv2.imread(os.path.join(self.base_image_dir, image_path))
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_sam = self.transform.apply_image(image)
        resize = image_sam.shape[:2]
        image_sam = self.preprocess(torch.from_numpy(image_sam).permute(2, 0, 1).contiguous())
        return image_sam, resize

    def get_cropped_image_clip(self, image_path, refer_image_mask_path):
        image = Image.open(os.path.join(self.base_image_dir, image_path))
        image = np.array(image.convert("RGB"))
        mask4cropped = cv2.imread(os.path.join(self.base_image_dir, refer_image_mask_path), cv2.IMREAD_GRAYSCALE)
        mask4cropped = mask4cropped.astype(np.uint8)
        image_masked = cv2.bitwise_and(image, image, mask=mask4cropped)
        x0, y0, x1, y1 = self.get_bbox(mask4cropped)
        max_width = max(x1 - x0, y1 - y0)
        image_masked_cropped = image_masked[x0:x1, y0:y1]
        image_masked_cropped_padded = np.zeros((max_width, max_width, image_masked.shape[-1]), dtype=image_masked.dtype)
        image_masked_cropped_padded[: image_masked_cropped.shape[0], : image_masked_cropped.shape[1]] = image_masked_cropped
        inputs_cropped_image = self.clip_image_processor.preprocess(
            Image.fromarray(image_masked_cropped_padded), return_tensors="pt"
        )["pixel_values"][0]
        label4box = Image.open(os.path.join(self.base_image_dir, refer_image_mask_path))
        label4box = torch.from_numpy(np.array(label4box)).long()
        bbox_coords_sam = torch.tensor(get_bounding_box_from_mask(label4box))
        return inputs_cropped_image, bbox_coords_sam

    def __getitem__(self, idx):
        if self.random_sample:
            idx = random.randint(0, len(self.list_data_dict) - 1)
        source = self.list_data_dict[idx]
        sources = [source]
        if len(sources) != 1:
            raise ValueError("Expected a single conversation item")

        clean_questions_human = []
        clean_answers_gpt = []
        cropped_image_clip_list = []
        input_bbox_coord_list = []
        mask_decode_gt_list = []
        ref_decode_gt_list = []
        input_image_id_in_one_conv = 0
        input_round_id_and_input_image_id_list = []
        image_clip_path = ""

        sources_p = copy.deepcopy([item["conversations"] for item in sources])
        if sources[0].get("task") != "segmentation":
            raise ValueError("Only segmentation conversations are supported")
        if len(sources_p[0]) % 2 != 0:
            raise ValueError("Conversation must contain paired user and assistant turns")

        for turn in sources_p[0]:
            src = turn["from"]
            val = turn["value"]
            if src == "human":
                for item in find_brackets(val):
                    if "IMAGE256" in item:
                        val = val.replace(item, "<image>\n")
                        image_clip_path = item.split(":", 1)[1].rstrip("]")
                        image_clip = self.get_image_clip(image_clip_path)
                        image_sam, resize = self.get_image_sam(image_clip_path)
                    if "MASK-ENCODE" in item:
                        val = val.replace(item, "[CROPPEDIMAGE]")
                        cropped_image_clip, _ = self.get_cropped_image_clip(image_clip_path, item.split(":", 1)[1].rstrip("]"))
                        cropped_image_clip_list.append(cropped_image_clip)
                    if "BOX-ENCODE" in item:
                        val = val.replace(item, "[BBOXINPUT]")
                        _, input_bbox_coord = self.get_cropped_image_clip(image_clip_path, item.split(":", 1)[1].rstrip("]"))
                        input_bbox_coord_list.append(input_bbox_coord)
                clean_questions_human.append(val)
                if "ind" in turn:
                    input_round_id_and_input_image_id_list.append((turn["ind"], input_image_id_in_one_conv))
                    input_image_id_in_one_conv += 1
                else:
                    input_round_id_and_input_image_id_list.append((-1, -1))
            elif src == "gpt":
                matches = find_brackets(val)
                if len(matches) == 2:
                    for item in matches:
                        if "MASK-DECODE" in item:
                            val = val.replace(item, "[SEG]")
                            mask_decode_gt_list.append(self.get_bitmask(os.path.join(self.base_image_dir, item.split(":", 1)[1].rstrip("]"))))
                        if "REF-DECODE" in item:
                            val = val.replace(item, "")
                            ref_decode_gt_list.append(self.get_bitmask(os.path.join(self.base_image_dir, item.split(":", 1)[1].rstrip("]"))))
                elif len(matches) == 1:
                    if not matches[0].startswith("[MASK-DECODE:"):
                        raise ValueError("Expected a MASK-DECODE target")
                    val = val.replace(matches[0], "[SEG]")
                    mask_decode_gt_list.append(self.get_bitmask(os.path.join(self.base_image_dir, matches[0].split(":", 1)[1].rstrip("]"))))
                    ref_decode_gt_list.append(torch.zeros(mask_decode_gt_list[0].shape, dtype=torch.bool))
                else:
                    raise ValueError("Unexpected target token count")
                clean_answers_gpt.append(val)

        conversations = []
        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        for question, answer in zip(clean_questions_human, clean_answers_gpt):
            conv.append_message(conv.roles[0], question)
            conv.append_message(conv.roles[1], answer)
        conversations.append(conv.get_prompt())

        masks_decode_gt = torch.stack(mask_decode_gt_list, dim=0)
        refs_decode_gt = torch.stack(ref_decode_gt_list, dim=0)
        label = mask_decode_gt_list[0]
        sampled_classes = [255] * len(mask_decode_gt_list)
        cropped_image_clip = torch.stack(cropped_image_clip_list, dim=0)
        input_bbox_coord = torch.stack(input_bbox_coord_list, dim=0)

        return (
            image_clip_path,
            image_sam,
            image_clip,
            conversations,
            masks_decode_gt,
            refs_decode_gt,
            label,
            resize,
            clean_questions_human,
            sampled_classes,
            cropped_image_clip,
            input_bbox_coord,
            input_round_id_and_input_image_id_list,
        )


class ValDataset_Multi_Turn_Med_Seg(MultiTurnMedSegDataset):
    def __init__(self, test_file_path, base_image_dir, tokenizer, vision_tower, image_size=1024):
        super().__init__(
            base_image_dir=base_image_dir,
            tokenizer=tokenizer,
            vision_tower=vision_tower,
            data_json=test_file_path,
            samples_per_epoch=None,
            image_size=image_size,
            random_sample=False,
        )

    def __getitem__(self, idx):
        return (*super().__getitem__(idx), True)
