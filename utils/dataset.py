import torch

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from model.llava.mm_utils import tokenizer_image_token
from utils.multi_turn_med_seg import MultiTurnMedSegDataset, ValDataset_Multi_Turn_Med_Seg
from utils.utils import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN


class HybridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_image_dir,
        tokenizer,
        vision_tower,
        samples_per_epoch,
        precision="fp32",
        image_size=1024,
        train_json=None,
    ):
        if train_json is None:
            raise ValueError("train_json is required")
        self.dataset = MultiTurnMedSegDataset(
            base_image_dir=base_image_dir,
            tokenizer=tokenizer,
            vision_tower=vision_tower,
            data_json=train_json,
            samples_per_epoch=samples_per_epoch,
            precision=precision,
            image_size=image_size,
            random_sample=True,
        )
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return (*self.dataset[idx], False)


def collate_fn(batch, tokenizer=None, conv_type="llava_v1", use_mm_start_end=True, local_rank=-1):
    image_path_list = []
    images_list = []
    images_clip_list = []
    conversation_list = []
    masks_list = []
    ref_masks_list = []
    label_list = []
    resize_list = []
    questions_list = []
    sampled_classes_list = []
    offset_list = [0]
    cnt = 0
    inferences = []
    images_cropped_list = []
    bboxes_list = []
    input_round_id_and_input_image_id_list = []

    for (
        image_path,
        images,
        images_clip,
        conversations,
        masks,
        ref_masks,
        label,
        resize,
        questions,
        sampled_classes,
        cropped_images,
        bboxes,
        round_image_ids,
        inference,
    ) in batch:
        image_path_list.append(image_path)
        images_list.append(images)
        images_clip_list.append(images_clip)
        conversation_list.extend(conversations)
        label_list.append(label)
        masks_list.append(masks.float())
        ref_masks_list.append(ref_masks.float())
        resize_list.append(resize)
        questions_list.append(questions)
        sampled_classes_list.append(sampled_classes)
        images_cropped_list.append(cropped_images)
        bboxes_list.append(bboxes)
        input_round_id_and_input_image_id_list.append(round_image_ids)
        cnt += len(conversations)
        offset_list.append(cnt)
        inferences.append(inference)

    if use_mm_start_end:
        for i in range(len(conversation_list)):
            replace_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
            conversation_list[i] = conversation_list[i].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversation_list]
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id)
    conv = conversation_lib.default_conversation.copy()
    targets = input_ids.clone()
    sep = conv.sep + conv.roles[1] + ": " if conv_type == "llava_v1" else "[/INST] "

    for conversation, target in zip(conversation_list, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for rou in rounds:
            if rou == "":
                break
            parts = rou.split(sep)
            if len(parts) != 2:
                raise ValueError("Conversation template mismatch")
            parts[0] += sep
            if DEFAULT_IMAGE_TOKEN in conversation:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length and cur_len != total_len:
            target[:] = IGNORE_INDEX
            print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. Ignored labels.")

    if not inferences[0]:
        truncate_len = tokenizer.model_max_length - 575
        if input_ids.shape[1] > truncate_len:
            raise ValueError("Input sequence is too long")

    return {
        "image_paths": image_path_list,
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),
        "input_ids": input_ids,
        "labels": targets,
        "attention_masks": attention_masks,
        "masks_list": masks_list,
        "ref_masks_list": ref_masks_list,
        "label_list": label_list,
        "resize_list": resize_list,
        "offset": torch.LongTensor(offset_list),
        "questions_list": questions_list,
        "sampled_classes_list": sampled_classes_list,
        "cropped_images_encode": images_cropped_list,
        "encode_bboxes_list": bboxes_list,
        "input_round_id_and_input_image_id_List": input_round_id_and_input_image_id_list,
        "inference": inferences[0],
        "conversation_list": conversation_list,
    }
