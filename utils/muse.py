import json  
import os  
import cv2  
import numpy as np  
import torch  
import torch.nn.functional as F  
from transformers import CLIPImageProcessor  
import transformers  
from pycocotools import mask as mask_utils  
from model.segment_anything.utils.transforms import ResizeLongestSide  
from model.llava import conversation as conversation_lib  
from .utils import (  
    ANSWER_LIST,  
    DEFAULT_IM_END_TOKEN,  
    DEFAULT_IM_START_TOKEN,  
    DEFAULT_IMAGE_PATCH_TOKEN,  
    DEFAULT_IMAGE_TOKEN,  
    LONG_QUESTION_LIST,  
    SHORT_QUESTION_LIST,  
)  
  
class CustomSegDataset(torch.utils.data.Dataset):  

    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)  
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)  
    img_size = 1024  
    ignore_label = 255  
  
    def __init__(  
        self,  
        base_image_dir,  
        tokenizer,  
        vision_tower,  
        json_file_path,  
        samples_per_epoch=500 * 8 * 2 * 10,  
        precision: str = "fp32",  
        image_size: int = 1024,  
        num_classes_per_sample: int = 3,  
        exclude_val=False,  
        seg_token_num=1,  
        pad_train_clip_images=False,  
        masks_process_with_clip=False,  
        preprocessor_config='',  
        inference=False, 
    ):  
        self.inference = inference         
        self.pad_train_clip_images = pad_train_clip_images  
        self.masks_process_with_clip = masks_process_with_clip  
        self.base_image_dir = base_image_dir  
        self.image_size = image_size  
        self.tokenizer = tokenizer  
        self.precision = precision  
        self.samples_per_epoch = samples_per_epoch  
        self.seg_token_num = seg_token_num  
          
        self.transform = ResizeLongestSide(image_size)  
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower) if preprocessor_config == '' else CLIPImageProcessor.from_pretrained(preprocessor_config)  
        self.transform_clip = ResizeLongestSide(self.clip_image_processor.size['shortest_edge'])  
        self.long_question_list = LONG_QUESTION_LIST  
          
       
        with open(json_file_path, 'r') as f:  
            self.data = json.load(f)  
          
        print(f"Loaded {len(self.data)} custom segmentation samples")  
  
    def __len__(self):  
    
        if self.samples_per_epoch == 0:
            return len(self.data)
        return self.samples_per_epoch  
  
    def preprocess(self, x: torch.Tensor, decoder_image_size) -> torch.Tensor:  
        """Normalize pixel values and pad to a square input."""  
        x = (x - self.pixel_mean) / self.pixel_std  
        h, w = x.shape[-2:]  
        padh = decoder_image_size - h  
        padw = decoder_image_size - w  
        x = F.pad(x, (0, padw, 0, padh))  
        return x  
  
    def __getitem__(self, idx):  
       
        if not self.inference:  
            idx = np.random.randint(0, len(self.data))  
        
        image_info = self.data[idx] 
          
      
        image_path = os.path.join(self.base_image_dir, f"{image_info['id']}.jpg")  
          
       
        img = cv2.imread(image_path)  
        if img is None:  
            print(f"Warning: Could not read image {image_path}")  
            if len(self.data) > 1:  
                return self[(idx + 1) % len(self.data)]  
            else:  
                raise FileNotFoundError(f"Cannot load any images from {self.base_image_dir}")  
          
        images = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
        ori_size = images.shape[:2]  
          
       
        if self.pad_train_clip_images:  
            image_clip = self.transform_clip.apply_image(images)  
            clip_resize = image_clip.shape[:2]  
            image_clip = self.preprocess(  
                torch.from_numpy(image_clip).permute(2, 0, 1).contiguous(),  
                self.clip_image_processor.size['shortest_edge']  
            )  
        else:  
            image_clip = self.clip_image_processor.preprocess(images, return_tensors="pt")["pixel_values"][0]  
            clip_resize = image_clip.shape[-2:]  
  
        images = self.transform.apply_image(images)  
        resize = images.shape[:2]  
          
       
        segs = image_info['ann_list']  
        masks = []  
          
        if len(segs) == 0:  
            print(f"Warning: No annotations for {image_path}")  
            if len(self.data) > 1:  
                return self[(idx + 1) % len(self.data)]  
            else:  
                raise ValueError(f"No valid annotations in dataset")
        valid_masks = []
        for ann in segs:
            points = ann['segmentation']
            if isinstance(points[0], list):
                points = points[0]

            
            if len(points) < 6: 
                print(f"Skipping invalid polygon (<3 points): {points}")
                continue

            xs = points[0::2]
            ys = points[1::2]
          
            if (max(xs) - min(xs) < 1) and (max(ys) - min(ys) < 1):
                print(f"Skipping degenerate polygon (same point repeated): {points}")
                continue

            
            try:

                rle = mask_utils.frPyObjects([points], image_info['height'], image_info['width'])
                m = mask_utils.decode(rle)
            except Exception as e:
                print(f"⚠️ Error decoding mask for {image_info['id']}: {e}")
                continue

           
            if len(m.shape) > 2:
                m = np.sum(m, axis=2)
            m = m.astype(np.uint8)

            
            if np.sum(m > 0) == 0:
                print(f"⚠️ Skipping empty mask for image {image_info['id']}")
                continue

            valid_masks.append(m)

        
        if len(valid_masks) == 0:
            print(f"⚠️ No valid masks in {image_info['id']}, skipping this sample.")
            if len(self.data) > 1:
                return self[(idx + 1) % len(self.data)]
            else:
                raise ValueError(f"No valid masks in dataset for {image_info['id']}")
        masks = valid_masks
        
        
       
            
          
       
        questions = image_info['questions']  
        answers = image_info['answers']  
       
        reasoning_types = image_info.get('reasoning_types', ['unknown'])
        category = reasoning_types[0] if isinstance(reasoning_types, list) and len(reasoning_types) > 0 else (reasoning_types if isinstance(reasoning_types, str) else 'unknown')
          
       
        conversations = []  
        conv = conversation_lib.default_conversation.copy()  
        seg_token = "[SEG]" if self.seg_token_num == 1 else ' '.join([f"[SEG{i}]" for i in range(self.seg_token_num)])  
          
       
        questions = image_info['questions']  
        answers = image_info['answers']  
        
        
        question = questions[0]  
        answer = answers[0]  
        
     
        conversations = []  
        conv = conversation_lib.default_conversation.copy()  
        seg_token = "[SEG]" if self.seg_token_num == 1 else ' '.join([f"[SEG{i}]" for i in range(self.seg_token_num)])  
        
       
        conv.messages = []  
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)  
        conv.append_message(conv.roles[1], seg_token)  
        conversations.append(conv.get_prompt())  
          
   
        images = self.preprocess(  
            torch.from_numpy(images).permute(2, 0, 1).contiguous(),  
            self.img_size  
        )  
          
     
        masks = np.stack(masks, axis=0)  
       
        masks = torch.from_numpy(masks)  
         

        label = torch.ones(masks.shape[1], masks.shape[2]) * self.ignore_label  
        
          
        if self.masks_process_with_clip:  
            mask_shape = image_clip.shape[-1]  
            masks = transform_mask(masks, mask_shape)  

       
        if self.inference:  
            
            return (  
                image_path, images, image_clip, conversations,  
                masks, label, resize, clip_resize,  
                questions, questions,  
                False,                   
                True,                      
                category         
            )  
        else:  
           
            return (  
                image_path, images, image_clip, conversations,  
                masks, label, resize, clip_resize,  
                questions, questions,  
                False,                    
                category          
            )
       
  
def transform_mask(masks, size):  
    """与 MultiReasonSegDataset 相同的掩码变换函数"""  
    height, width = masks.shape[-2:]  
    short, long = (width, height) if width <= height else (height, width)  
    requested_new_short = size  
    new_short, new_long = requested_new_short, int(requested_new_short * long / short)  
    new_shape = (new_long, new_short) if width <= height else (new_short, new_long)  
    masks = F.interpolate(masks[None].float(), size=new_shape, mode="nearest")[0].bool()  
  
    orig_height, orig_width = new_shape  
    crop_height, crop_width = size, size  
    top = (orig_height - crop_height) // 2  
    bottom = top + crop_height  
    left = (orig_width - crop_width) // 2  
    right = left + crop_width  
      
    assert top >= 0 and bottom <= orig_height and left >= 0 and right <= orig_width  
    masks = masks[..., top:bottom, left:right]  
      
    return masks