import json
import os
import random
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy


class DataCollatorWithImage:
    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 padding: Union[bool, str, PaddingStrategy] = True,
                 max_length: Optional[int] = None,
                 pad_to_multiple_of: Optional[int] = None,
                 return_tensors: str = "pt"):
        
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        images = [item['image'] for item in features]
        input_ids = [item['input_ids'][0] for item in features]
        attention_mask = [item['attention_mask'][0] for item in features]
        origin_features = [
            {'input_ids': ids, 'attention_mask': mask} for ids,mask in zip(input_ids, attention_mask)
        ]
        
        batch = self.tokenizer.pad(
            origin_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        batch['images'] = images

        if 'text' in features[0]:
            texts = [item['text'] for item in features]
            batch['texts'] = texts
        
        if 'caption' in features[0]:
            captions = [item['caption'] for item in features]
            batch['captions'] = captions
            
        return batch


class Object365Dataset(Dataset):

    def __init__(self, image_path, tokenizer):
        self.annotation = json.load(open('data/object365.json'))
        self.image_path = image_path
        # `VG_100K_2/4.jpg` is here only as a placeholder
        self.prompt_template = '<img>VG_100K_2/4.jpg</img><ref>This</ref><box>({:},{:}),({:},{:})</box> is'
        self.cached_data_dict = {}
        self.tokenizer = tokenizer

    def bbox_format(self, bbox):
        return [int(bbox[0] * 999), int(bbox[1] * 999), \
                int(bbox[2] * 999), int(bbox[3] * 999)]

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        if index in self.cached_data_dict:
            return self.cached_data_dict[index]

        while(True):
            try:
                annotation = self.annotation[index]

                image_name = annotation['image_id']
                image_path = os.path.join(self.image_path, image_name)
                image = Image.open(image_path).convert('RGB')

                norm_bboxes = self.bbox_format(annotation['bbox'])
                prompts = self.prompt_template.format(norm_bboxes[0], norm_bboxes[1], \
                                                        norm_bboxes[2], norm_bboxes[3])

                prompts_tokenized = self.tokenizer([prompts], return_tensors='pt', padding='longest')
                attention_mask = prompts_tokenized.attention_mask
                input_ids = prompts_tokenized.input_ids

                ret = dict(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image=image)
                
                self.cached_data_dict[index] = ret
                return ret

            except Exception as e:
                index = random.randint(0, len(self.annotation) - 1)