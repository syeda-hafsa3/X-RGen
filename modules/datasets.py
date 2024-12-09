import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torchvision.transforms as transforms


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.max_seq_length_bert = args.max_seq_length_bert
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        
        # Load annotations
        with open(self.ann_path, 'r') as f:
            self.ann = json.load(f)

        self.tokenizer_bert = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

        # Preprocess annotations
        self.examples = self.ann[self.split]
        for example in self.examples:
            example['ids_bert'] = self.tokenizer_bert(example['report'])['input_ids'][:self.max_seq_length_bert]
            example['mask_bert'] = [1] * len(example['ids_bert'])
            example['ids'] = self.tokenizer(example['report'])['input_ids'][:self.max_seq_length]
            example['mask'] = [1] * len(example['ids'])

    def __len__(self):
        return len(self.examples)


class MultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_paths = example['image_path']

        # Load images
        image_1 = Image.open(os.path.join(self.image_dir, image_paths[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_paths[1])).convert('RGB')

        # Apply transformations if provided
        if self.transform:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)

        # Stack images along the first dimension
        images = torch.stack((image_1, image_2), 0)

        # Get report and tokenization
        report_ids = example['ids']
        report_masks = example['mask']
        report_ids_bert = example['ids_bert']
        report_masks_bert = example['mask_bert']
        
        seq_length = len(report_ids)
        seq_length_bert = len(report_ids_bert)

        # If a tag exists in the example, include it in the returned sample
        if 'tag' in example:
            image_tag = example['tag']
            sample = (image_id, image_tag, images, report_ids, report_masks, report_ids_bert, report_masks_bert, seq_length, seq_length_bert)
        else:
            sample = (image_id, images, report_ids, report_masks, report_ids_bert, report_masks_bert, seq_length, seq_length_bert)

        return sample


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Pretrained normalization
])

# Function to get DataLoader for custom dataset
def get_dataloader(image_dir, batch_size=32, shuffle=True):
    dataset = CustomDataset(image_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
