import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import XGDataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from modules.xgren import XGRenModel
from pycocoevalcap.meteor import Meteor
import wandb
import os

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/images/', help='Path to the directory containing images.')
    parser.add_argument('--ann_path', type=str, default='data/annotation.json', help='Path to the annotation file.')

    # Data loader settings
    parser.add_argument('--use_topic', type=bool, default=True, help='Whether to use topics.')
    parser.add_argument('--topic_type', nargs='+', default=['iu', 'knee', 'axr', 'shoulder', 'hip', 'wrist'], 
                        help='List of body parts to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='Max sequence length for reports.')
    parser.add_argument('--max_seq_length_bert', type=int, default=80, help='Max sequence length for contrastive learning.')
    parser.add_argument('--threshold', type=int, default=3, help='Cutoff frequency for words.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for data loader.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')

    # Model settings
    parser.add_argument('--visual_extractor', type=str, default='resnet101', choices=['resnet101', 'vit'], 
                        help='Visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='Whether to use pretrained weights.')
    parser.add_argument('--clip_pretrained_path', type=str, default='models/medclip-pretrained/pytorch_model.bin', 
                        help='Path to the pretrained model for MedClip.')
    parser.add_argument('--fix_text_encoder', type=bool, default=True, help='Whether to fix text encoder during training.')

    # Transformer settings
    parser.add_argument('--d_model', type=int, default=512, help='Dimension of the Transformer.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads in the Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of Transformer layers.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for Transformer.')

    # Training settings
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--save_dir', type=str, default='results/XRGen', help='Directory to save the model checkpoints.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed.')

    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    wandb.init(project='X-RGen', name=args.save_dir.split('/')[1])

    # Fix random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Tokenizer
    tokenizer = Tokenizer(args)

    # Data loaders
    train_loader = XGDataLoader(args, tokenizer, split='train', shuffle=True)
    val_loader = XGDataLoader(args, tokenizer, split='val', shuffle=False)
    test_loader = XGDataLoader(args, tokenizer, split='test', shuffle=False)

    # Build model
    model = XGRenModel(args, tokenizer).to('cuda' if torch.cuda.is_available() else 'cpu')

    # Loss function, metrics, optimizer, scheduler
    criterion = compute_loss
    metrics = compute_scores
    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler()

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            images, labels = batch['images'].to('cuda'), batch['labels'].to('cuda')

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

        scheduler.step()

        # Validation (optional)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch['images'].to('cuda'), batch['labels'].to('cuda')
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {running_loss/len(train_loader)}, Val Loss: {val_loss/len(val_loader)}")

    wandb.finish()

if __name__ == '__main__':
    main()
