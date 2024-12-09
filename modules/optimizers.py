import torch

# Function to build the optimizer
def build_optimizer(args, model):
    # Check if contrastive loss is enabled and text encoder is not fixed
    if (args.contras_loss_w > 0) and (args.fix_text_encoder is not True):
        # Get parameter IDs for visual extractor and text encoder (for knowledge distillation)
        ve_params = list(map(id, model.visual_extractor.parameters()))
        te_params = list(map(id, model.text_encoder_kd.parameters()))
        
        # Filter out visual extractor and text encoder parameters from the rest of the model parameters
        ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
        ed_params = filter(lambda x: id(x) not in te_params, ed_params)

        # Create the optimizer with different learning rates for different parts of the model
        optimizer = getattr(torch.optim, args.optim)(
            [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},  # Visual extractor
             {'params': model.text_encoder_kd.parameters(), 'lr': args.lr_te},  # Text encoder (knowledge distillation)
             {'params': ed_params, 'lr': args.lr_ed}],  # Rest of the model
            weight_decay=args.weight_decay,  # Apply weight decay
            amsgrad=args.amsgrad  # Use AMSGrad if specified
        )
    else:
        # Only apply learning rate adjustments for visual extractor and other model parts
        ve_params = list(map(id, model.visual_extractor.parameters()))
        ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())

        optimizer = getattr(torch.optim, args.optim)(
            [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},  # Visual extractor
             {'params': ed_params, 'lr': args.lr_ed}],  # Rest of the model
            weight_decay=args.weight_decay,  # Apply weight decay
            amsgrad=args.amsgrad  # Use AMSGrad if specified
        )
    
    return optimizer

# Function to build the learning rate scheduler
def build_lr_scheduler(args, optimizer):
    # Create the learning rate scheduler based on the specified type and parameters
    lr_scheduler = getattr(torch.optim.lr_scheduler, args.lr_scheduler)(
        optimizer,  # Optimizer to apply the scheduler to
        args.step_size,  # Step size for the scheduler
        args.gamma  # Multiplicative factor for learning rate decay
    )
    return lr_scheduler
