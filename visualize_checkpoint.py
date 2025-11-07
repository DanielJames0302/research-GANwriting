#!/usr/bin/env python3
"""
Simple script to visualize generated images from a saved checkpoint.
Loads a model checkpoint, generates an image from text, and saves it to a PNG file.
"""

import argparse
import os
import cv2
import numpy as np
import torch
from network_tro import ConTranModel
from load_data import (
    IMG_HEIGHT, IMG_WIDTH, NUM_WRITERS, letter2index, index2letter, 
    tokens, num_tokens, OUTPUT_MAX_LEN, loadCaptchaData, loadData as load_data_func
)
from modules_tro import normalize


def create_dummy_reference_images(num_images=50):
    """Create dummy reference images (zeros) for generation when dataset is not available."""
    dummy_img = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    # Normalize like real images
    mean = 0.5
    std = 0.5
    dummy_img = (dummy_img - mean) / std
    # Repeat to create batch of reference images
    dummy_imgs = np.stack([dummy_img] * num_images, axis=0)
    return dummy_imgs


def load_reference_images_from_dataset(dataset_type='captcha', captcha_dir=None):
    """Load reference images from the dataset if available."""
    try:
        if dataset_type == 'captcha' and captcha_dir and os.path.exists(captcha_dir):
            _, data_test = loadCaptchaData(True, captcha_dir)
            # Get a sample batch
            sample = data_test[0]
            # sample[3] is train_img which contains the reference images
            ref_imgs = sample[3]  # Shape: (NUM_CHANNEL, IMG_HEIGHT, IMG_WIDTH)
            return ref_imgs
        elif dataset_type == 'iam':
            _, data_test = load_data_func(True)
            sample = data_test[0]
            ref_imgs = sample[3]  # Shape: (NUM_CHANNEL, IMG_HEIGHT, IMG_WIDTH)
            return ref_imgs
    except Exception as e:
        print(f"Warning: Could not load reference images from dataset: {e}")
        print("Using dummy reference images instead.")
    return None


def text_to_label(text, num_tokens):
    """Convert text string to label tensor format."""
    ll = [letter2index[i] for i in text if i in letter2index]
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    ll = [tokens['GO_TOKEN']] + ll + [tokens['END_TOKEN']]
    num = OUTPUT_MAX_LEN - len(ll)
    if num > 0:
        ll.extend([tokens['PAD_TOKEN']] * num)
    return np.array(ll)


def generate_image(model, reference_imgs, text, gpu):
    """Generate an image from text using the model."""
    # Prepare reference images: add batch dimension and move to GPU
    # reference_imgs shape: (NUM_CHANNEL, IMG_HEIGHT, IMG_WIDTH)
    # Need: (1, NUM_CHANNEL, IMG_HEIGHT, IMG_WIDTH)
    ref_imgs_tensor = torch.from_numpy(reference_imgs).unsqueeze(0).to(gpu)
    
    # Prepare text label
    label = text_to_label(text, num_tokens)
    label_tensor = torch.from_numpy(label).unsqueeze(0).to(gpu)  # (1, OUTPUT_MAX_LEN)
    
    model.eval()
    with torch.no_grad():
        # Encode reference images
        f_xs = model.gen.enc_image(ref_imgs_tensor)
        
        # Encode text
        f_xt, f_embed = model.gen.enc_text(label_tensor, f_xs.shape)
        
        # Mix content and style
        f_mix = model.gen.mix(f_xs, f_embed)
        
        # Decode to generate image
        xg = model.gen.decode(f_mix, f_xt)
        
        # Convert to numpy and process
        xg = xg.cpu().numpy().squeeze()  # Remove batch dimension
        
        # Normalize and invert (matching test scripts)
        xg = normalize(xg)
        xg = 255 - xg  # Invert to show white background, black text
        
        return xg


def main():
    parser = argparse.ArgumentParser(description='Visualize generated image from checkpoint')
    parser.add_argument('checkpoint', type=str, help='Path to model checkpoint file')
    parser.add_argument('--text', type=str, default='hello', 
                       help='Text to generate (default: hello)')
    parser.add_argument('--dataset', type=str, default='captcha', 
                       choices=['iam', 'captcha'], 
                       help='Dataset type: iam or captcha (default: captcha)')
    parser.add_argument('--captcha_dir', type=str, default='test/correct_test',
                       help='Directory containing captcha images (only used when --dataset=captcha)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output PNG file path (default: generated_<text>.png)')
    parser.add_argument('--use_dummy_ref', action='store_true',
                       help='Use dummy reference images instead of loading from dataset')
    
    args = parser.parse_args()
    
    # Set output filename
    if args.output is None:
        safe_text = ''.join(c if c.isalnum() else '_' for c in args.text)
        args.output = f'generated_{safe_text}.png'
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        return
    
    # Setup GPU
    gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU (may be slow)")
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = ConTranModel(NUM_WRITERS, 0, True).to(gpu)
    model.load_state_dict(torch.load(args.checkpoint, map_location=gpu))
    print("Model loaded successfully.")
    
    # Load or create reference images
    if args.use_dummy_ref:
        print("Using dummy reference images...")
        reference_imgs = create_dummy_reference_images()
    else:
        print(f"Loading reference images from {args.dataset} dataset...")
        reference_imgs = load_reference_images_from_dataset(
            args.dataset, 
            args.captcha_dir if args.dataset == 'captcha' else None
        )
        if reference_imgs is None:
            print("Using dummy reference images instead...")
            reference_imgs = create_dummy_reference_images()
    
    # Generate image
    print(f"Generating image for text: '{args.text}'...")
    try:
        generated_img = generate_image(model, reference_imgs, args.text, gpu)
        
        # Save image
        print(f"Saving image to {args.output}...")
        cv2.imwrite(args.output, generated_img)
        print(f"Successfully saved image to {args.output}")
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

