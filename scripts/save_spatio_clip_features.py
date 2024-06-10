import os
import sys
import math
import torch
import pickle
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from glob import glob
# from decord import VideoReader, cpu
from torch import nn

from transformers import CLIPVisionModel, CLIPImageProcessor



def load_image(img_path):
    return Image.open(img_path).convert("RGB")
    
def get_spatial_features(features):
    # For images, we just have spatial dimensions (height x width x channels).
    # We'll simply average over the spatial dimensions to get a representative feature.
    spatial_tokens = np.mean(features, axis=0)

    return spatial_tokens

def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--image_dir_path", required=True, help="Path to read the image from.")
    parser.add_argument("--clip_feat_path", required=True, help="The output dir to save the features in.")
    parser.add_argument("--infer_batch", required=False, type=int, default=32,
                        help="Number of frames/images to perform batch inference.")

    args = parser.parse_args()

    return args




def main():
    args = parse_args()
    image_dir_path = args.image_dir_path
    clip_feat_path = args.clip_feat_path
    infer_batch = args.infer_batch
    os.makedirs(clip_feat_path, exist_ok=True)

    # Initialize the CLIP model
    image_processor = CLIPImageProcessor.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16)
    vision_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14', torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True).cuda()
    
    vision_tower.eval()
    
    all_images = glob(image_dir_path + "/*.jpg")


    image_clip_features = {}
    counter = 0
    for image_path in tqdm(all_images):
        image_id = image_path.split("/")[-1].strip(".jpg")
        if os.path.exists(f"{clip_feat_path}/{image_id}.pkl"):  # Check if the file is already processed
            continue
        try:
            image = load_image(image_path)
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
            image_tensor = image_tensor.half()

            n_chunk = len(image_tensor)
            image_features = torch.FloatTensor(n_chunk, 256, 1024).fill_(0)
            n_iter = int(math.ceil(n_chunk / float(infer_batch)))
            for i in range(n_iter):
                min_ind = i * infer_batch
                max_ind = (i + 1) * infer_batch
                image_batch = image_tensor[min_ind:max_ind].cuda()

                image_forward_outs = vision_tower(image_batch, output_hidden_states=True)

                select_hidden_state_layer = -2
                select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                batch_features = select_hidden_state[:, 1:]
                image_features[min_ind:max_ind] = batch_features.detach().cpu()

            image_clip_features[image_id] = get_spatial_features(image_features.numpy().astype("float16"))
            counter += 1

        except Exception as e:
            print(f"Can't process {image_path}")

        if counter % 512 == 0:  # Save after every 512 images, update this number as per your requirements
            for key in image_clip_features.keys():
                features = image_clip_features[key]
                with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
                    pickle.dump(features, f)
            image_clip_features = {}
    for key in image_clip_features.keys():
        features = image_clip_features[key]
        with open(f"{clip_feat_path}/{key}.pkl", 'wb') as f:
            pickle.dump(features, f)


if __name__ == "__main__":
    main()
