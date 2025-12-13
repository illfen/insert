from typing import List, Tuple

import timm
import torch
import torchvision.transforms as T
from PIL import Image


class DinoSimilaritySelector:
    def __init__(self, model_name: str = "vit_base_patch16_224_dino"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(model_name, pretrained=True).to(self.device).eval()
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        img = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model.forward_features(img)
            features = outputs[:, 0]  # CLS token
        return features.squeeze(0)

    def most_similar(self, target_img: Image.Image, candidate_imgs: List[Image.Image]) -> Tuple[int, float]:
        target_feat = self.extract_features(target_img)
        candidate_feats = torch.stack([self.extract_features(img) for img in candidate_imgs])
        similarities = torch.nn.functional.cosine_similarity(candidate_feats, target_feat.unsqueeze(0), dim=1)
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        return best_idx, best_score


selector = DinoSimilaritySelector()

import argparse
import os

parser = argparse.ArgumentParser(description="DINO Similarity Selection")
parser.add_argument("--target_image", type=str, required=True, help="Path to the target image")
parser.add_argument("--candidates_dir", type=str, required=True, help="Directory of candidate images")
args = parser.parse_args()

candidates = []
img_path_list = []
for file in os.listdir(args.candidates_dir):
    if file.endswith(".jpg") or file.endswith(".png"):
        img_path = os.path.join(args.candidates_dir, file)
        img_path_list.append(img_path)
        candidates.append(Image.open(img_path))

target_img = Image.open(args.target_image)

# 查找最相似的候选图
idx, score = selector.most_similar(target_img, candidates)

print(f"Most similar image: {img_path_list[idx]} with similarity score: {score:.4f}")

candidates[idx].save(img_path_list[idx])