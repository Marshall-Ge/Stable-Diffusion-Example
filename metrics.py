from torchvision import transforms
from torch.nn.functional import adaptive_avg_pool2d
from scipy import linalg
import clip
from pytorch_fid.inception import InceptionV3
from cleanfid.inception_torchscript import InceptionV3W as CleanInceptionV3
import torch
from PIL import Image
import numpy as np


class Evaluator:
    def __init__(self, device='cuda'):
        self.device = device
        # Load CLIP model for CLIP Score
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=device)
        # Load InceptionV3 for FID and IS
        self.inception_model = InceptionV3([3]).to(device)
        self.inception_model.eval()
        # Standard image transform for inception
        self.inception_transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def calculate_clip_score(self, images, prompts):
        """Calculate CLIP Score between images and text prompts"""
        if isinstance(images, Image.Image):
            images = [images]
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Preprocess images
        processed_images = torch.stack([self.clip_preprocess(img).to(self.device) for img in images])
        
        # Encode images and text
        with torch.no_grad():
            image_features = self.clip_model.encode_image(processed_images)
            text_features = self.clip_model.encode_text(clip.tokenize(prompts).to(self.device))
        
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarity scores
        similarity = (100.0 * image_features @ text_features.T).diagonal()
        return similarity.mean().item()

    def get_inception_features(self, images):
        """Extract features from Inception model for FID and IS"""
        if isinstance(images, Image.Image):
            images = [images]
            
        features = []
        for img in images:
            img_tensor = self.inception_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                feature = self.inception_model(img_tensor)[0]
            features.append(feature.squeeze().cpu().numpy())
        return np.array(features)
    
    def calculate_fid(self, real_images, generated_images):
        """Calculate FID between real and generated images"""
        real_features = self.get_inception_features(real_images)
        gen_features = self.get_inception_features(generated_images)
        
        # Calculate mean and covariance
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)
        
        # Calculate FID
        ssdiff = np.sum((mu1 - mu2) ** 2)
        covmean = linalg.sqrtm(sigma1.dot(sigma2))
        
        # Check if covmean has complex values
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid
    
    def calculate_inception_score(self, images, splits=10):
        """Calculate Inception Score for generated images"""
        if isinstance(images, Image.Image):
            images = [images]
            
        # Get features for IS
        features = []
        for img in images:
            img_tensor = self.inception_transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = torch.nn.functional.softmax(self.inception_model(img_tensor)[0], dim=1)
            features.append(pred.squeeze().cpu().numpy())
        features = np.array(features)
        
        # Split into groups
        if len(features) < splits:
            splits = len(features)
        
        scores = []
        for i in range(splits):
            part = features[i * (len(features) // splits): (i + 1) * (len(features) // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, axis=0), 0)))
            kl = np.mean(np.sum(kl, axis=1))
            scores.append(np.exp(kl))
            
        return np.mean(scores), np.std(scores)