import os
import safetensors
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class ResBlock(nn.Module):
    """Linear block with residuals"""

    def __init__(self, ch):
        super().__init__()
        self.join = nn.ReLU()
        self.long = nn.Sequential(
            nn.Linear(ch, ch),
            nn.LeakyReLU(0.1),
            nn.Linear(ch, ch),
            nn.LeakyReLU(0.1),
            nn.Linear(ch, ch),
        )

    def forward(self, x):
        return self.join(self.long(x) + x)


class PredictorModel(nn.Module):
    """Main predictor class"""

    def __init__(self, features=768, outputs=1, hidden=1024):
        super().__init__()
        self.features = features
        self.outputs = outputs
        self.hidden = hidden
        self.up = nn.Sequential(
            nn.Linear(self.features, self.hidden),
            ResBlock(ch=self.hidden),
        )
        self.down = nn.Sequential(
            nn.Linear(self.hidden, 128),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, self.outputs),
        )
        self.out = nn.Softmax(dim=1) if self.outputs > 1 else nn.Tanh()

    def forward(self, x):
        y = self.up(x)
        z = self.down(y)
        if self.outputs > 1:
            return self.out(z)
        else:
            return (self.out(z) + 1.0) / 2.0


class CityAesthetics:
    def __init__(self, model_path, clip_model_path, device='cpu'):
        self.device = device
        self.model_path = model_path
        self.clip_model_path = clip_model_path
        self.initialize_model()

    def initialize_model(self):
        # Load CityAesthetics model weights
        statedict = safetensors.torch.load_file(self.model_path)
        self.model = PredictorModel()
        self.model.eval()
        self.model.load_state_dict(statedict)
        self.model.to(self.device)

        # Load CLIP model and processor
        self.clip_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.clip_model = CLIPVisionModelWithProjection.from_pretrained(
            self.clip_model_path,  # Use the local CLIP model path
            device_map=self.device,
        )

    def score(self, prompt, image):
        pil_image = None  # Initialize pil_image with None

        if isinstance(image, Image.Image):
            pil_image = image
        elif isinstance(image, str):
            if os.path.isfile(image):
                pil_image = Image.open(image)

        # Process image and text with CLIP
        # Handle the case where pil_image is still None (image loading failed)
        if pil_image is not None:
            inputs = self.clip_processor(text=[prompt], images=pil_image, return_tensors="pt", padding=True)
            inputs = inputs.to(self.device)
            text_embeds = self.clip_model.get_text_features(**inputs)
            image_embeds = self.clip_model.get_image_features(**inputs)

            # Normalize embeddings
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

            # Concatenate embeddings and get aesthetic score
            global_features = torch.cat([text_embeds, image_embeds], dim=-1)
            score = self.model(global_features)[0][0].item() * 10

            return score
        else:
            # Handle the case where image loading failed
            print(f"Failed to load image: {image}")
            return 0.0  # Or raise an exception, depending on your error handling strategy
