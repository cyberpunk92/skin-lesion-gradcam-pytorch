import os
import requests
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

# 2. Automatically download a sample dermoscopic image if not present
SAMPLE_IMG_PATH = "sample_lesion.jpg"
def download_sample_image():
    if not os.path.exists(SAMPLE_IMG_PATH):
        print("Downloading sample dermoscopic skin lesion image...")
        url = "https://upload.wikimedia.org/wikipedia/commons/6/6c/Melanoma.jpg"
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        if response.status_code == 200:
            with open(SAMPLE_IMG_PATH, 'wb') as f:
                f.write(response.content)
            print("Sample image downloaded successfully!")
        else:
            raise Exception("Failed to download sample image.")

# 3. Build Model Pipeline
def build_model():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 7)
    model.to(device)
    model.eval()
    return model

# 4. Image Preprocessing Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 5. Run Grad-CAM Generation
def generate_explainability_map(image_path, output_filename="sample_gradcam_result.png"):
    model = build_model()
    
    # Load and preprocess image
    rgb_img = Image.open(image_path).convert('RGB')
    rgb_img_resized = rgb_img.resize((224, 224))
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)

    # Target final convolutional feature layer of EfficientNet-B0
    target_layers = [model.features[-1]]

    # Compute Grad-CAM heatmaps
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]

    # Normalize image array for overlay blending
    rgb_np = np.array(rgb_img_resized, dtype=np.float32) / 255.0
    visualization = show_cam_on_image(rgb_np, grayscale_cam, use_rgb=True)

    # Export side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(rgb_img_resized)
    axes[0].set_title("Input Medical Image")
    axes[0].axis('off')

    axes[1].imshow(visualization)
    axes[1].set_title("Grad-CAM Focus Region")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_filename, dpi=300)
    print(f"Success! Explainability plot saved as '{output_filename}'.")

if __name__ == "__main__":
    download_sample_image()
    generate_explainability_map(SAMPLE_IMG_PATH)