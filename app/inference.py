import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

def remove_hair(image):
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Gunakan kernel ukuran 9x9 untuk keseimbangan penghapusan rambut
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    # Naikkan threshold ke 20 agar lebih ketat dalam mendeteksi warna gelap
    ret, thresh2 = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    # Dilation secukupnya
    thresh2 = cv2.dilate(thresh2, np.ones((3,3), np.uint8), iterations=1)
    # Gunakan algoritma Inpaint Telea kembali agar proses jauh lebih cepat
    dst = cv2.inpaint(image, thresh2, 3, cv2.INPAINT_TELEA)
    return dst

def adaptive_clahe(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = [gray[0:20, 0:20], gray[0:20, -20:], gray[-20:, 0:20], gray[-20:, -20:]]
    mean_corner_intensity = np.mean([np.mean(c) for c in corners])
    
    if mean_corner_intensity > 150:
        clip_limit = 2.0
    elif mean_corner_intensity < 100:
        clip_limit = 4.0
    else:
        clip_limit = 3.0
        
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

def preprocess_image_pipeline(image_bytes, use_preprocessing=True):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Keep original for display (resize to 256 for consistency)
    orig_resized = cv2.resize(img, (256, 256))
    
    # Conditional Preprocessing
    if use_preprocessing:
        img_clean = remove_hair(orig_resized)
        img_final = adaptive_clahe(img_clean)
    else:
        img_final = orig_resized
    
    # To Tensor
    img_tensor = img_final.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)
    
    return orig_resized, img_tensor, img_final

class SegmentationGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor):
        self.model.eval()
        self.model.zero_grad()
        
        output = self.model(input_tensor)
        score = output.sum()
        score.backward()
        
        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, output

def tensor_to_base64(img_np):
    img_pil = Image.fromarray(img_np.astype('uint8'))
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def process_inference(model, input_tensor, orig_image, preprocessed_image):
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
        
    # Set up GradCAM depending on model type
    # For ResUNet, usually target the last decoder or bridge. Let's use decoder1's final conv block
    # For CBAMResUNet, we target decoder1's cbam or conv block
    if hasattr(model, 'decoder1'):
        if hasattr(model.decoder1, 'cbam'):
            target = model.decoder1.cbam # CBAM block
        else:
            target = model.decoder1.conv_block # Normal ResUNet
    else:
        target = list(model.children())[-2] # Fallback
        
    grad_cam = SegmentationGradCAM(model, target)
    cam, output = grad_cam.generate(input_tensor)
    
    # Prediction mask
    pred = output.squeeze().detach().cpu().numpy()
    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    
    # Post-Processing: Morfologi untuk menghaluskan tepi yang bergerigi (Smooth edges)
    kernel = np.ones((7, 7), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel) # Menutup lubang-lubang kecil di dalam
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)  # Menghilangkan noise bintik di luar dan merapikan tepi
    
    
    # Blend Mask
    mask_colored = np.zeros_like(orig_image)
    mask_colored[pred_mask == 255] = [255, 255, 255] # White mask
    
    # Blend Grad-CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam_blended = cv2.addWeighted(orig_image, 0.5, heatmap, 0.5, 0)
    
    return tensor_to_base64(orig_image), tensor_to_base64(preprocessed_image), tensor_to_base64(mask_colored), tensor_to_base64(cam_blended)
