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


def compute_gradcam(model, input_tensor, target_layer):
    """
    Compute Grad-CAM using register_full_backward_hook (non-deprecated).
    Returns a normalized heatmap (H x W) in [0, 1].
    """
    activations = {}
    gradients = {}

    # Use the modern, non-deprecated hook API
    def fwd_hook(module, inp, out):
        activations['feat'] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients['feat'] = grad_out[0].detach()

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.eval()
        model.zero_grad()

        # Single forward + backward pass (gradient-enabled)
        output = model(input_tensor)
        score = output.sum()
        score.backward()

        grads = gradients['feat'].cpu().numpy()[0]       # (C, H, W)
        acts  = activations['feat'].cpu().numpy()[0]     # (C, H, W)

        # Global Average Pooling on gradients → per-channel weight
        weights = np.mean(grads, axis=(1, 2))            # (C,)

        # Weighted combination of activations
        cam = np.zeros(acts.shape[1:], dtype=np.float32) # (H, W)
        for i, w in enumerate(weights):
            cam += w * acts[i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)

        # Also get the final output prediction (no grad needed now)
        with torch.no_grad():
            output_final = model(input_tensor)

        return cam, output_final

    finally:
        # Always remove hooks to avoid memory leaks across requests
        fwd_handle.remove()
        bwd_handle.remove()
        model.zero_grad()


def tensor_to_base64(img_np):
    img_pil = Image.fromarray(img_np.astype('uint8'))
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()


def process_inference(model, input_tensor, orig_image, preprocessed_image):
    model.eval()
    
    # Identify the Grad-CAM target layer
    if hasattr(model, 'decoder1'):
        if hasattr(model.decoder1, 'cbam'):
            target = model.decoder1.cbam   # CBAM attention block
        else:
            target = model.decoder1.conv_block  # Standard ResUNet
    else:
        # Fallback: last conv-like layer before the output head
        children = list(model.children())
        target = children[-2] if len(children) >= 2 else children[-1]

    # Compute Grad-CAM + get prediction in a single pass
    cam, output = compute_gradcam(model, input_tensor, target)

    # Prediction mask
    pred = output.squeeze().detach().cpu().numpy()
    pred_mask = (pred > 0.5).astype(np.uint8) * 255
    
    # Post-Processing: Morfologi untuk menghaluskan tepi
    kernel = np.ones((7, 7), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)
    
    # Blend Mask
    mask_colored = np.zeros_like(orig_image)
    mask_colored[pred_mask == 255] = [255, 255, 255]
    
    # Blend Grad-CAM
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam_blended = cv2.addWeighted(orig_image, 0.5, heatmap, 0.5, 0)
    
    return (
        tensor_to_base64(orig_image),
        tensor_to_base64(preprocessed_image),
        tensor_to_base64(mask_colored),
        tensor_to_base64(cam_blended)
    )
