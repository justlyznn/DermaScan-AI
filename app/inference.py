import torch
import torch.nn.functional as F
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

def remove_hair(image):
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9))
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    ret, thresh2 = cv2.threshold(blackhat, 20, 255, cv2.THRESH_BINARY)
    thresh2 = cv2.dilate(thresh2, np.ones((3,3), np.uint8), iterations=1)
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
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_resized = cv2.resize(img, (256, 256))
    
    if use_preprocessing:
        img_clean = remove_hair(orig_resized)
        img_final = adaptive_clahe(img_clean)
    else:
        img_final = orig_resized
    
    img_tensor = img_final.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_tensor).permute(2, 0, 1).unsqueeze(0)
    
    return orig_resized, img_tensor, img_final


def compute_activation_heatmap(model, input_tensor, target_layer):
    """
    Fast activation-based heatmap using ONLY a forward pass.
    No backward pass / no gradients needed → very fast on CPU.
    
    Strategy: capture intermediate feature activations, average across
    channels, and use the mean activation map as the heatmap.
    This shows where the model is most 'active' in its decision.
    """
    captured = {}

    def fwd_hook(module, inp, out):
        # Detach immediately to avoid holding computation graph
        captured['feat'] = out.detach()

    handle = target_layer.register_forward_hook(fwd_hook)

    try:
        with torch.no_grad():
            output = model(input_tensor)

        # Average across channels → (H, W)
        feat = captured['feat'].cpu().numpy()[0]  # (C, H, W)
        cam  = np.mean(feat, axis=0)              # (H, W)

        # ReLU — keep only positive activations
        cam = np.maximum(cam, 0)

        H, W = input_tensor.shape[2], input_tensor.shape[3]
        cam = cv2.resize(cam, (W, H))

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, output

    finally:
        handle.remove()


def tensor_to_base64(img_np):
    img_pil = Image.fromarray(img_np.astype('uint8'))
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG", quality=85)
    return base64.b64encode(buffered.getvalue()).decode()


def process_inference(model, input_tensor, orig_image, preprocessed_image):
    model.eval()

    # Pick the target layer for activation heatmap
    if hasattr(model, 'decoder1'):
        if hasattr(model.decoder1, 'cbam'):
            target = model.decoder1.cbam       # CBAM attention block
        else:
            target = model.decoder1.conv_block  # Standard ResUNet
    else:
        children = list(model.children())
        target = children[-2] if len(children) >= 2 else children[-1]

    # Fast activation heatmap (forward-only, no backward)
    cam, output = compute_activation_heatmap(model, input_tensor, target)

    # Prediction mask from model output
    pred = output.squeeze().detach().cpu().numpy()
    pred_mask = (pred > 0.5).astype(np.uint8) * 255

    # Morphological post-processing for smooth edges
    kernel = np.ones((7, 7), np.uint8)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)
    pred_mask = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernel)

    # Blend Mask (white on black)
    mask_colored = np.zeros_like(orig_image)
    mask_colored[pred_mask == 255] = [255, 255, 255]

    # Blend heatmap with original image (JET colormap)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    cam_blended = cv2.addWeighted(orig_image, 0.5, heatmap, 0.5, 0)

    return (
        tensor_to_base64(orig_image),
        tensor_to_base64(preprocessed_image),
        tensor_to_base64(mask_colored),
        tensor_to_base64(cam_blended)
    )
