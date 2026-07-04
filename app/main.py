from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import torch
import os
import io

# Import models and inference logic
from app.model import ResUNet, CBAMResUNet
from app.inference import preprocess_image_pipeline, process_inference

app = FastAPI(title="SkinLens AI API")

# Setup directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Load models globally (on CPU for Hugging Face free tier compatibility)
DEVICE = torch.device('cpu')
print("Loading models to CPU...")

model_resunet = ResUNet(encoder_name='resnet34', pretrained=False).to(DEVICE)
model_cbam = CBAMResUNet(encoder_name='resnet34', pretrained=False).to(DEVICE)

# Path to weights
resunet_weights_path = os.path.join(BASE_DIR, "..", "models", "best_resunet_seed42.pth")
cbam_weights_path = os.path.join(BASE_DIR, "..", "models", "best_cbam_resunet_seed42.pth")

# Fallback for old Docker structure
if not os.path.exists(resunet_weights_path):
    resunet_weights_path = "/models/best_resunet_seed42.pth"
if not os.path.exists(cbam_weights_path):
    cbam_weights_path = "/models/best_cbam_resunet_seed42.pth"

if os.path.exists(resunet_weights_path):
    model_resunet.load_state_dict(torch.load(resunet_weights_path, map_location=DEVICE))
    model_resunet.eval()
    print("✅ ResUNet weights loaded successfully.")
else:
    print(f"❌ WARNING: ResUNet weights NOT FOUND at {resunet_weights_path}. Model will use random untrained weights!")

if os.path.exists(cbam_weights_path):
    model_cbam.load_state_dict(torch.load(cbam_weights_path, map_location=DEVICE))
    model_cbam.eval()
    print("✅ CBAM-ResUNet weights loaded successfully.")
else:
    print(f"❌ WARNING: CBAM-ResUNet weights NOT FOUND at {cbam_weights_path}. Model will use random untrained weights!")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/analyze")
async def analyze(
    file: UploadFile = File(None),
    sample_path: str = Form(None),
    model_type: str = Form(...),
    use_preprocessing: bool = Form(True)
):
    try:
        # Get image bytes
        if file:
            image_bytes = await file.read()
        elif sample_path:
            # Construct absolute path to sample image
            # sample_path comes in as '/static/samples/sample_1.jpg'
            # Strip the leading '/static/' or 'static/' to join with static folder
            rel_path = sample_path.replace("/static/", "").replace("static/", "")
            full_path = os.path.join(BASE_DIR, "static", rel_path)
            if not os.path.exists(full_path):
                raise HTTPException(status_code=404, detail="Sample image not found.")
            with open(full_path, "rb") as f:
                image_bytes = f.read()
        else:
            raise HTTPException(status_code=400, detail="No image provided.")

        # Preprocess
        orig_img, tensor_img, preprocessed_img = preprocess_image_pipeline(image_bytes, use_preprocessing)
        
        # Select Model
        if model_type == 'cbam':
            model = model_cbam
        else:
            model = model_resunet
            
        # Inference & GradCAM
        orig_b64, prep_b64, mask_b64, cam_b64 = process_inference(model, tensor_img, orig_img, preprocessed_img)
        
        return {
            "status": "success",
            "original": orig_b64,
            "preprocessed": prep_b64,
            "mask": mask_b64,
            "gradcam": cam_b64
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
