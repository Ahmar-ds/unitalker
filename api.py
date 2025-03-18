from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import os
import uuid
import shutil
import librosa
import numpy as np
import torch
from model import EmoTalk
from scipy.signal import savgol_filter
import random
import subprocess
import shlex
from argparse import Namespace
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="EmoTalk API", description="API for speech-driven 3D facial animation")

# Configuration
SETTINGS = {
    "model_args": {
        "bs_dim": 52,
        "feature_dim": 832,
        "period": 30,
        "max_seq_len": 5000,
        "num_workers": 0,
        "batch_size": 1
    },
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_path": "./pretrain_model/EmoTalk.pth",
    "blender_path": "./blender/blender",
    "post_processing": True,
    "temp_dir": "./tmp"
}

# Eye blink patterns
EYE_BLINKS = [
    np.array([0.36537236, 0.950235724, 0.95593375, 0.916715622, 0.367256105, 0.119113259, 0.025357503]),
    np.array([0.234776169, 0.909951985, 0.944758058, 0.777862132, 0.191071674, 0.235437036, 0.089163929]),
    np.array([0.870040774, 0.949833691, 0.949418545, 0.695911646, 0.191071674, 0.072576277, 0.007108896]),
    np.array([0.000307991, 0.556701422, 0.952656746, 0.942345619, 0.425857186, 0.148335218, 0.017659493])
]

# Initialize model
args = Namespace(**SETTINGS["model_args"])
model = EmoTalk(args)
try:
    model.load_state_dict(
        torch.load(SETTINGS["model_path"], map_location=torch.device(SETTINGS["device"])),
        strict=False
    )
    model = model.to(SETTINGS["device"])
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to initialize model")

def process_audio(wav_path: str, result_dir: str):
    """Process audio file and generate facial animation"""
    try:
        # Load audio
        speech_array, _ = librosa.load(wav_path, sr=16000)
        audio_tensor = torch.FloatTensor(speech_array).unsqueeze(0).to(SETTINGS["device"])
        
        # Generate predictions
        with torch.no_grad():
            level = torch.tensor([1]).to(SETTINGS["device"])
            person = torch.tensor([0]).to(SETTINGS["device"])
            prediction = model.predict(audio_tensor, level, person)
        
        prediction = prediction.squeeze().cpu().numpy()
        
        # Post-processing
        if SETTINGS["post_processing"]:
            output = np.zeros_like(prediction)
            for i in range(prediction.shape[1]):
                output[:, i] = savgol_filter(prediction[:, i], 5, 2)
            
            # Add eye blinks
            output[:, 8] = 0
            output[:, 9] = 0
            i = random.randint(0, 60)
            while i < output.shape[0] - 7:
                eye = random.choice(EYE_BLINKS)
                output[i:i+7, 8] = eye
                output[i:i+7, 9] = eye
                i += random.randint(60, 180)
        else:
            output = prediction
        
        # Save results
        file_name = os.path.splitext(os.path.basename(wav_path))[0]
        np.save(os.path.join(result_dir, f"{file_name}.npy"), output)
        return file_name
    
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise

def render_animation(result_dir: str, file_name: str, wav_path: str):
    """Render animation video using Blender and FFmpeg"""
    try:
        # Create paths
        image_dir = os.path.join(result_dir, file_name)
        os.makedirs(image_dir, exist_ok=True)
        output_video = os.path.join(result_dir, f"{file_name}.mp4")
        
        # Run Blender rendering
        blender_cmd = [
            SETTINGS["blender_path"],
            "-t", "64",
            "-b", "./render.blend",
            "-P", "./render.py",
            "--", result_dir, file_name
        ]
        subprocess.run(blender_cmd, check=True)
        
        # Combine with FFmpeg
        ffmpeg_cmd = [
            "ffmpeg",
            "-r", "30",
            "-i", os.path.join(image_dir, "%d.png"),
            "-i", wav_path,
            "-pix_fmt", "yuv420p",
            "-s", "512x768",
            output_video,
            "-y"
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        
        return output_video
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Rendering failed: {str(e)}")
        raise
    finally:
        # Cleanup temporary images
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)

@app.post("/animate", summary="Generate facial animation from audio")
async def generate_animation(audio_file: UploadFile = File(..., description="WAV audio file")):
    """Generate 3D facial animation from uploaded audio file"""
    # Create temporary workspace
    request_id = uuid.uuid4().hex
    workspace = os.path.join(SETTINGS["temp_dir"], request_id)
    os.makedirs(workspace, exist_ok=True)
    
    try:
        # Save uploaded file
        wav_path = os.path.join(workspace, audio_file.filename)
        with open(wav_path, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Process audio
        file_name = process_audio(wav_path, workspace)
        
        # Render video
        video_path = render_animation(workspace, file_name, wav_path)
        
        # Return result
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=f"{file_name}.mp4"
        )
    
    except Exception as e:
        logger.error(f"Request {request_id} failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temporary files
        if os.path.exists(workspace):
            shutil.rmtree(workspace, ignore_errors=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
