from diffusers_helper.hf_login import login

import os
import subprocess
import platform
import random
import time  # Add time module for duration tracking
import shutil
import glob
import re # Added for timestamp parsing
import math
from typing import Optional, Dict, Any # Added for type hinting
import sys # Added for RIFE
import cv2 # Added for RIFE
import json # <-- ADDED FOR PRESETS
from natsort import natsorted # <-- ADDED FOR BATCH SORTING

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse

from PIL import Image
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp, save_bcthw_as_gif, save_bcthw_as_apng, save_bcthw_as_webp, generate_new_timestamp, save_individual_frames
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.load_lora import load_lora, set_adapters # <-- IMPORT set_adapters
from diffusers_helper.bucket_tools import find_nearest_bucket


parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
args = parser.parse_args()

print(args)

# --- Model Names ---
MODEL_NAME_ORIGINAL = 'lllyasviel/FramePackI2V_HY'
MODEL_NAME_F1 = 'lllyasviel/FramePack_F1_I2V_HY_20250503'
MODEL_DISPLAY_NAME_ORIGINAL = "原始版FramePack"
MODEL_DISPLAY_NAME_F1 = "新版FramePack F1"
DEFAULT_MODEL_NAME = MODEL_DISPLAY_NAME_ORIGINAL

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = False # Set high_vram based on actual memory if needed free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

# --- Load common components ---
text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

# --- Load initial transformer (default) ---
print(f"Loading initial transformer model: {DEFAULT_MODEL_NAME}")
transformer: HunyuanVideoTransformer3DModelPacked # Type hint
if DEFAULT_MODEL_NAME == MODEL_DISPLAY_NAME_ORIGINAL:
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(MODEL_NAME_ORIGINAL, torch_dtype=torch.bfloat16).cpu()
    active_model_name = MODEL_DISPLAY_NAME_ORIGINAL
elif DEFAULT_MODEL_NAME == MODEL_DISPLAY_NAME_F1:
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(MODEL_NAME_F1, torch_dtype=torch.bfloat16).cpu()
    active_model_name = MODEL_DISPLAY_NAME_F1
else:
    raise ValueError(f"Unknown default model name: {DEFAULT_MODEL_NAME}")

print(f"Initial model '{active_model_name}' loaded to CPU.")

# --- Configure models ---
vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()
transformer.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

transformer.high_quality_fp32_output_for_inference = True
print('transformer.high_quality_fp32_output_for_inference = True')

# Set dtypes
transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

# Set requires_grad
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

# --- Apply memory management ---
if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu) # Keep text encoder swappable
    # VAE and Image Encoder will be loaded/unloaded manually as needed
else:
    # Keep non-transformer models on CPU initially in high VRAM mode too,
    # load them to GPU only when needed or keep them there if switching cost is low.
    # For simplicity matching low VRAM, we'll load them on demand.
    # If performance dictates, they can be moved permanently to GPU here.
    # transformer.to(gpu) # Transformer goes to GPU only when generating
    pass # Other models stay on CPU until needed

stream = AsyncStream()

# Change directory paths to use absolute paths
current_dir = os.path.dirname(os.path.abspath(__file__))
outputs_folder = os.path.join(current_dir, 'outputs')
used_images_folder = os.path.join(outputs_folder, 'used_images')
intermediate_videos_folder = os.path.join(outputs_folder, 'intermediate_videos')
gif_videos_folder = os.path.join(outputs_folder, 'gif_videos')
apng_videos_folder = os.path.join(outputs_folder, 'apng_videos')
webp_videos_folder = os.path.join(outputs_folder, 'webp_videos')
webm_videos_folder = os.path.join(outputs_folder, 'webm_videos')
intermediate_gif_videos_folder = os.path.join(outputs_folder, 'intermediate_gif_videos')
intermediate_apng_videos_folder = os.path.join(outputs_folder, 'intermediate_apng_videos')
intermediate_webp_videos_folder = os.path.join(outputs_folder, 'intermediate_webp_videos')
intermediate_webm_videos_folder = os.path.join(outputs_folder, 'intermediate_webm_videos')
individual_frames_folder = os.path.join(outputs_folder, 'individual_frames')
intermediate_individual_frames_folder = os.path.join(individual_frames_folder, 'intermediate_videos')
last_frames_folder = os.path.join(outputs_folder, 'last_frames')  # Add last_frames folder
intermediate_last_frames_folder = os.path.join(last_frames_folder, 'intermediate_videos')  # Add intermediate last frames folder
loras_folder = os.path.join(current_dir, 'loras')  # Add loras folder
presets_folder = os.path.join(current_dir, 'presets') # <-- ADDED FOR PRESETS
last_used_preset_file = os.path.join(presets_folder, '_lastused.txt') # <-- ADDED FOR PRESETS

# --- Add Global Stop Flag ---
batch_stop_requested = False
# --- End Add Global Stop Flag ---

# --- Global LoRA State --- (MODIFICATION 1 - Unchanged conceptually)
currently_loaded_lora_info = {
    "adapter_name": None,
    "lora_path": None # Store path for potential re-verification if needed
}
# --- End Global LoRA State ---

# Ensure all directories exist with proper error handling
for directory in [
    outputs_folder,
    used_images_folder,
    intermediate_videos_folder,
    gif_videos_folder,
    apng_videos_folder,
    webp_videos_folder,
    webm_videos_folder,
    intermediate_gif_videos_folder,
    intermediate_apng_videos_folder,
    intermediate_webp_videos_folder,
    intermediate_webm_videos_folder,
    individual_frames_folder,
    intermediate_individual_frames_folder,
    last_frames_folder,
    intermediate_last_frames_folder,
    loras_folder,
    presets_folder # <-- ADDED FOR PRESETS
]:
    try:
        os.makedirs(directory, exist_ok=True)
        # print(f"Created directory: {directory}") # Reduce console spam
    except Exception as e:
        print(f"Error creating directory {directory}: {str(e)}")

# Add batch processing output folder
outputs_batch_folder = os.path.join(outputs_folder, 'batch_outputs')
try:
    os.makedirs(outputs_batch_folder, exist_ok=True)
    print(f"Created batch outputs directory: {outputs_batch_folder}")
except Exception as e:
    print(f"Error creating batch outputs directory: {str(e)}")

def open_folder(folder_path):
    """Opens the specified folder in the file explorer/manager in a cross-platform way."""
    try:
        folder_path = os.path.abspath(folder_path)
        if not os.path.exists(folder_path):
             return f"Folder does not exist: {folder_path}"
        if platform.system() == "Windows":
            os.startfile(folder_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", folder_path])
        else:  # Linux
            subprocess.run(["xdg-open", folder_path])
        return f"Opened {os.path.basename(folder_path)} folder"
    except Exception as e:
        return f"Error opening folder: {str(e)}"

def print_supported_image_formats():
    """Print information about supported image formats for batch processing."""
    # List of extensions we handle
    extensions = [
        '.png', '.jpg', '.jpeg', '.bmp', '.webp',
        '.tif', '.tiff', '.gif', '.eps', '.ico',
        '.ppm', '.pgm', '.pbm', '.tga', '.exr', '.dib'
    ]

    # Check which formats PIL actually supports in this environment
    supported_formats = []
    unsupported_formats = []

    for ext in extensions:
        format_name = ext[1:].upper()  # Remove dot and convert to uppercase
        if format_name == 'JPG':
            format_name = 'JPEG'  # PIL uses JPEG, not JPG

        try:
            Image.init()
            if format_name in Image.ID or format_name in Image.MIME:
                supported_formats.append(ext)
            else:
                unsupported_formats.append(ext)
        except:
            unsupported_formats.append(ext)

    # print("\nSupported image formats for batch processing:")
    # print(", ".join(supported_formats))

    # if unsupported_formats:
    #     print("\nUnsupported formats in this environment:")
    #     print(", ".join(unsupported_formats))

    return supported_formats

def get_images_from_folder(folder_path):
    """Get all image files from a folder, sorted naturally."""
    if not folder_path or not os.path.exists(folder_path):
        return []

    # Get dynamically supported image formats
    image_extensions = print_supported_image_formats()
    images = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in image_extensions:
            images.append(file_path)

    # --- Use natsorted for natural sorting ---
    print(f"Found {len(images)} images. Sorting naturally...")
    sorted_images = natsorted(images)
    # print("Sorted file list:", sorted_images) # Optional: for debugging sort order
    return sorted_images
    # --- End natural sorting ---

def get_prompt_from_txt_file(image_path):
    """Check for a matching txt file with the same name as the image and return its content as prompt."""
    txt_path = os.path.splitext(image_path)[0] + '.txt'
    if os.path.exists(txt_path):
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                # Return raw content with newlines preserved, don't strip() the whole content
                content = f.read()
                return content.rstrip()  # Just remove trailing whitespace but keep newlines
        except Exception as e:
            print(f"Error reading prompt file {txt_path}: {str(e)}")
    return None

def format_time_human_readable(seconds):
    """Format time in a human-readable format (e.g., 3 min 11 seconds, 1 hour 12 min 15 seconds)."""
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{hours} hour{'s' if hours > 1 else ''} {minutes} min {seconds} seconds"
    elif minutes > 0:
        return f"{minutes} min {seconds} seconds"
    else:
        return f"{seconds} seconds"

def save_processing_metadata(output_path, metadata):
    """Save processing metadata to a text file."""
    metadata_path = os.path.splitext(output_path)[0] + '_metadata.txt'
    try:
        # Ensure directory exists before saving
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            # --- Add Model Name to Metadata ---
            f.write(f"Model: {metadata.pop('Model', 'Unknown')}\n") # Extract and write first
            # --- End Add Model Name ---
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
        return True
    except Exception as e:
        print(f"Error saving metadata to {metadata_path}: {str(e)}")
        return False

def move_and_rename_output_file(original_file, target_folder, original_image_filename):
    """Move and rename the output file to match the input image filename."""
    if not original_file or not os.path.exists(original_file):
        return None

    # Get the original extension
    ext = os.path.splitext(original_file)[1]

    # Create the new filename with the same name as the input image
    new_filename = os.path.splitext(original_image_filename)[0] + ext
    new_filepath = os.path.join(target_folder, new_filename)

    try:
        # Ensure target directory exists
        os.makedirs(os.path.dirname(new_filepath), exist_ok=True)

        # Copy instead of move to preserve the original in outputs folder
        import shutil
        shutil.copy2(original_file, new_filepath)
        print(f"Saved output to {new_filepath}")
        return new_filepath
    except Exception as e:
        print(f"Error moving/renaming file to {new_filepath}: {str(e)}")
        return None

# Add function to scan for LoRA files
def scan_lora_files():
    """Scan the loras folder for LoRA files (.safetensors or .pt) and return a list of them."""
    try:
        safetensors_files = glob.glob(os.path.join(loras_folder, "**/*.safetensors"), recursive=True)
        pt_files = glob.glob(os.path.join(loras_folder, "**/*.pt"), recursive=True)
        all_lora_files = safetensors_files + pt_files

        # Format for dropdown: Use basename without extension as display name, full path as value
        lora_options = [("None", "none")]  # Add "None" option
        for lora_file in all_lora_files:
            display_name = os.path.splitext(os.path.basename(lora_file))[0]
            lora_options.append((display_name, lora_file))

        return lora_options
    except Exception as e:
        print(f"Error scanning LoRA files: {str(e)}")
        return [("None", "none")]

def get_lora_path_from_name(display_name):
    """Convert a LoRA display name to its file path."""
    if display_name == "None":
        return "none"

    lora_options = scan_lora_files()
    for name, path in lora_options:
        if name == display_name:
            return path

    # If not found, return none
    print(f"Warning: LoRA '{display_name}' not found in options, using None")
    return "none"

def refresh_loras():
    """Refresh the LoRA dropdown with newly scanned files."""
    lora_options = scan_lora_files()
    return gr.update(choices=[name for name, _ in lora_options], value="None")

# --- safe_unload_lora and force_remove_lora_modules remain unchanged ---
def safe_unload_lora(model, device=None):
    """
    Safely unload LoRA weights from the model, handling different model types.

    Args:
        model: The model to unload LoRA weights from
        device: Optional device to move the model to before unloading
    """
    if device is not None:
        model.to(device)

    # Check if this is a DynamicSwap wrapped model
    is_dynamic_swap = 'DynamicSwap' in model.__class__.__name__

    try:
        # First try the standard unload_lora_weights method
        if hasattr(model, "unload_lora_weights"):
            print("Unloading LoRA using unload_lora_weights method")
            model.unload_lora_weights()
            # Also try disabling adapters if available, for good measure
            if hasattr(model, "disable_adapters"):
                model.disable_adapters()
                print("Additionally called disable_adapters.")
            if hasattr(model, "peft_config"):
                model.peft_config = {} # Clear peft config
                print("Cleared peft_config.")
            return True
        # Try peft's adapter handling if available
        elif hasattr(model, "peft_config") and model.peft_config:
            if hasattr(model, "disable_adapters"):
                print("Unloading LoRA using disable_adapters method")
                model.disable_adapters()
                model.peft_config = {} # Clear peft config
                print("Cleared peft_config.")
                return True
            # For PEFT models without disable_adapters method
            elif hasattr(model, "active_adapters") and model.active_adapters:
                print("Clearing active adapters list")
                model.active_adapters = []
                model.peft_config = {} # Clear peft config
                print("Cleared peft_config.")
                return True
        # Special handling for DynamicSwap models
        elif is_dynamic_swap:
            print("DynamicSwap model detected, attempting to reset internal model state")

            # For DynamicSwap models, try to check if there's an internal model that has LoRA attributes
            internal_model = model # Start with the wrapper itself
            if hasattr(model, "model"): # Check if it has an inner .model attribute
                internal_model = model.model

            print(f"Attempting unload on model type: {type(internal_model).__name__}")

            unloaded_internal = False
            if hasattr(internal_model, "unload_lora_weights"):
                print("Unloading LoRA from internal model using unload_lora_weights")
                internal_model.unload_lora_weights()
                unloaded_internal = True
            if hasattr(internal_model, "peft_config") and internal_model.peft_config:
                if hasattr(internal_model, "disable_adapters"):
                    print("Disabling adapters on internal model")
                    internal_model.disable_adapters()
                if hasattr(internal_model, "active_adapters"):
                     internal_model.active_adapters = []
                internal_model.peft_config = {} # Clear peft config
                print("Cleared peft_config on internal model.")
                unloaded_internal = True

            if unloaded_internal:
                # Also clear config on the wrapper if present
                if hasattr(model, "peft_config"): model.peft_config = {}
                if hasattr(model, "active_adapters"): model.active_adapters = []
                print("Cleared LoRA state on DynamicSwap wrapper.")
                return True

            # If all else fails with DynamicSwap, try to directly remove LoRA modules
            print("Attempting direct LoRA module removal as fallback")
            return force_remove_lora_modules(model) # Pass the original wrapper
        else:
            print("No LoRA adapter found to unload")
            return True
    except Exception as e:
        print(f"Error during LoRA unloading: {str(e)}")
        traceback.print_exc()

        # Last resort - try direct module removal
        print("Attempting direct LoRA module removal after error")
        return force_remove_lora_modules(model)

    return False

def force_remove_lora_modules(model):
    """
    Force-remove LoRA modules by directly modifying the model's state.
    This is a last-resort method when normal unloading fails.

    Args:
        model: The model to remove LoRA modules from

    Returns:
        bool: Whether the operation was successful
    """
    try:
        # Look for any LoRA-related modules
        lora_removed = False
        # Iterate through modules of the potentially wrapped model
        model_to_check = model.model if hasattr(model, "model") else model
        print(f"Force removing modules from: {type(model_to_check).__name__}")

        for name, module in list(model_to_check.named_modules()):
            # Check for typical PEFT/LoRA module names or base classes
            is_lora_layer = hasattr(module, 'lora_A') or hasattr(module, 'lora_B') or 'lora' in name.lower()
            is_peft_layer = 'lora' in getattr(module, '__class__', type(None)).__module__.lower() # Check if module comes from peft.lora

            if is_lora_layer or is_peft_layer:
                print(f"Found potential LoRA module: {name}")
                lora_removed = True

                # Get parent module and attribute name from the checked model
                parent_name, _, attr_name = name.rpartition('.')
                if parent_name:
                    try:
                        parent = model_to_check.get_submodule(parent_name)
                        if hasattr(parent, attr_name):
                            # Try to restore original module if possible (peft often stores it)
                            original_module = getattr(module, 'base_layer', getattr(module, 'model', getattr(module, 'base_model', None))) # Common names for base layer in peft
                            if original_module is not None:
                                setattr(parent, attr_name, original_module)
                                print(f"Restored original module for {name}")
                            else:
                                print(f"Could not find original module for {name} to restore.")
                        else:
                             print(f"Parent module {parent_name} does not have attribute {attr_name}")
                    except Exception as e:
                        print(f"Error accessing parent module {parent_name}: {str(e)}")
                else:
                    # Handle top-level modules if necessary
                    if hasattr(model_to_check, name):
                         original_module = getattr(module, 'base_layer', getattr(module, 'model', getattr(module, 'base_model', None)))
                         if original_module is not None:
                            setattr(model_to_check, name, original_module)
                            print(f"Restored original top-level module for {name}")


        # Clear PEFT configuration on both wrapper and inner model if they exist
        for m in [model, model_to_check]:
             if hasattr(m, "peft_config"):
                 m.peft_config = {}
                 print(f"Cleared peft_config on {type(m).__name__}")
                 lora_removed = True
             if hasattr(m, "active_adapters"):
                 m.active_adapters = []
                 print(f"Cleared active_adapters on {type(m).__name__}")
                 lora_removed = True

        return lora_removed
    except Exception as e:
        print(f"Error during force LoRA removal: {str(e)}")
        traceback.print_exc()
        return False

print_supported_image_formats()

# --- Preset Functions START --- (Unchanged)
def get_preset_path(name: str) -> str:
    """Constructs the full path for a preset file."""
    # Sanitize name slightly to prevent path traversal issues, though Gradio input might handle some of this.
    # A more robust sanitization might be needed depending on security requirements.
    safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).rstrip()
    if not safe_name:
        safe_name = "Unnamed_Preset"
    return os.path.join(presets_folder, f"{safe_name}.json")

def scan_presets() -> list[str]:
    """Scans the presets folder for .json files and returns a list of names."""
    presets = ["Default"] # Always include Default
    try:
        os.makedirs(presets_folder, exist_ok=True) # Ensure folder exists
        for filename in os.listdir(presets_folder):
            if filename.endswith(".json") and filename != "Default.json":
                preset_name = os.path.splitext(filename)[0]
                if preset_name != "_lastused": # Exclude internal file if saved as json
                    presets.append(preset_name)
    except Exception as e:
        print(f"Error scanning presets folder: {e}")
    return sorted(list(set(presets))) # Ensure Default is present and list is unique and sorted

def save_last_used_preset_name(name: str):
    """Saves the name of the last used preset."""
    try:
        with open(last_used_preset_file, 'w', encoding='utf-8') as f:
            f.write(name)
    except Exception as e:
        print(f"Error saving last used preset name: {e}")

def load_last_used_preset_name() -> Optional[str]:
    """Loads the name of the last used preset."""
    if os.path.exists(last_used_preset_file):
        try:
            with open(last_used_preset_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error loading last used preset name: {e}")
    return None

def create_default_preset_if_needed(components_values: dict):
    """Creates Default.json if it doesn't exist, using current component values."""
    default_path = get_preset_path("Default")
    if not os.path.exists(default_path):
        print("Default preset not found, creating...")
        try:
            # Filter out None values which might occur during initial setup if components aren't fully ready
            # Although we will call this later with actual default values.
            valid_values = {k: v for k, v in components_values.items() if v is not None}
            if valid_values: # Only save if we have some values
                with open(default_path, 'w', encoding='utf-8') as f:
                    json.dump(valid_values, f, indent=4)
                print("Created Default.json")
            else:
                print("Warning: Could not create Default.json - no valid component values provided.")
        except Exception as e:
            print(f"Error creating default preset: {e}")

def load_preset_data(name: str) -> Optional[dict]:
    """Loads preset data from a JSON file."""
    if not name:
        return None
    preset_path = get_preset_path(name)
    if os.path.exists(preset_path):
        try:
            with open(preset_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading preset '{name}': {e}")
            return None
    else:
        print(f"Preset file not found: {preset_path}")
        return None
# --- Preset Functions END ---

# --- save_last_frame_to_file remains unchanged ---
def save_last_frame_to_file(frames, output_dir, filename_base):
    """Save only the last frame from a video frames tensor (bcthw format).

    Args:
        frames: Tensor of frames in bcthw format
        output_dir: Specific directory to save the last frame (subfolder within last_frames)
        filename_base: Base filename to use for the saved frame (e.g., 'timestamp_lastframe')
    """
    try:
        # Make sure output directory exists (already done before calling, but safe)
        os.makedirs(output_dir, exist_ok=True)

        # Extract only the last frame but keep the tensor structure (b,c,t,h,w)
        # where t=1 to be compatible with save_individual_frames function
        if frames is None:
            print("Error: frames tensor is None")
            return None

        # Check if frames has the expected shape (b,c,t,h,w)
        if not (isinstance(frames, torch.Tensor) and len(frames.shape) == 5):
            print(f"Error: Invalid frames tensor shape: {frames.shape if hasattr(frames, 'shape') else 'unknown'}")
            return None

        try:
            last_frame_tensor = frames[:, :, -1:, :, :]  # Slicing keeps the dimension
        except Exception as slicing_error:
            print(f"Error slicing last frame: {str(slicing_error)}")
            print(f"Frames tensor shape: {frames.shape}")
            return None

        # Use the existing utils function to ensure consistent color processing
        try:
            from diffusers_helper.utils import save_individual_frames  # Local import ok here
            # Call save_individual_frames with the single frame tensor and the base name
            # It will append '_0000.png' by default, resulting in filename_base_0000.png
            frame_paths = save_individual_frames(last_frame_tensor, output_dir, filename_base, return_frame_paths=True)
        except ImportError:
            print("Error importing save_individual_frames, trying to import at global scope")
            # Try to import at global scope if local import fails
            import sys
            sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diffusers_helper'))
            from utils import save_individual_frames
            frame_paths = save_individual_frames(last_frame_tensor, output_dir, filename_base, return_frame_paths=True)

        if frame_paths and len(frame_paths) > 0:
            print(f"Saved last frame to {frame_paths[0]}")
            return frame_paths[0]
        else:
            print("No frames were saved by save_individual_frames")
            return None
    except Exception as e:
        print(f"Error saving last frame: {str(e)}")
        traceback.print_exc()
        return None

# --- parse_simple_timestamped_prompt remains unchanged ---
def parse_simple_timestamped_prompt(prompt_text: str, total_duration: float, latent_window_size: int, fps: int) -> Optional[list[tuple[float, str]]]:
    """
    Parses prompts in the format '[seconds] prompt_text' per line.
    No longer reverses the timestamps - processes them in the order written.

    Args:
        prompt_text: The full prompt text, potentially multi-line.
        total_duration: Total video duration in seconds.
        latent_window_size: Latent window size (e.g., 9).
        fps: Frames per second (e.g., 30).

    Returns:
        A list of tuples sorted by start time: [(start_time, prompt), ...],
        or None if the format is not detected or invalid.
    """
    lines = prompt_text.strip().split('\n')
    sections = []
    has_timestamps = False
    default_prompt = None

    # Regex to match '[seconds]' at the start of a line - updated to handle both [0] and [0s] formats
    pattern = r'^\s*\[(\d+(?:\.\d+)?(?:s)?)\]\s*(.*)'

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        match = re.match(pattern, line)
        if match:
            has_timestamps = True
            try:
                time_str = match.group(1)
                # Remove 's' suffix if present
                if time_str.endswith('s'):
                    time_str = time_str[:-1]
                time_sec = float(time_str)
                text = match.group(2).strip()
                if text: # Only add if there's actual prompt text
                    sections.append({"original_time": time_sec, "prompt": text})
            except ValueError:
                print(f"Warning: Invalid time format in line: {line}")
                return None # Indicate failure if format is wrong
        elif i == 0 and not has_timestamps:
            # If the first line doesn't have a timestamp, consider it the default
            # only if *no* timestamps are found later. We'll handle this after the loop.
             default_prompt = line
        elif has_timestamps:
             # If we've already seen timestamps, ignore subsequent lines without them
             print(f"Warning: Ignoring line without timestamp after timestamped lines detected: {line}")

    # Rest of the function with modifications to prevent reversing
    if not has_timestamps:
        # No timestamp format detected, return None to signal using the original prompt as is
        return None

    if not sections and default_prompt:
         # Edge case: Only a default prompt line was found, but we were expecting timestamps.
         # Treat as invalid timestamp format.
         return None
    elif not sections:
         # No valid sections found at all
         return None

    # Sort by original time
    sections.sort(key=lambda x: x["original_time"])

    # Add a 0s section if one doesn't exist, using the first prompt found if needed
    if not any(s['original_time'] == 0.0 for s in sections):
        first_prompt = sections[0]['prompt'] if sections else "default prompt"
        sections.insert(0, {"original_time": 0.0, "prompt": first_prompt})
        sections.sort(key=lambda x: x["original_time"]) # Re-sort after insertion

    # Create the final sections list - NO LONGER REVERSING
    final_sections = []
    for i in range(len(sections)):
        start_time = sections[i]["original_time"]
        prompt_text = sections[i]["prompt"]
        final_sections.append((start_time, prompt_text))

    print(f"Parsed timestamped prompts (in original order): {final_sections}")
    return final_sections

# --- update_iteration_info remains unchanged ---
def update_iteration_info(vid_len_s, fps_val, win_size):
    """Calculates and formats information about generation sections."""

    # --- ADD TYPE CHECKING AND CASTING ---
    try:
        # Ensure inputs are numeric before calculations
        # Handle potential None values during initial setup if necessary
        vid_len_s = float(vid_len_s) if vid_len_s is not None else 0.0
        fps_val = int(fps_val) if fps_val is not None else 0
        win_size = int(win_size) if win_size is not None else 0
        print(f"DEBUG update_iteration_info - Converted inputs: vid_len_s={vid_len_s}, fps_val={fps_val}, win_size={win_size}")
    except (ValueError, TypeError, AttributeError) as e:
        print(f"Error converting types in update_iteration_info: {e}")
        print(f"Received types: vid_len_s={type(vid_len_s)}, fps_val={type(fps_val)}, win_size={type(win_size)}")
        # Return an error message or default if types are wrong
        return "Error: Invalid input types for calculation."
    # --- END TYPE CHECKING ---

    # (Keep the initial validation and calculation parts the same)
    if fps_val <= 0 or win_size <= 0:
        return "Invalid FPS or Latent Window Size."

    try:
        # Calculate total sections using the same logic as the worker
        total_frames_needed = vid_len_s * fps_val
        frames_per_section_calc = win_size * 4 # Used for section count and duration timing

        # Calculate total sections needed (ensure division by zero doesn't happen)
        total_latent_sections = 0
        if frames_per_section_calc > 0:
            # --- ADJUSTMENT FOR F1 MODEL ---
            # F1 model generates sections differently (extends from start frame).
            # The concept of 'total sections' is less direct than the original overlapping window approach.
            # However, for user feedback and prompt timing, we can estimate based on the original logic.
            # The number of loops in the F1 worker determines generation length.
            # Let's keep the original calculation for the info display, but add a note if F1 is active.
            # The actual loop count in worker for F1 depends on total_second_length, fps, and latent_window_size.
            # F1 adds 'latent_window_size * 4 - 3' frames per loop after the first frame.
            frames_to_generate = total_frames_needed - 1 # Minus the start frame
            frames_per_f1_loop = (win_size * 4 - 3)
            if frames_per_f1_loop > 0:
                total_f1_loops = math.ceil(frames_to_generate / frames_per_f1_loop) if frames_to_generate > 0 else 0
                total_latent_sections = max(total_f1_loops, 1) # Ensure at least 1 if duration > 0
            else:
                return "Invalid parameters leading to zero frames per section."
            # --- END F1 ADJUSTMENT ---
        else:
             return "Invalid parameters leading to zero frames per section." # Handle division by zero case


        # Calculate the exact duration represented per section for timing
        section_duration_seconds = frames_per_section_calc / fps_val

        # Calculate total frames generated in one section (including overlaps) for info
        frames_in_one_section = win_size * 4 - 3

        # Create message highlighting exact timing if properly configured
        timing_description = ""
        if abs(section_duration_seconds - 1.0) < 0.01: # Within 0.01 of exactly 1 second
            timing_description = "**precisely 1.0 second**"
        else:
            timing_description = f"~**{section_duration_seconds:.2f} seconds**"

        # --- Update info text based on model ---
        global active_model_name # Need to know which model is active
     

        # --- CORRECTED TIP (Applies mainly to Original model's timing) ---
        if active_model_name == MODEL_DISPLAY_NAME_ORIGINAL:
            ideal_lws_float = fps_val / 4.0
            ideal_lws_int = round(ideal_lws_float)
            ideal_lws_clamped = max(1, min(ideal_lws_int, 33)) # Clamp to slider range [1, 33]

            if win_size != ideal_lws_clamped:
                ideal_duration = (ideal_lws_clamped * 4) / fps_val
                if abs(ideal_duration - 1.0) < 0.01:
                     info_text += f"\n\n*Tip: Set Latent Window Size to **{ideal_lws_clamped}** for (near) exact 1-second sections at {fps_val} FPS.*"

        return info_text
    except Exception as e:
        print(f"Error calculating iteration info: {e}")
        traceback.print_exc() # Add traceback
        return "匹夫与你同在"
# --- End Updated Function ---


# --- MODIFICATION 3a: Modify worker signature ADD active_model ---
@torch.no_grad()
def worker(input_image, end_image, prompt, n_prompt, seed, use_random_seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality='high', export_gif=False, export_apng=False, export_webp=False, num_generations=1, resolution="640", fps=30,
         # --- LoRA Args Changed ---
         adapter_name_to_use: str = "None", # Receive adapter name
         lora_scale: float = 1.0,           # Keep scale
         # --- End LoRA Args Changed ---
         save_individual_frames_flag=False, save_intermediate_frames_flag=False, save_last_frame_flag=False, use_multiline_prompts_flag=False, rife_enabled=False, rife_multiplier="2x FPS",
         # --- ADDED ---
         active_model: str = MODEL_DISPLAY_NAME_ORIGINAL # Default to original if not passed
         ):
    # --- MODIFICATION 3b: Remove LoRA loading/cleanup block --- (Already Done)

    # Declare globals needed elsewhere in the function
    global transformer, text_encoder, text_encoder_2, image_encoder, vae
    global individual_frames_folder, intermediate_individual_frames_folder, last_frames_folder, intermediate_last_frames_folder # Ensure these are accessible if modified

    # --- Calculate total sections based on active model ---
    total_latent_sections = 0
    frames_per_section_calc = latent_window_size * 4
    if frames_per_section_calc <= 0:
        raise ValueError("Invalid Latent Window Size or FPS leading to zero frames per section")

    total_frames_needed = total_second_length * fps

    if active_model == MODEL_DISPLAY_NAME_F1:
        # F1 calculation: loops needed to generate enough frames after the start frame
        frames_to_generate = total_frames_needed - 1
        frames_per_f1_loop = latent_window_size * 4 - 3
        if frames_per_f1_loop <= 0: raise ValueError("Invalid LWS for F1 model")
        total_latent_sections = math.ceil(frames_to_generate / frames_per_f1_loop) if frames_to_generate > 0 else 0
        total_latent_sections = max(total_latent_sections, 1) # Ensure at least 1 loop if duration > 0
        print(f"F1 Model: Calculated {total_latent_sections} generation loops needed.")
    else: # Original Model calculation
        total_latent_sections = int(max(round(total_frames_needed / frames_per_section_calc), 1))
        print(f"Original Model: Calculated {total_latent_sections} sections needed.")
    # --- End section calculation ---

    # --- Check for timestamped prompts (No changes here) ---
    parsed_prompts = None
    encoded_prompts = {} # Dictionary to store encoded prompts {prompt_text: (tensors)}
    using_timestamped_prompts = False
    if not use_multiline_prompts_flag:
        parsed_prompts = parse_simple_timestamped_prompt(prompt, total_second_length, latent_window_size, fps)
        if parsed_prompts:
            using_timestamped_prompts = True
            print("Using timestamped prompts.")
        else:
            print("Timestamped prompt format not detected or invalid, using the entire prompt as one.")
    else:
        print("Multi-line prompts enabled, skipping timestamp parsing.")
    # --- End timestamped prompt check ---


    current_seed = seed
    all_outputs = {}
    last_used_seed = seed

    # Timing variables
    start_time = time.time() # Overall start time for the whole worker call (for total time calculation)
    generation_times = []

    # Post-processing time estimates (in seconds)
    estimated_vae_time_per_frame = 0.05  # Estimate for VAE decoding per frame
    estimated_save_time = 2.0  # Estimate for saving video to disk

    # If we have past generation data, we can update these estimates
    vae_time_history = []
    save_time_history = []

    for gen_idx in range(num_generations):
        # Track start time for current generation
        gen_start_time = time.time() # <<<--- START TIME FOR THIS SPECIFIC GENERATION

        if stream.input_queue.top() == 'end':
            stream.output_queue.push(('end', None))
            print("Worker detected end signal at start of generation")
            return

        # Update seed for this generation
        if use_random_seed:
            current_seed = random.randint(1, 2147483647)
        elif gen_idx > 0:  # increment seed for non-random seeds after first generation
            current_seed += 1

        last_used_seed = current_seed
        stream.output_queue.push(('seed_update', current_seed))

        # Use the new naming scheme for job_id
        job_id = generate_new_timestamp()
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Starting generation {gen_idx+1}/{num_generations} with seed {current_seed} using {active_model}...'))))

        try:
            # Clean GPU
            if not high_vram:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer # Unload all potentially loaded models
                )

            # Text encoding
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

            if not high_vram:
                # Load text encoders to GPU only when needed
                fake_diffusers_current_device(text_encoder, gpu) # Use fake device for faster potential swap
                load_model_as_complete(text_encoder_2, target_device=gpu)

            # --- Pre-encode prompts (No changes needed here) ---
            if using_timestamped_prompts:
                unique_prompts = set(p[1] for p in parsed_prompts)
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Encoding {len(unique_prompts)} unique timestamped prompts...'))))
                for p_text in unique_prompts:
                    if p_text not in encoded_prompts:
                         llama_vec_p, clip_l_pooler_p = encode_prompt_conds(p_text, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                         llama_vec_p, llama_attention_mask_p = crop_or_pad_yield_mask(llama_vec_p, length=512)
                         encoded_prompts[p_text] = (llama_vec_p, llama_attention_mask_p, clip_l_pooler_p)
                print(f"Pre-encoded {len(encoded_prompts)} unique prompts.")
                if not parsed_prompts:
                    raise ValueError("Timestamped prompts were detected but parsing resulted in an empty list.")
                # Use the prompt for time 0.0 as the initial prompt
                initial_prompt_text = "default prompt"
                for t, p_txt in parsed_prompts:
                     if t == 0.0:
                         initial_prompt_text = p_txt
                         break
                     # If no 0.0 found explicitly, the first one (sorted by time) will be used implicitly later.
                     elif not initial_prompt_text or t < parsed_prompts[0][0]: # Find the earliest one if 0.0 isn't there
                         initial_prompt_text = p_txt

                if initial_prompt_text not in encoded_prompts:
                     # This case shouldn't happen if parsing worked, but as a fallback:
                     print(f"Warning: Initial prompt text '{initial_prompt_text}' not found in encoded prompts. Using first available.")
                     initial_prompt_text = list(encoded_prompts.keys())[0]

                llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[initial_prompt_text]

            else:
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Encoding single prompt...'))))
                llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
                encoded_prompts[prompt] = (llama_vec, llama_attention_mask, clip_l_pooler)

            # Handle negative prompt encoding
            if cfg > 1.0: # Only encode negative if CFG needs it
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
            else:
                # Create zero tensors if CFG is 1.0 (or less, though UI minimum is 1.0)
                first_prompt_key = list(encoded_prompts.keys())[0]
                ref_llama_vec = encoded_prompts[first_prompt_key][0]
                ref_clip_l = encoded_prompts[first_prompt_key][2]
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(ref_llama_vec), torch.zeros_like(ref_clip_l)
                ref_llama_mask = encoded_prompts[first_prompt_key][1]
                llama_attention_mask_n = torch.zeros_like(ref_llama_mask)

            # Move encoded prompts to correct dtype (do this once)
            target_dtype = transformer.dtype # Use transformer's dtype (bfloat16)
            for p_text in encoded_prompts:
                 l_vec, l_mask, c_pool = encoded_prompts[p_text]
                 encoded_prompts[p_text] = (l_vec.to(target_dtype), l_mask, c_pool.to(target_dtype))

            llama_vec_n = llama_vec_n.to(target_dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(target_dtype)
            # --- End prompt encoding ---

            # Determine bucket size based on start image
            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=resolution)
            print(f"Found best resolution bucket {width} x {height}")

            # Processing input image (start frame)
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing start frame ...'))))
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            try:
                os.makedirs(used_images_folder, exist_ok=True)
                Image.fromarray(input_image_np).save(os.path.join(used_images_folder, f'{job_id}_start.png'))
                print(f"Saved start image to {os.path.join(used_images_folder, f'{job_id}_start.png')}")
            except Exception as e:
                print(f"Error saving start image: {str(e)}")
            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None] # B C T H W (T=1)

            # Processing end image (if provided) - Only relevant for Original model?
            # F1 demo doesn't show end image use. Assume it should work but might be less effective.
            has_end_image = end_image is not None
            end_image_np = None
            end_image_pt = None
            end_latent = None # Initialize end_latent
            if has_end_image:
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing end frame ...'))))
                end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
                try:
                    os.makedirs(used_images_folder, exist_ok=True)
                    Image.fromarray(end_image_np).save(os.path.join(used_images_folder, f'{job_id}_end.png'))
                    print(f"Saved end image to {os.path.join(used_images_folder, f'{job_id}_end.png')}")
                except Exception as e:
                    print(f"Error saving end image: {str(e)}")
                end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
                end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None] # B C T H W (T=1)

            # VAE encoding
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
            if not high_vram:
                load_model_as_complete(vae, target_device=gpu) # Load VAE to GPU
            start_latent = vae_encode(input_image_pt, vae)
            if has_end_image:
                end_latent = vae_encode(end_image_pt, vae) # Encode end latent
            # Offload VAE if not high VRAM
            if not high_vram:
                 unload_complete_models(vae) # Unload VAE after use

            # CLIP Vision encoding
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu) # Load Image Encoder to GPU

            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
            if has_end_image:
                end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
                end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
                # Combine embeddings (simple average) - F1 might use this differently or not at all
                image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2.0
                print("Combined start and end frame CLIP vision embeddings.")
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(target_dtype)

            # Offload Image Encoder if not high VRAM
            if not high_vram:
                unload_complete_models(image_encoder)

            # --- LORA LOADING BLOCK DELETED (Handled externally before worker) ---

            # Sampling
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Start sampling ({active_model}) generation {gen_idx+1}/{num_generations}...'))))

            rnd = torch.Generator("cpu").manual_seed(current_seed)
            num_frames = latent_window_size * 4 - 3 # Frames generated per section/loop

            # --- Initialize history based on model ---
            history_latents = None
            history_pixels = None
            total_generated_latent_frames = 0 # Counter for total latent frames including start

            if active_model == MODEL_DISPLAY_NAME_F1:
                # F1 starts with the encoded start latent and builds from there
                # F1 history needs space for 1 (start) + 16 (4x) + 2 (2x) + 1 (1x) = 20 reference frames? Check F1 demo.
                # Demo: history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
                # Demo then does: history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
                # Let's simplify: Start with start_latent directly.
                history_latents = start_latent.clone().cpu() # Start with the first frame latent on CPU
                total_generated_latent_frames = 1
                print(f"F1 Initial history latent shape: {history_latents.shape}")
            else: # Original Model
                # Original model uses padding and overlapping windows. History stores full context.
                # Needs space for: 1 (clean_pre) + latent_padding_size + latent_window_size + 1 (clean_post) + 2 (2x) + 16 (4x)
                # Let's stick to the original calculation size, it adapts based on padding.
                # Max padding is ~ total_latent_sections * latent_window_size. Max LWS=33, max sections high.
                # Original code used fixed size, let's keep that structure:
                history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
                # history_latents will be populated within the loop for the original model.
                total_generated_latent_frames = 0 # Start frame isn't counted until the first section completes
                print(f"Original Model Initial history latent shape: {history_latents.shape}")
            # --- End history initialization ---


            # --- Loop setup based on model ---
            loop_iterator = None
            if active_model == MODEL_DISPLAY_NAME_F1:
                loop_iterator = range(total_latent_sections) # F1 loops 'total_latent_sections' times
            else: # Original Model
                # Original model uses padding logic
                base_latent_paddings = reversed(range(total_latent_sections))
                if total_latent_sections > 4:
                    latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
                else:
                    latent_paddings = list(base_latent_paddings)
                loop_iterator = enumerate(latent_paddings) # Original iterates through paddings
            # --- End loop setup ---

            current_prompt_text_for_callback = prompt

            # --- Main Generation Loop ---
            for loop_info in loop_iterator:
                # --- Unpack loop info based on model ---
                latent_padding = 0
                i = 0 # Loop index
                if active_model == MODEL_DISPLAY_NAME_F1:
                    i = loop_info # F1 just gives the index
                    is_last_section = (i == total_latent_sections - 1)
                    is_first_section = (i == 0)
                    # F1 doesn't use latent_padding in the same way
                    print(f'F1 Loop {i+1}/{total_latent_sections}, is_last_section = {is_last_section}')
                else: # Original Model
                    i, latent_padding = loop_info
                    is_last_section = latent_padding == 0
                    is_first_section = (i == 0)
                    latent_padding_size = latent_padding * latent_window_size
                    print(f'Original Loop {i+1}/{total_latent_sections}, latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')
                # --- End unpack loop info ---

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    print(f"Worker detected end signal during {active_model} loop")
                    try:
                        if not high_vram: unload_complete_models() # Clean up any loaded models
                    except Exception as cleanup_error: print(f"Error during cleanup: {str(cleanup_error)}")
                    return

                # --- Determine current prompt tensors (Timestamp logic) ---
                first_prompt_key = list(encoded_prompts.keys())[0] # Fallback prompt
                active_llama_vec, active_llama_mask, active_clip_pooler = encoded_prompts[first_prompt_key]
                current_prompt_text_for_callback = first_prompt_key

                if using_timestamped_prompts:
                    # Calculate current video time based on frames generated so far
                    current_frames_generated = 0
                    if active_model == MODEL_DISPLAY_NAME_F1:
                        # F1: Start frame + frames from previous loops
                        current_frames_generated = 1 + i * (latent_window_size * 4 - 3)
                    else: # Original Model
                        # Original: Need to estimate based on section progress
                        # This uses the reversed generation logic - time corresponds to *final* video time
                        section_duration_seconds = (latent_window_size * 4) / fps
                        # Time decreases as we iterate through reversed padding
                        current_video_time = total_second_length - (i * section_duration_seconds)
                        # Failsafe clamp
                        if current_video_time < 0: current_video_time = 0.0
                        current_frames_generated = int(current_video_time * fps) # Approx frame number

                    current_video_time_sec = current_frames_generated / fps

                    print(f"\n===== PROMPT DEBUG INFO ({active_model}) =====")
                    print(f"Loop/Iter: {i} / {total_latent_sections -1}")
                    print(f"Current video time (estimated): {current_video_time_sec:.2f}s (Frame ~{current_frames_generated})")
                    print(f"Available prompts: {parsed_prompts}")

                    # Find the prompt active at this time
                    selected_prompt_text = parsed_prompts[0][1] # Default to the earliest
                    last_matching_time = parsed_prompts[0][0]
                    epsilon = 1e-4 # Small tolerance for float comparison

                    print(f"Checking against prompts...")
                    for start_time_prompt, p_text in parsed_prompts:
                        print(f"  - Checking time {start_time_prompt:.2f}s ('{p_text[:20]}...') vs current_video_time {current_video_time_sec:.2f}s")
                        # Prompt activates if current time is >= its start time
                        if current_video_time_sec >= (start_time_prompt - epsilon):
                             selected_prompt_text = p_text
                             last_matching_time = start_time_prompt
                             print(f"    - MATCH: Current time {current_video_time_sec:.2f}s >= {start_time_prompt}s. Tentative selection: '{selected_prompt_text[:20]}...'")
                        else:
                            # Since prompts are sorted by time, we can stop once current time is less
                            print(f"    - NO MATCH: Current time {current_video_time_sec:.2f}s < {start_time_prompt}s. Stopping search.")
                            break

                    print(f"Final selected prompt active at/before {current_video_time_sec:.2f}s is the one from {last_matching_time}s: '{selected_prompt_text}'")
                    print(f"===== END DEBUG INFO ({active_model}) =====\n")

                    # Use the selected prompt's encoded tensors
                    active_llama_vec, active_llama_mask, active_clip_pooler = encoded_prompts[selected_prompt_text]
                    current_prompt_text_for_callback = selected_prompt_text
                    print(f'---> Generating section corresponding to video time >= {last_matching_time:.2f}s, Using prompt: "{selected_prompt_text[:50]}..."')

                else: # Not using timestamped prompts, use the single full prompt
                     active_llama_vec, active_llama_mask, active_clip_pooler = encoded_prompts[prompt]
                     current_prompt_text_for_callback = prompt
                # --- End prompt tensor determination ---


                # --- Prepare Latent Indices and Clean Latents based on Model ---
                latent_indices = None
                clean_latents = None
                clean_latent_indices = None
                clean_latents_2x = None
                clean_latent_2x_indices = None
                clean_latents_4x = None
                clean_latent_4x_indices = None

                if active_model == MODEL_DISPLAY_NAME_F1:
                    # F1 Indices (based on demo_gradio_f1.py)
                    # indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
                    # clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
                    # clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)

                    # Recalculate F1 indices to match expected input shapes for sample_hunyuan
                    # sample_hunyuan expects: latent_indices(LWS), clean_latent_indices(2), clean_latent_2x_indices(2), clean_latent_4x_indices(16)
                    # F1 Demo Indices: clean_start(1), 4x(16), 2x(2), 1x(1), latent(LWS)
                    # Let's map F1 demo indices to sample_hunyuan expected indices:
                    total_f1_indices = 1 + 16 + 2 + 1 + latent_window_size
                    indices = torch.arange(0, total_f1_indices).unsqueeze(0)

                    # Split according to F1 demo structure first
                    f1_clean_start_idx, f1_clean_4x_idx, f1_clean_2x_idx, f1_clean_1x_idx, f1_latent_idx = indices.split([1, 16, 2, 1, latent_window_size], dim=1)

                    # Map to sample_hunyuan's expectations:
                    latent_indices = f1_latent_idx
                    clean_latent_indices = torch.cat([f1_clean_start_idx, f1_clean_1x_idx], dim=1) # Should be shape [1, 2]
                    clean_latent_2x_indices = f1_clean_2x_idx # Should be shape [1, 2]
                    clean_latent_4x_indices = f1_clean_4x_idx # Should be shape [1, 16]

                    # F1 Clean Latents (based on demo_gradio_f1.py)
                    # Needs last 16+2+1 frames from history_latents
                    required_history_len = 16 + 2 + 1
                    if history_latents.shape[2] < required_history_len:
                        # Pad history if it's too short (should only happen on first loop if start frame isn't enough)
                        padding_needed = required_history_len - history_latents.shape[2]
                        padding_tensor = torch.zeros(
                            (history_latents.shape[0], history_latents.shape[1], padding_needed, history_latents.shape[3], history_latents.shape[4]),
                            dtype=history_latents.dtype, device=history_latents.device
                        )
                        current_history_segment = torch.cat([padding_tensor, history_latents], dim=2).to(gpu) # Move padded segment to GPU
                        print(f"F1 Warning: Padded history by {padding_needed} frames.")
                    else:
                        current_history_segment = history_latents[:, :, -required_history_len:, :, :].to(gpu) # Get last N frames and move to GPU

                    # Split the segment according to F1 demo structure
                    clean_latents_4x, clean_latents_2x, clean_latents_1x = current_history_segment.split([16, 2, 1], dim=2)

                    # Map to sample_hunyuan's expectations:
                    clean_latents = torch.cat([start_latent.to(gpu), clean_latents_1x], dim=2) # Needs start_latent + 1x latent

                else: # Original Model
                    # Original Indices (based on original app.py)
                    latent_padding_size = latent_padding * latent_window_size # Recalculate here
                    total_original_indices = sum([1, latent_padding_size, latent_window_size, 1, 2, 16])
                    indices = torch.arange(0, total_original_indices).unsqueeze(0)
                    clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1) # Shape [1, 2]

                    # Original Clean Latents (based on original app.py)
                    # Needs start_latent, end_latent (maybe), and history
                    clean_latents_pre = start_latent.to(gpu) # Move start latent to GPU
                    # Get reference frames from history (these might be zeros initially)
                    clean_latents_post_orig, clean_latents_2x_orig, clean_latents_4x_orig = history_latents[:, :, :1 + 2 + 16, :, :].to(gpu).split([1, 2, 16], dim=2)

                    clean_latents_post = clean_latents_post_orig
                    # Use end_latent ONLY if available AND it's the first section (highest padding)
                    if has_end_image and is_first_section and end_latent is not None:
                        clean_latents_post = end_latent.to(gpu) # Use end latent, move to GPU
                        print("Using end_latent for clean_latents_post in the first section.")

                    # Combine clean latents
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                    clean_latents_2x = clean_latents_2x_orig
                    clean_latents_4x = clean_latents_4x_orig

                # --- Move Model to GPU ---
                if not high_vram:
                    unload_complete_models() # Ensure others are unloaded
                    move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                else:
                    # In high VRAM, assume transformer is already on GPU OR move it now if it wasn't persistent
                    if transformer.device != gpu:
                         print("Moving transformer to GPU (High VRAM mode)...")
                         transformer.to(gpu)
                # --- End Move Model to GPU ---

                # --- NEW: Explicitly Sync LoRA Parameters to GPU ---
                if adapter_name_to_use != "None":
                    print(f"Worker: Ensuring LoRA adapter '{adapter_name_to_use}' parameters are on {gpu}...")
                    try:
                        lora_params_synced = 0
                        for name, param in transformer.named_parameters():
                            if 'lora_' in name or f"'{adapter_name_to_use}'" in name:
                                if param.device != gpu:
                                    param.data = param.data.to(gpu)
                                    lora_params_synced += 1
                        if lora_params_synced > 0:
                             print(f"Worker: Synced {lora_params_synced} LoRA parameters to {gpu}.")
                    except Exception as sync_err:
                        print(f"Worker ERROR: Failed to sync LoRA parameters to {gpu}: {sync_err}")
                        traceback.print_exc()
                # --- END NEW LoRA Sync Block ---


                # --- Apply LoRA Scale ---
                if adapter_name_to_use != "None":
                    try:
                        set_adapters(transformer, [adapter_name_to_use], [lora_scale])
                        print(f"Worker: Applied scale {lora_scale} to adapter '{adapter_name_to_use}'")
                    except Exception as e:
                         print(f"Worker ERROR applying LoRA scale: {e}")
                         traceback.print_exc()
                elif hasattr(transformer, 'disable_adapters'): # Ensure disabled if target is None
                     try:
                         transformer.disable_adapters()
                     except Exception as e:
                         print(f"Trying to disable adapters (this is not an error if you did not select any LoRA): {e}")
                # --- End Apply LoRA Scale ---

                # --- Initialize TeaCache based on threshold value ---
                # F1 demo uses teacache=True by default. Let's use the UI toggle.
                use_teacache_effective = teacache_threshold > 0.0
                if use_teacache_effective:
                    print(f"TeaCache: Enabled (Threshold: {teacache_threshold})")
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_threshold)
                else:
                    print("TeaCache: Disabled")
                    transformer.initialize_teacache(enable_teacache=False)
                # --- End Initialize TeaCache ---

                sampling_start_time = time.time()

                # --- Callback definition (No changes here, uses global total_generated_latent_frames) ---
                def callback(d):
                    preview = d['denoised']
                    preview = vae_decode_fake(preview) # Uses global vae
                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        print("\n" + "="*50)
                        print("USER REQUESTED TO END GENERATION - STOPPING...")
                        print("="*50)
                        raise KeyboardInterrupt('User ends the task.')

                    current_step = d['i'] + 1
                    percentage = int(100.0 * current_step / steps)
                    elapsed_time = time.time() - sampling_start_time
                    time_per_step = elapsed_time / current_step if current_step > 0 else 0
                    remaining_steps = steps - current_step
                    eta_seconds = time_per_step * remaining_steps

                    # Calculate expected frames differently for ETA
                    # This preview shows intermediate *latent* frames being denoised
                    # Use the number of frames in the current sampling call
                    expected_latent_frames_in_section = latent_indices.shape[1] # LWS usually
                    expected_output_frames_from_section = num_frames # latent_window_size * 4 - 3

                    if current_step == steps:
                        # Estimate time for VAE decode + save for the frames *from this section*
                        post_processing_eta = expected_output_frames_from_section * estimated_vae_time_per_frame + estimated_save_time
                        eta_seconds = post_processing_eta
                    else:
                        # Add VAE/save estimate for this section to sampling ETA
                        post_processing_eta = expected_output_frames_from_section * estimated_vae_time_per_frame + estimated_save_time
                        eta_seconds += post_processing_eta

                    eta_str = format_time_human_readable(eta_seconds)
                    total_elapsed = time.time() - gen_start_time # Time for this specific generation index
                    elapsed_str = format_time_human_readable(total_elapsed)
                    hint = f'Sampling {current_step}/{steps} (Gen {gen_idx+1}/{num_generations}, Seed {current_seed}, Loop {i+1}/{total_latent_sections})'

                    # Calculate total *output* frames generated so far for description
                    total_output_frames_so_far = 0
                    if active_model == MODEL_DISPLAY_NAME_F1:
                        # F1: Start frame + (loops completed * frames per loop)
                         total_output_frames_so_far = 1 + i * num_frames
                         # Add estimate for current incomplete loop based on step progress
                         total_output_frames_so_far += int((current_step / steps) * num_frames)
                    else: # Original Model
                        # Original: Estimate based on history_pixels if available, otherwise guess from loop index
                        if history_pixels is not None:
                             total_output_frames_so_far = history_pixels.shape[2]
                        else: # Rough estimate before first decode
                             total_output_frames_so_far = i * num_frames # Approximation

                    desc = f'Total generated frames: ~{int(max(0, total_output_frames_so_far))}, Video length: {max(0, total_output_frames_so_far / fps) :.2f} seconds (FPS-{fps}).'
                    if using_timestamped_prompts:
                        desc += f' Current Prompt: "{current_prompt_text_for_callback[:50]}..."'
                    time_info = f'Elapsed: {elapsed_str} | ETA: {eta_str}'
                    print(f"\rProgress: {percentage}% | {hint} | {time_info}     ", end="")
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, f"{hint}<br/>{time_info}"))))
                    return
                # --- End callback ---

                try:
                    # --- sample_hunyuan call (Arguments prepared above based on model) ---
                    generated_latents = sample_hunyuan(
                        transformer=transformer, # Global transformer, should be on GPU now
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=num_frames,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
                        num_inference_steps=steps,
                        generator=rnd,
                        prompt_embeds=active_llama_vec.to(gpu), # Ensure prompt tensors are on GPU
                        prompt_embeds_mask=active_llama_mask.to(gpu),
                        prompt_poolers=active_clip_pooler.to(gpu),
                        negative_prompt_embeds=llama_vec_n.to(gpu),
                        negative_prompt_embeds_mask=llama_attention_mask_n.to(gpu),
                        negative_prompt_poolers=clip_l_pooler_n.to(gpu),
                        device=gpu,
                        dtype=target_dtype, # Use transformer's dtype
                        image_embeddings=image_encoder_last_hidden_state.to(gpu), # Ensure image embed on GPU
                        latent_indices=latent_indices.to(gpu),
                        clean_latents=clean_latents.to(gpu), # Already moved to GPU during prep
                        clean_latent_indices=clean_latent_indices.to(gpu),
                        clean_latents_2x=clean_latents_2x.to(gpu), # Already moved to GPU
                        clean_latent_2x_indices=clean_latent_2x_indices.to(gpu),
                        clean_latents_4x=clean_latents_4x.to(gpu), # Already moved to GPU
                        clean_latent_4x_indices=clean_latent_4x_indices.to(gpu),
                        callback=callback,
                    )
                except ConnectionResetError as e:
                    print(f"Connection Reset Error caught during sampling: {str(e)}")
                    print("Continuing with the process anyway...")
                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        return
                    # Create empty latents to avoid breaking downstream code
                    empty_shape = (1, 16, latent_window_size, height // 8, width // 8)
                    generated_latents = torch.zeros(empty_shape, dtype=torch.float32).cpu()
                    print("Skipping to next generation due to connection error")
                    break # Skip current generation's remaining loops

                section_time = time.time() - sampling_start_time
                print(f"\nSection/Loop {i+1} completed sampling in {section_time:.2f} seconds")

                # --- Update History Latents (on CPU) ---
                if active_model == MODEL_DISPLAY_NAME_F1:
                    history_latents = torch.cat([history_latents, generated_latents.cpu()], dim=2)
                    total_generated_latent_frames = history_latents.shape[2] # Update total count
                else: # Original Model
                    # Original logic: add start latent only on last section, update history
                    current_section_latents = generated_latents.cpu()
                    if is_last_section:
                        # Prepend start latent if it's the final (first generated) section
                        current_section_latents = torch.cat([start_latent.cpu(), current_section_latents], dim=2)
                    # Update history by prepending the new section's latents
                    history_latents = torch.cat([current_section_latents, history_latents], dim=2)
                    # Keep track of total frames added (crude estimate, refined by decode)
                    total_generated_latent_frames += int(current_section_latents.shape[2])

                print(f"Updated history_latents shape (CPU): {history_latents.shape}, Total latent frames: {total_generated_latent_frames}")

                # --- VAE Decoding ---
                print(f"VAE decoding started... {'Using standard decoding' if high_vram else 'Using memory optimization: VAE offloading'}")
                if not high_vram:
                    offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation) # Offload transformer
                    load_model_as_complete(vae, target_device=gpu) # Load VAE

                vae_start_time = time.time()

                # --- Decode logic based on model ---
                # We need the full sequence generated *so far* for decoding
                # The 'history_latents' on CPU should contain this
                # However, the original model's 'history_latents' includes padding/context not meant for final output.
                # Let's refine the VAE input based on the model.

                latents_to_decode = None
                if active_model == MODEL_DISPLAY_NAME_F1:
                    # F1 history is straightforward: just the sequence generated
                    latents_to_decode = history_latents.to(gpu) # Move the whole history to GPU
                else: # Original Model
                    # Original model needs slicing to get only the 'real' frames
                    # Use total_generated_latent_frames as the count from the 'end' (which is start of tensor)
                    latents_to_decode = history_latents[:, :, :total_generated_latent_frames, :, :].to(gpu)

                print(f"Latents to decode shape: {latents_to_decode.shape}")

                # --- Soft Append Logic (Only needed for Original Model?) ---
                # F1 demo decodes the relevant part each time. Original used soft_append.
                # Let's try decoding the full relevant history each time for both models for simplicity,
                # unless performance becomes an issue. Soft append adds complexity.

                # Decode the full sequence generated so far
                current_pixels = vae_decode(latents_to_decode, vae).cpu() # Decode on GPU, move result to CPU
                history_pixels = current_pixels # Overwrite history_pixels with the latest full sequence

                # --- Original Soft Append Logic (commented out for unified approach) ---
                # if active_model != MODEL_DISPLAY_NAME_F1:
                #     if history_pixels is None:
                #         history_pixels = vae_decode(latents_to_decode, vae).cpu()
                #     else:
                #         # Calculate frames decoded in this section
                #         section_latent_frames = latents_to_decode.shape[2] # Frames in current decode batch
                #         # How many frames overlap? Depends on LWS
                #         overlapped_frames = latent_window_size * 4 - 3 # Check if this is right for soft append
                #         current_pixels_section = vae_decode(latents_to_decode, vae).cpu() # Decode only the latest part? Needs adjustment
                #         # Soft append needs careful implementation if used. Sticking to full decode for now.
                #         history_pixels = soft_append_bcthw(current_pixels_section, history_pixels, overlapped_frames) # Needs adjustment
                # else: # F1 - just decode the history
                #      history_pixels = vae_decode(latents_to_decode, vae).cpu()
                # --- End Original Soft Append Logic ---


                vae_time = time.time() - vae_start_time
                num_frames_decoded = history_pixels.shape[2] # Use actual decoded frames count
                vae_time_per_frame = vae_time / num_frames_decoded if num_frames_decoded > 0 else estimated_vae_time_per_frame
                vae_time_history.append(vae_time_per_frame)
                if len(vae_time_history) > 0:
                    estimated_vae_time_per_frame = sum(vae_time_history) / len(vae_time_history)
                print(f"VAE decoding completed in {vae_time:.2f} seconds ({vae_time_per_frame:.3f} sec/frame)")
                print(f'Decoded pixel shape {history_pixels.shape}')

                # --- Offload VAE if needed ---
                if not high_vram:
                    unload_complete_models(vae) # Unload VAE after use

                # --- Saving logic (mostly unchanged) ---
                is_intermediate = not is_last_section
                # Use job_id and frame count for intermediate naming consistency
                intermediate_suffix = f"_{history_pixels.shape[2]}frames" # Use actual pixel frame count
                output_filename_base = f"{job_id}"
                if is_intermediate:
                     output_filename_base += intermediate_suffix

                output_filename = os.path.join(intermediate_videos_folder if is_intermediate else outputs_folder, f'{output_filename_base}.mp4')
                webm_output_filename = os.path.join(intermediate_webm_videos_folder if is_intermediate else webm_videos_folder, f'{output_filename_base}.webm')

                save_start_time = time.time()
                try:
                    # --- Existing MP4 Saving ---
                    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                    save_bcthw_as_mp4(history_pixels, output_filename, fps=fps, video_quality=video_quality)
                    print(f"Saved MP4 video to {output_filename}")

                    # --- SAVE LAST FRAME (MP4 ONLY) ---
                    if save_last_frame_flag is True and output_filename and os.path.exists(output_filename):
                        try:
                            print(f"Attempting to save last frame for {output_filename}")
                            last_frame_base_name_for_save = os.path.splitext(os.path.basename(output_filename))[0]
                            frames_output_dir = os.path.join(
                                intermediate_last_frames_folder if is_intermediate else last_frames_folder,
                                last_frame_base_name_for_save # Subfolder named after the video file base
                            )
                            os.makedirs(frames_output_dir, exist_ok=True)
                            save_last_frame_to_file(history_pixels, frames_output_dir, f"{last_frame_base_name_for_save}_lastframe")
                        except Exception as lf_err:
                            print(f"Error saving last frame for {output_filename}: {str(lf_err)}")
                            traceback.print_exc()
                    # --- END SAVE LAST FRAME ---

                    # --- START OF RIFE INTEGRATION ---
                    if rife_enabled and output_filename and os.path.exists(output_filename):
                        print(f"RIFE Enabled: Processing {output_filename}")
                        try:
                            cap = cv2.VideoCapture(output_filename)
                            source_fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                            print(f"Source MP4 FPS: {source_fps:.2f}")

                            if source_fps <= 60: # Limit RIFE application
                                multiplier_val = "4" if rife_multiplier == "4x FPS" else "2"
                                print(f"Using RIFE multiplier: {multiplier_val}x")
                                rife_output_filename = os.path.splitext(output_filename)[0] + '_extra_FPS.mp4'
                                print(f"RIFE output filename: {rife_output_filename}")
                                rife_script_path = os.path.abspath(os.path.join(current_dir, "Practical-RIFE", "inference_video.py"))
                                rife_model_path = os.path.abspath(os.path.join(current_dir, "Practical-RIFE", "train_log"))

                                if not os.path.exists(rife_script_path): print(f"ERROR: RIFE script not found at {rife_script_path}")
                                elif not os.path.exists(rife_model_path): print(f"ERROR: RIFE model directory not found at {rife_model_path}")
                                else:
                                    cmd = (
                                        f'"{sys.executable}" "{rife_script_path}" '
                                        f'--model="{rife_model_path}" '
                                        f'--multi={multiplier_val} '
                                        f'--video="{os.path.abspath(output_filename)}" '
                                        f'--output="{os.path.abspath(rife_output_filename)}"'
                                    )
                                    print(f"Executing RIFE command: {cmd}")
                                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ)
                                    if result.returncode == 0:
                                        if os.path.exists(rife_output_filename):
                                            print(f"Successfully applied RIFE. Saved as: {rife_output_filename}")
                                            stream.output_queue.push(('rife_file', rife_output_filename)) # Yield RIFE file path
                                        else:
                                             print(f"RIFE command succeeded but output file missing: {rife_output_filename}")
                                             print(f"RIFE stdout:\n{result.stdout}")
                                             print(f"RIFE stderr:\n{result.stderr}")
                                    else:
                                        print(f"Error applying RIFE (return code {result.returncode}).")
                                        print(f"RIFE stdout:\n{result.stdout}")
                                        print(f"RIFE stderr:\n{result.stderr}")
                            else:
                                print(f"Skipping RIFE because source FPS ({source_fps:.2f}) is > 60.")
                        except Exception as rife_err:
                            print(f"Error during RIFE processing for {output_filename}: {str(rife_err)}")
                            traceback.print_exc()
                    # --- END OF RIFE INTEGRATION ---

                    # --- Existing WebM Saving ---
                    if video_quality == 'web_compatible':
                        os.makedirs(os.path.dirname(webm_output_filename), exist_ok=True)
                        save_bcthw_as_mp4(history_pixels, webm_output_filename, fps=fps, video_quality=video_quality, format='webm')
                        print(f"Saved WebM video to {webm_output_filename}")

                    # Save individual frames if enabled
                    if ((is_intermediate and save_intermediate_frames_flag) or
                        (not is_intermediate and save_individual_frames_flag)):
                        frames_output_dir = os.path.join(
                            intermediate_individual_frames_folder if is_intermediate else individual_frames_folder,
                            os.path.splitext(os.path.basename(output_filename))[0] # Subfolder named after video
                        )
                        from diffusers_helper.utils import save_individual_frames # Import here just in case
                        save_individual_frames(history_pixels, frames_output_dir, job_id) # Use job_id as base for frame names
                        print(f"Saved individual frames to {frames_output_dir}")

                except ConnectionResetError as e:
                    print(f"Connection Reset Error during video saving: {str(e)}")
                    print("Continuing with the process anyway...")
                    output_filename = None
                    webm_output_filename = None
                except Exception as e:
                    print(f"Error saving MP4/WebM video or associated last frame: {str(e)}")
                    traceback.print_exc()
                    output_filename = None
                    webm_output_filename = None

                # --- Metadata Saving ---
                save_metadata_enabled = True # Assume true (controlled by UI checkbox later)
                if save_metadata_enabled and is_last_section: # Only save final metadata
                    gen_time_current = time.time() - gen_start_time
                    generation_time_seconds = int(gen_time_current)
                    generation_time_formatted = format_time_human_readable(gen_time_current)
                    metadata_prompt = prompt # Save the original full prompt
                    if using_timestamped_prompts:
                        # Include the full multiline prompt if timestamps were used
                         metadata_prompt = prompt

                    metadata = {
                        "Model": active_model, # <-- ADDED MODEL NAME
                        "Prompt": metadata_prompt,
                        "Negative Prompt": n_prompt,
                        "Seed": current_seed,
                        "TeaCache": f"Enabled (Threshold: {teacache_threshold})" if teacache_threshold > 0.0 else "Disabled",
                        "Video Length (seconds)": total_second_length,
                        "FPS": fps,
                        "Latent Window Size": latent_window_size,
                        "Steps": steps,
                        "CFG Scale": cfg,
                        "Distilled CFG Scale": gs,
                        "Guidance Rescale": rs,
                        "Resolution": resolution,
                        "Generation Time": generation_time_formatted,
                        "Total Seconds": f"{generation_time_seconds} seconds",
                        "Start Frame Provided": True,
                        "End Frame Provided": has_end_image,
                        "Timestamped Prompts Used": using_timestamped_prompts,
                    }
                    if adapter_name_to_use != "None":
                        metadata["LoRA"] = adapter_name_to_use
                        metadata["LoRA Scale"] = lora_scale

                    # Save metadata for all generated final formats
                    if output_filename: save_processing_metadata(output_filename, metadata.copy())
                    final_gif_path = os.path.join(gif_videos_folder, f'{job_id}.gif')
                    if export_gif and os.path.exists(final_gif_path):
                        save_processing_metadata(final_gif_path, metadata.copy())
                    final_apng_path = os.path.join(apng_videos_folder, f'{job_id}.png')
                    if export_apng and os.path.exists(final_apng_path):
                         save_processing_metadata(final_apng_path, metadata.copy())
                    final_webp_path = os.path.join(webp_videos_folder, f'{job_id}.webp')
                    if export_webp and os.path.exists(final_webp_path):
                         save_processing_metadata(final_webp_path, metadata.copy())
                    final_webm_path = os.path.join(webm_videos_folder, f'{job_id}.webm')
                    if video_quality == 'web_compatible' and final_webm_path and os.path.exists(final_webm_path):
                        save_processing_metadata(final_webm_path, metadata.copy())
                    rife_final_path = os.path.splitext(output_filename)[0] + '_extra_FPS.mp4' if output_filename else None
                    if rife_enabled and rife_final_path and os.path.exists(rife_final_path):
                         save_processing_metadata(rife_final_path, metadata.copy())

                # --- End Metadata Saving ---


                # Save additional formats (GIF, APNG, WebP)
                try:
                    gif_filename = os.path.join(intermediate_gif_videos_folder if is_intermediate else gif_videos_folder, f'{output_filename_base}.gif')
                    if export_gif:
                        try:
                            os.makedirs(os.path.dirname(gif_filename), exist_ok=True)
                            save_bcthw_as_gif(history_pixels, gif_filename, fps=fps)
                            print(f"Saved GIF animation to {gif_filename}")
                        except Exception as e: print(f"Error saving GIF: {str(e)}")

                    apng_filename = os.path.join(intermediate_apng_videos_folder if is_intermediate else apng_videos_folder, f'{output_filename_base}.png')
                    if export_apng:
                        try:
                            os.makedirs(os.path.dirname(apng_filename), exist_ok=True)
                            save_bcthw_as_apng(history_pixels, apng_filename, fps=fps)
                            print(f"Saved APNG animation to {apng_filename}")
                        except Exception as e: print(f"Error saving APNG: {str(e)}")

                    webp_filename = os.path.join(intermediate_webp_videos_folder if is_intermediate else webp_videos_folder, f'{output_filename_base}.webp')
                    if export_webp:
                        try:
                            os.makedirs(os.path.dirname(webp_filename), exist_ok=True)
                            save_bcthw_as_webp(history_pixels, webp_filename, fps=fps)
                            print(f"Saved WebP animation to {webp_filename}")
                        except Exception as e: print(f"Error saving WebP: {str(e)}")
                except ConnectionResetError as e:
                    print(f"Connection Reset Error during additional format saving: {str(e)}")
                    print("Continuing with the process anyway...")

                save_time = time.time() - save_start_time
                save_time_history.append(save_time)
                if len(save_time_history) > 0:
                    estimated_save_time = sum(save_time_history) / len(save_time_history)
                print(f"Saving operations completed in {save_time:.2f} seconds")

                # Yield the primary output file (MP4 or WebM) to the UI
                primary_output_file = output_filename
                if video_quality == 'web_compatible' and webm_output_filename and os.path.exists(webm_output_filename):
                     primary_output_file = webm_output_filename
                stream.output_queue.push(('file', primary_output_file))

                if is_last_section:
                    break # End of generation loop for this generation index

            # --- Generation Loop Timing ---
            gen_time_completed = time.time() - gen_start_time
            generation_times.append(gen_time_completed)
            avg_gen_time = sum(generation_times) / len(generation_times)
            remaining_gens = num_generations - (gen_idx + 1)
            estimated_remaining_time = avg_gen_time * remaining_gens

            print(f"\nGeneration {gen_idx+1}/{num_generations} completed in {gen_time_completed:.2f} seconds")
            if remaining_gens > 0:
                print(f"Estimated time for remaining generations: {estimated_remaining_time/60:.1f} minutes")

            stream.output_queue.push(('timing', {'gen_time': gen_time_completed, 'avg_time': avg_gen_time, 'remaining_time': estimated_remaining_time}))
            # --- End Generation Loop Timing ---

        except KeyboardInterrupt as e:
            if str(e) == 'User ends the task.':
                print("\n" + "="*50 + "\nGENERATION ENDED BY USER\n" + "="*50)
                if not high_vram:
                    print("Unloading models from memory...")
                    unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                stream.output_queue.push(('end', None))
                return # Exit worker function
            else: raise # Re-raise other KeyboardInterrupts
        except ConnectionResetError as e:
            print(f"Connection Reset Error outside main processing loop: {str(e)}")
            print("Trying to continue with next generation if possible...")
            if not high_vram:
                try: unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                except Exception as cleanup_error: print(f"Error during memory cleanup: {str(cleanup_error)}")
            if gen_idx == num_generations - 1: # If it was the last planned generation
                stream.output_queue.push(('end', None))
                return # Exit worker
            continue # Continue to next generation index
        except Exception as e:
            print("\n" + "="*50 + f"\nERROR DURING GENERATION: {str(e)}\n" + "="*50)
            traceback.print_exc()
            print("="*50)
            if not high_vram:
                try: unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                except Exception as cleanup_error: print(f"Error during memory cleanup after exception: {str(cleanup_error)}")
            if gen_idx == num_generations - 1: # If it was the last planned generation
                stream.output_queue.push(('end', None))
                return # Exit worker
            continue # Continue to next generation index

    # --- End of all generations loop ---
    total_time_worker = time.time() - start_time # Use overall worker start time for total duration
    print(f"\nTotal worker time for {num_generations} generation(s): {total_time_worker:.2f} seconds ({total_time_worker/60:.2f} minutes)")

    # --- Final cleanup (ensure models used in loop are unloaded if low VRAM) ---
    if not high_vram:
        unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer) # Final attempt to clear

    stream.output_queue.push(('final_timing', {'total_time': total_time_worker, 'generation_times': generation_times}))
    stream.output_queue.push(('final_seed', last_used_seed))
    stream.output_queue.push(('end', None))
    return

# --- Helper Function manage_lora_structure (Unchanged conceptually, acts on global transformer) ---
def manage_lora_structure(selected_lora_dropdown_value: str):
    """
    Ensures the correct LoRA structure is loaded/unloaded onto the global transformer.
    MUST be called when the transformer is expected to be on the CPU.
    """
    global transformer, currently_loaded_lora_info, text_encoder, text_encoder_2, image_encoder, vae # Need other models for unload check

    lora_path = get_lora_path_from_name(selected_lora_dropdown_value) # Get full path

    target_adapter_name = None
    if lora_path != "none":
        target_adapter_name = os.path.splitext(os.path.basename(lora_path))[0]

    # Check if the target is different from what's currently loaded
    if target_adapter_name != currently_loaded_lora_info["adapter_name"]:
        print(f"LoRA Change Detected: Target='{target_adapter_name}', Current='{currently_loaded_lora_info['adapter_name']}'")

        # --- Ensure model is on CPU before modification ---
        transformer_on_cpu = False
        if transformer.device == cpu or (hasattr(transformer, 'model') and transformer.model.device == cpu): # Check wrapper and internal model device
             transformer_on_cpu = True
        elif not high_vram:
             # If low VRAM, assume DynamicSwap handled it or unload manually
             print("Ensuring transformer is on CPU before LoRA structure change (Low VRAM mode)...")
             unload_complete_models(transformer) # Attempt to unload it fully
             transformer_on_cpu = True # Assume success for now
        else: # High VRAM mode, explicitly move to CPU
             try:
                 print("Moving transformer to CPU for LoRA structure change (High VRAM mode)...")
                 transformer.to(cpu)
                 torch.cuda.empty_cache()
                 transformer_on_cpu = True
             except Exception as e:
                 print(f"Warning: Failed to move transformer to CPU in high VRAM mode: {e}")
                 # Proceed cautiously, LoRA ops might fail if still on GPU
        # --- End CPU Check ---

        if not transformer_on_cpu:
             print("ERROR: Transformer could not be confirmed on CPU. Aborting LoRA structure change.")
             raise RuntimeError("Failed to ensure transformer is on CPU for LoRA modification.")


        # 1. Unload existing LoRA structure if one is loaded
        if currently_loaded_lora_info["adapter_name"] is not None:
            print(f"Unloading previous LoRA structure: {currently_loaded_lora_info['adapter_name']}")
            unload_success = safe_unload_lora(transformer, cpu) # Pass cpu hint
            if unload_success:
                print(f"Successfully unloaded {currently_loaded_lora_info['adapter_name']}.")
                currently_loaded_lora_info["adapter_name"] = None
                currently_loaded_lora_info["lora_path"] = None
            else:
                print(f"ERROR: Failed to unload LoRA {currently_loaded_lora_info['adapter_name']}! State may be corrupt.")
                # Don't raise error here, allow attempting to load the new one over it as a fallback
                # raise RuntimeError(f"Failed to unload previous LoRA '{currently_loaded_lora_info['adapter_name']}'. Cannot safely proceed.")

        # 2. Load new LoRA structure if target is not "None"
        if target_adapter_name is not None and lora_path != "none":
            print(f"Loading new LoRA structure: {target_adapter_name} from {lora_path}")
            try:
                lora_dir, lora_filename = os.path.split(lora_path)
                # Ensure load_lora works correctly with DynamicSwap models if applicable
                load_lora(transformer, lora_dir, lora_filename) # Assume modification in-place on CPU model
                print(f"Successfully loaded structure for {target_adapter_name}.")
                currently_loaded_lora_info["adapter_name"] = target_adapter_name
                currently_loaded_lora_info["lora_path"] = lora_path
            except Exception as e:
                print(f"ERROR loading LoRA structure for {target_adapter_name}: {e}")
                traceback.print_exc()
                currently_loaded_lora_info["adapter_name"] = None # Reset state on failure
                currently_loaded_lora_info["lora_path"] = None
                raise RuntimeError(f"Failed to load LoRA structure '{target_adapter_name}'.")
        else:
             print("Target LoRA is 'None'. No new structure loaded.")
             currently_loaded_lora_info["adapter_name"] = None # Ensure state is None if target is None
             currently_loaded_lora_info["lora_path"] = None

        # --- Optional: Move model back to GPU if in high_vram mode ---
        # Let's NOT move it back here. Let the worker handle moving it when needed.
        # if high_vram and transformer_on_cpu: # Only if we explicitly moved it
        #     try:
        #         print("Moving transformer back to GPU after LoRA change (High VRAM mode)...")
        #         transformer.to(gpu)
        #     except Exception as e:
        #         print(f"Warning: Failed to move transformer back to GPU in high VRAM mode: {e}")
        # --- End Optional Move Back ---
    else:
        print(f"No LoRA structure change needed. Current: '{currently_loaded_lora_info['adapter_name']}'")


# --- Function to Switch Active Model ---
def switch_active_model(target_model_display_name: str, progress=gr.Progress()):
    """Unloads current transformer, loads the target one, handles LoRA state."""
    global transformer, active_model_name, currently_loaded_lora_info

    if target_model_display_name == active_model_name:
        print(f"Model '{active_model_name}' is already active.")
        return active_model_name, f"Model '{active_model_name}' is already active."

    progress(0, desc=f"Switching model to '{target_model_display_name}'...")
    print(f"Switching model from '{active_model_name}' to '{target_model_display_name}'...")

    # 1. Unload any existing LoRA from the current transformer FIRST
    if currently_loaded_lora_info["adapter_name"] is not None:
        print(f"Unloading LoRA '{currently_loaded_lora_info['adapter_name']}' before switching model...")
        progress(0.1, desc=f"Unloading LoRA '{currently_loaded_lora_info['adapter_name']}'...")
        try:
            # Ensure current transformer is on CPU for safe unload
            if transformer.device != cpu:
                if not high_vram:
                    unload_complete_models(transformer) # Try dynamic swap unload first
                if transformer.device != cpu: # If still not on CPU (e.g., high VRAM)
                    transformer.to(cpu)
                    torch.cuda.empty_cache()

            unload_success = safe_unload_lora(transformer, cpu)
            if unload_success:
                print(f"Successfully unloaded LoRA '{currently_loaded_lora_info['adapter_name']}'.")
                currently_loaded_lora_info = {"adapter_name": None, "lora_path": None} # Reset LoRA state
            else:
                print(f"Warning: Failed to cleanly unload LoRA '{currently_loaded_lora_info['adapter_name']}'. Proceeding with model switch, but LoRA state might be inconsistent.")
                currently_loaded_lora_info = {"adapter_name": None, "lora_path": None} # Reset state anyway
        except Exception as e:
            print(f"Error unloading LoRA during model switch: {e}")
            traceback.print_exc()
            currently_loaded_lora_info = {"adapter_name": None, "lora_path": None} # Reset state on error

    # 2. Unload the current transformer model
    progress(0.3, desc=f"Unloading current model '{active_model_name}'...")
    print(f"Unloading current model '{active_model_name}'...")
    try:
        is_dynamic_swap = hasattr(transformer, '_hf_hook') and isinstance(transformer._hf_hook, DynamicSwapInstaller.SwapHook)
        if is_dynamic_swap:
            print("Uninstalling DynamicSwap from current transformer...")
            DynamicSwapInstaller.uninstall_model(transformer) # Remove hooks before deleting

        del transformer # Remove reference
        torch.cuda.empty_cache() # Clear VRAM
        print(f"Model '{active_model_name}' unloaded.")
    except Exception as e:
        print(f"Error during model unload: {e}")
        traceback.print_exc()
        # Continue, try to load the new one anyway

    # 3. Load the new transformer model
    new_model_hf_name = None
    if target_model_display_name == MODEL_DISPLAY_NAME_ORIGINAL:
        new_model_hf_name = MODEL_NAME_ORIGINAL
    elif target_model_display_name == MODEL_DISPLAY_NAME_F1:
        new_model_hf_name = MODEL_NAME_F1
    else:
        error_msg = f"Unknown target model name: {target_model_display_name}"
        print(f"ERROR: {error_msg}")
        # Try to reload the default as a fallback? Or just fail? Let's fail for clarity.
        # Re-assigning global `transformer` requires it to be defined.
        # This state is problematic. Best to raise error or return status indicating failure.
        # For Gradio yield, we return the original active_model_name and an error message.
        return active_model_name, f"Error: {error_msg}. Model not switched."

    progress(0.5, desc=f"Loading new model '{target_model_display_name}' from {new_model_hf_name}...")
    print(f"Loading new model '{target_model_display_name}' from {new_model_hf_name}...")
    try:
        transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(new_model_hf_name, torch_dtype=torch.bfloat16).cpu()
        transformer.eval()
        transformer.high_quality_fp32_output_for_inference = True
        transformer.to(dtype=torch.bfloat16) # Ensure dtype
        transformer.requires_grad_(False)
        print(f"New model '{target_model_display_name}' loaded to CPU.")

        # 4. Apply DynamicSwap if needed
        if not high_vram:
            progress(0.8, desc=f"Applying memory optimization...")
            print("Applying DynamicSwap to new transformer...")
            DynamicSwapInstaller.install_model(transformer, device=gpu)

        # 5. Update global state
        active_model_name = target_model_display_name
        progress(1.0, desc=f"Model switched successfully to '{active_model_name}'.")
        print(f"Model switched successfully to '{active_model_name}'.")
        # Refresh iteration info display after model switch
        # This requires access to the UI components - better to return the value and update UI separately
        # info_text = update_iteration_info(current_vid_len, current_fps, current_win_size) # Need current UI values
        return active_model_name, f"模型已成功切换 '{active_model_name}'."

    except Exception as e:
        error_msg = f"Failed to load model '{target_model_display_name}': {e}"
        print(f"ERROR: {error_msg}")
        traceback.print_exc()
        # Attempt to reload the previous model as a fallback? Risky.
        # Return the previous model name and error status.
        # Need to ensure `transformer` variable exists globally even if loading failed.
        # Let's try reloading the original default as a last resort.
        try:
             print("Attempting to reload default model as fallback...")
             default_hf_name = MODEL_NAME_ORIGINAL if DEFAULT_MODEL_NAME == MODEL_DISPLAY_NAME_ORIGINAL else MODEL_NAME_F1
             transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(default_hf_name, torch_dtype=torch.bfloat16).cpu()
             transformer.eval()
             transformer.high_quality_fp32_output_for_inference = True
             transformer.to(dtype=torch.bfloat16)
             transformer.requires_grad_(False)
             if not high_vram: DynamicSwapInstaller.install_model(transformer, device=gpu)
             active_model_name = DEFAULT_MODEL_NAME # Reset active name to default
             return active_model_name, f"Error: {error_msg}. Reverted to default model '{active_model_name}'."
        except Exception as fallback_e:
             fatal_error_msg = f"CRITICAL ERROR: Failed to load target model AND fallback model. Error: {fallback_e}"
             print(fatal_error_msg)
             # Application might be unusable now.
             # Returning original name but state is broken.
             return active_model_name, fatal_error_msg # Return original name, but signal critical failure


# --- Modify process function signature & add model switch check ---
def process(input_image, end_image, prompt, n_prompt, seed, use_random_seed, num_generations, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality='high', export_gif=False, export_apng=False, export_webp=False, save_metadata=True, resolution="640", fps=30, lora_scale=1.0, use_multiline_prompts=False, save_individual_frames=False, save_intermediate_frames=False, save_last_frame=False, rife_enabled=False, rife_multiplier="2x FPS",
             selected_lora_dropdown_value="None",
             # --- ADDED ---
             selected_model_display_name=DEFAULT_MODEL_NAME):

    global stream, currently_loaded_lora_info, active_model_name # Need global state access
    assert input_image is not None, 'No start input image!'

    # --- CHECK AND SWITCH MODEL (if necessary) ---
    # This should ideally happen via a dedicated UI interaction, but can be checked here too.
    # However, doing the potentially long switch here makes the 'Start' button slow.
    # Assume `switch_active_model` was called via UI change event before this.
    # We just need to use the current `active_model_name`.
    if selected_model_display_name != active_model_name:
         print(f"Warning: Selected model '{selected_model_display_name}' differs from active model '{active_model_name}'. Using the active model.")
         # Or trigger switch here? Let's rely on the UI event for switching.
    current_active_model = active_model_name # Use the globally set active model
    # --- END MODEL CHECK ---


    # --- MANAGE LORA STRUCTURE ---
    try:
        # Ensure transformer is on CPU before managing LoRA (manage_lora_structure handles this)
        manage_lora_structure(selected_lora_dropdown_value)
    except RuntimeError as e:
         print(f"LoRA Management Error: {e}")
         yield None, None, f"Error managing LoRA: {e}", '', gr.update(interactive=True), gr.update(interactive=False), seed, '' # Reset buttons
         return # Stop processing
    # --- END LORA MANAGEMENT ---

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), seed, ''

    if use_multiline_prompts and prompt.strip():
        prompt_lines = [line.strip() for line in prompt.split('\n')]
        prompt_lines = [line for line in prompt_lines if len(line) >= 2]
        if not prompt_lines: prompt_lines = [prompt.strip()]
        print(f"Multi-line enabled: Processing {len(prompt_lines)} prompts individually.")
    else:
        prompt_lines = [prompt.strip()]
        if not use_multiline_prompts: print("Multi-line disabled: Passing full prompt to worker for potential timestamp parsing.")
        else: print("Multi-line enabled, but prompt seems empty or invalid, using as single line.")

    total_prompts_or_loops = len(prompt_lines)
    final_video = None

    for prompt_idx, current_prompt_line in enumerate(prompt_lines):
        stream = AsyncStream()
        print(f"Starting processing loop {prompt_idx+1}/{total_prompts_or_loops} using model '{current_active_model}'")
        status_msg = f"Processing prompt {prompt_idx+1}/{total_prompts_or_loops} ({current_active_model})" if use_multiline_prompts else f"Starting generation ({current_active_model})"
        yield None, None, status_msg, '', gr.update(interactive=False), gr.update(interactive=True), seed, ''

        prompt_to_worker = prompt if not use_multiline_prompts else current_prompt_line

        # --- MODIFIED WORKER CALL (Add active_model) ---
        current_adapter_name = currently_loaded_lora_info["adapter_name"] if currently_loaded_lora_info["adapter_name"] else "None"
        async_run(worker, input_image, end_image, prompt_to_worker, n_prompt, seed, use_random_seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality, export_gif, export_apng, export_webp, num_generations, resolution, fps,
                  # --- LoRA Args Changed ---
                  adapter_name_to_use=current_adapter_name, # Pass name
                  lora_scale=lora_scale,                   # Keep scale
                  # --- End LoRA Args Changed ---
                  save_individual_frames_flag=save_individual_frames,
                  save_intermediate_frames_flag=save_intermediate_frames,
                  save_last_frame_flag=save_last_frame,
                  use_multiline_prompts_flag=use_multiline_prompts,
                  rife_enabled=rife_enabled, rife_multiplier=rife_multiplier,
                  # --- ADDED ---
                  active_model=current_active_model
                 )
        # --- END MODIFIED WORKER CALL ---

        output_filename = None
        webm_filename = None
        gif_filename = None
        apng_filename = None
        webp_filename = None
        rife_final_video_path = None # Track potential RIFE output
        current_seed_display = seed # Keep seed consistent for UI during multi-prompt
        timing_info = ""
        last_output = None

        while True:
            flag, data = stream.output_queue.next()

            if flag == 'seed_update':
                current_seed_display = data # Update UI seed display
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed_display, timing_info

            if flag == 'final_seed':
                 pass # Worker finished, final seed is known internally

            if flag == 'timing':
                gen_time = data['gen_time']
                avg_time = data['avg_time']
                remaining_time = data['remaining_time']
                eta_str = f"{remaining_time/60:.1f} minutes" if remaining_time > 60 else f"{remaining_time:.1f} seconds"
                timing_info = f"Last generation: {gen_time:.2f}s | Average: {avg_time:.2f}s | ETA: {eta_str}"
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed_display, timing_info

            if flag == 'final_timing':
                total_time = data['total_time']
                timing_info = f"Total generation time: {total_time:.2f}s ({total_time/60:.2f} min)"
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed_display, timing_info

            if flag == 'file':
                output_filename = data # This is the primary output (MP4 or WebM)
                if output_filename is None:
                    print("Warning: No primary output file was generated by worker")
                    yield None, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), current_seed_display, timing_info
                    continue # Wait for more messages or 'end'

                last_output = output_filename
                final_video = output_filename # Assume this is the final unless RIFE replaces it

                # Check for associated RIFE file (worker might have generated it)
                potential_rife_path = os.path.splitext(output_filename)[0] + '_extra_FPS.mp4'
                if rife_enabled and os.path.exists(potential_rife_path):
                    rife_final_video_path = potential_rife_path
                    final_video = rife_final_video_path # RIFE output becomes the final video
                    print(f"RIFE output detected: {rife_final_video_path}")

                # Determine which file to display
                display_video = final_video

                prompt_info = f" (Prompt {prompt_idx+1}/{total_prompts_or_loops}, {current_active_model})" if use_multiline_prompts else f" ({current_active_model})"
                yield display_video, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), current_seed_display, timing_info + prompt_info

            if flag == 'rife_file':
                 # Worker explicitly signals RIFE file completion
                 rife_video_file = data
                 print(f"Displaying RIFE-enhanced video: {rife_video_file}")
                 rife_final_video_path = rife_video_file
                 final_video = rife_video_file # Update final video
                 prompt_info = f" (Prompt {prompt_idx+1}/{total_prompts_or_loops}, {current_active_model})" if use_multiline_prompts else f" ({current_active_model})"
                 yield rife_video_file, gr.update(), gr.update(value=f"RIFE-enhanced video ready ({rife_multiplier})"), gr.update(), gr.update(interactive=False), gr.update(interactive=True), current_seed_display, timing_info + prompt_info

            if flag == 'progress':
                preview, desc, html = data
                # Add model and prompt info to progress
                model_prompt_info = f" ({current_active_model}"
                if use_multiline_prompts:
                     model_prompt_info += f", Prompt {prompt_idx+1}/{total_prompts_or_loops}"
                model_prompt_info += ")"

                if html:
                    import re
                    hint_match = re.search(r'>(.*?)<br', html) # Find text before <br/>
                    if hint_match:
                        hint = hint_match.group(1)
                        new_hint = f"{hint}{model_prompt_info}"
                        # Escape potential special characters in hint for regex replacement
                        escaped_hint = re.escape(hint)
                        html = re.sub(f">{escaped_hint}<br", f">{new_hint}<br", html, count=1)
                    else: # Fallback if structure changes
                         html += f"<span>{model_prompt_info}</span>"

                if desc:
                    desc += model_prompt_info

                yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), current_seed_display, timing_info

            if flag == 'end':
                # Determine final video to display after loop ends
                display_video = final_video if final_video else None

                if prompt_idx == len(prompt_lines) - 1: # Last prompt processed
                    yield display_video, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False), current_seed_display, timing_info
                else: # More prompts to go
                    yield display_video, gr.update(visible=False), f"Completed prompt {prompt_idx+1}/{total_prompts_or_loops} ({current_active_model})", '', gr.update(interactive=False), gr.update(interactive=True), current_seed_display, timing_info
                break # Exit the inner while loop, proceed to next prompt_line or finish

        # Update seed for the next prompt line if not using random seeds
        if not use_random_seed and prompt_idx < len(prompt_lines) - 1:
             seed += 1 # Increment seed for next line's generation(s)


        if not use_multiline_prompts:
            break # Only run once if multi-line is off

    # Final yield after all prompts are done
    display_video = final_video if final_video else None
    yield display_video, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False), current_seed_display, timing_info


# --- Modify batch_process function signature & add model switch check ---
def batch_process(input_folder, output_folder, batch_end_frame_folder, prompt, n_prompt, seed, use_random_seed, total_second_length,
                   latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold,
                   video_quality='high', export_gif=False, export_apng=False, export_webp=False,
                   skip_existing=True, save_metadata=True, num_generations=1, resolution="640", fps=30,
                   lora_scale=1.0, batch_use_multiline_prompts=False,
                   batch_save_individual_frames=False, batch_save_intermediate_frames=False, batch_save_last_frame=False,
                   rife_enabled=False, rife_multiplier="2x FPS",
                   selected_lora_dropdown_value="None",
                   # --- ADDED ---
                   selected_model_display_name=DEFAULT_MODEL_NAME):

    global stream, batch_stop_requested, currently_loaded_lora_info, active_model_name # Need global state access

    print("Resetting batch stop flag.")
    batch_stop_requested = False

    # --- CHECK AND SWITCH MODEL (if necessary) ---
    # Assume model switch happened via UI before starting batch
    if selected_model_display_name != active_model_name:
         print(f"Warning: Selected batch model '{selected_model_display_name}' differs from active model '{active_model_name}'. Using the active model for the entire batch.")
    current_active_model = active_model_name # Use the globally set active model for the whole batch
    # --- END MODEL CHECK ---

    # --- MANAGE LORA STRUCTURE (ONCE FOR BATCH) ---
    try:
        # Ensure transformer is on CPU before managing LoRA
        manage_lora_structure(selected_lora_dropdown_value)
    except RuntimeError as e:
         print(f"LoRA Management Error during batch setup: {e}")
         yield None, None, f"Error managing LoRA: {e}", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""
         return
    # --- END LORA MANAGEMENT ---


    if not input_folder or not os.path.exists(input_folder):
        return None, f"Input folder does not exist: {input_folder}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""
    if not output_folder:
        output_folder = outputs_batch_folder
    else:
        try: os.makedirs(output_folder, exist_ok=True)
        except Exception as e: return None, f"Error creating output folder: {str(e)}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""

    use_end_frames = batch_end_frame_folder and os.path.isdir(batch_end_frame_folder)
    if batch_end_frame_folder and not use_end_frames:
         print(f"Warning: End frame folder provided but not found or not a directory: {batch_end_frame_folder}. Proceeding without end frames.")

    image_files = get_images_from_folder(input_folder) # Uses natsorted now
    if not image_files:
        return None, f"No image files found in {input_folder}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""

    yield None, None, f"Found {len(image_files)} images to process using model '{current_active_model}'. End frames {'enabled' if use_end_frames else 'disabled'}.", "", gr.update(interactive=False), gr.update(interactive=True), seed, ""

    final_output = None
    current_batch_seed = seed # Use a separate seed tracker for the batch
    current_adapter_name = currently_loaded_lora_info["adapter_name"] if currently_loaded_lora_info["adapter_name"] else "None"


    # --- OUTER BATCH LOOP ---
    for idx, image_path in enumerate(image_files):
        if batch_stop_requested:
            print("Batch stop requested. Exiting batch process.")
            yield final_output, gr.update(visible=False), "Batch processing stopped by user.", "", gr.update(interactive=True), gr.update(interactive=False), current_batch_seed, ""
            return

        start_image_basename = os.path.basename(image_path)
        output_filename_base = os.path.splitext(start_image_basename)[0]

        current_prompt_text = prompt # Default prompt
        custom_prompt = get_prompt_from_txt_file(image_path)
        if custom_prompt:
            current_prompt_text = custom_prompt
            print(f"Using custom prompt from txt file for {image_path}")
        else:
            print(f"Using default batch prompt for {image_path}")


        if batch_use_multiline_prompts:
            potential_lines = current_prompt_text.split('\n')
            prompt_lines_or_fulltext = [line.strip() for line in potential_lines if line.strip()]
            prompt_lines_or_fulltext = [line for line in prompt_lines_or_fulltext if len(line) >= 2]
            if not prompt_lines_or_fulltext: prompt_lines_or_fulltext = [current_prompt_text.strip()]
            print(f"Batch multi-line enabled: Processing {len(prompt_lines_or_fulltext)} prompts for {start_image_basename}")
        else:
            prompt_lines_or_fulltext = [current_prompt_text.strip()]
            print(f"Batch multi-line disabled: Passing full prompt text to worker for {start_image_basename}")

        total_prompts_or_loops = len(prompt_lines_or_fulltext)

        # --- Skip Check Logic ---
        skip_this_image = False
        if skip_existing:
            # Check for the first potential output file's existence
            # Needs to account for multi-prompt and multi-generation naming
            first_prompt_idx = 0
            first_gen_idx = 1 # Generations are 1-based in naming
            output_check_suffix = ""
            if batch_use_multiline_prompts:
                 output_check_suffix += f"_p{first_prompt_idx+1}"
            if num_generations > 1:
                 output_check_suffix += f"_g{first_gen_idx}"
            elif not batch_use_multiline_prompts and num_generations > 1 : # Original naming for single prompt multi-gen
                 output_check_suffix += f"_{first_gen_idx}"

            # Check for MP4, WebM (if web_compatible), and maybe RIFE output
            output_check_mp4 = os.path.join(output_folder, f"{output_filename_base}{output_check_suffix}.mp4")
            output_check_webm = os.path.join(output_folder, f"{output_filename_base}{output_check_suffix}.webm")
            output_check_rife = os.path.join(output_folder, f"{output_filename_base}{output_check_suffix}_extra_FPS.mp4")

            exists_mp4 = os.path.exists(output_check_mp4)
            exists_webm = video_quality == 'web_compatible' and os.path.exists(output_check_webm)
            exists_rife = rife_enabled and os.path.exists(output_check_rife)

            # Skip if the expected primary output exists (or RIFE if enabled)
            if rife_enabled:
                 skip_this_image = exists_rife or exists_mp4 # Skip if RIFE exists, or base MP4 if RIFE failed/not run yet
            elif video_quality == 'web_compatible':
                 skip_this_image = exists_webm or exists_mp4 # Skip if WebM exists, or base MP4 otherwise
            else:
                 skip_this_image = exists_mp4 # Skip if MP4 exists

        if skip_this_image:
            print(f"Skipping {image_path} - output already exists")
            yield None, None, f"Skipping {idx+1}/{len(image_files)}: {start_image_basename} - already processed", "", gr.update(interactive=False), gr.update(interactive=True), current_batch_seed, ""
            continue
        # --- End Skip Check ---


        try:
            img = Image.open(image_path)
            if img.mode != 'RGB': img = img.convert('RGB')
            input_image = np.array(img)
            if len(input_image.shape) == 2: input_image = np.stack([input_image]*3, axis=2)
            print(f"Loaded start image {image_path} with shape {input_image.shape} and dtype {input_image.dtype}")
        except Exception as e:
            print(f"Error loading start image {image_path}: {str(e)}")
            yield None, None, f"Error processing {idx+1}/{len(image_files)}: {start_image_basename} - {str(e)}", "", gr.update(interactive=False), gr.update(interactive=True), current_batch_seed, ""
            continue

        current_end_image = None
        end_image_path_str = "None"
        if use_end_frames:
            # Try matching filename exactly in the end frame folder
            potential_end_path = os.path.join(batch_end_frame_folder, start_image_basename)
            if os.path.exists(potential_end_path):
                try:
                    end_img = Image.open(potential_end_path)
                    if end_img.mode != 'RGB': end_img = end_img.convert('RGB')
                    current_end_image = np.array(end_img)
                    if len(current_end_image.shape) == 2: current_end_image = np.stack([current_end_image]*3, axis=2)
                    print(f"Loaded matching end image: {potential_end_path}")
                    end_image_path_str = potential_end_path
                except Exception as e:
                    print(f"Error loading end image {potential_end_path}: {str(e)}. Processing without end frame.")
                    current_end_image = None # Ensure it's None on error
            else:
                 print(f"No matching end frame found for {start_image_basename} in {batch_end_frame_folder}")


        # --- INNER PROMPT LOOP (if multi-line enabled for batch) ---
        for prompt_idx, current_prompt_segment in enumerate(prompt_lines_or_fulltext):
            if batch_stop_requested:
                print("Batch stop requested during prompt loop. Exiting batch process.")
                yield final_output, gr.update(visible=False), "Batch processing stopped by user.", "", gr.update(interactive=True), gr.update(interactive=False), current_batch_seed, ""
                return

            # Handle seed for this image/prompt combination
            seed_for_this_prompt = current_batch_seed
            if use_random_seed:
                # Use a new random seed for each image/prompt combo if random is checked
                seed_for_this_prompt = random.randint(1, 2147483647)
            elif idx > 0 or prompt_idx > 0:
                # Increment seed from the previous image/prompt unless it's the very first one
                 seed_for_this_prompt = current_batch_seed + (idx * total_prompts_or_loops) + prompt_idx

            current_batch_seed_display = seed_for_this_prompt # Use this for UI display and worker

            prompt_info = ""
            if batch_use_multiline_prompts:
                prompt_info = f" (Prompt {prompt_idx+1}/{total_prompts_or_loops}: {current_prompt_segment[:30]}{'...' if len(current_prompt_segment) > 30 else ''})"
            elif not batch_use_multiline_prompts and custom_prompt:
                 prompt_info = " (Using .txt prompt - potential timestamps)"
            elif not batch_use_multiline_prompts and not custom_prompt:
                 prompt_info = " (Using default prompt - potential timestamps)"


            yield None, None, f"Processing {idx+1}/{len(image_files)}: {start_image_basename} ({current_active_model}, End: {os.path.basename(end_image_path_str) if current_end_image is not None else 'No'}) with {num_generations} generation(s){prompt_info}", "", gr.update(interactive=False), gr.update(interactive=True), current_batch_seed_display, ""


            gen_start_time_batch = time.time() # Track start time for metadata for this image/prompt

            stream = AsyncStream()

            # --- Setup batch-specific output folders for frames ---
            # This part seems complex to override globally. The worker uses global paths.
            # For batch, let's save frames relative to the batch output folder instead of the main outputs.
            # We need to temporarily modify the global paths within the worker call.

            batch_individual_frames_folder_abs = None
            batch_intermediate_individual_frames_folder_abs = None
            batch_last_frames_folder_abs = None
            batch_intermediate_last_frames_folder_abs = None

            override_paths_needed = batch_save_individual_frames or batch_save_intermediate_frames or batch_save_last_frame
            if override_paths_needed:
                batch_base_frame_folder = os.path.join(output_folder, "frames_output") # Central folder for frames in batch output
                batch_individual_frames_folder_abs = os.path.join(batch_base_frame_folder, 'individual_frames')
                batch_intermediate_individual_frames_folder_abs = os.path.join(batch_individual_frames_folder_abs, 'intermediate_videos')
                batch_last_frames_folder_abs = os.path.join(batch_base_frame_folder, 'last_frames')
                batch_intermediate_last_frames_folder_abs = os.path.join(batch_last_frames_folder_abs, 'intermediate_videos')
                try: # Create these folders within the batch output directory
                    os.makedirs(batch_individual_frames_folder_abs, exist_ok=True)
                    os.makedirs(batch_intermediate_individual_frames_folder_abs, exist_ok=True)
                    os.makedirs(batch_last_frames_folder_abs, exist_ok=True)
                    os.makedirs(batch_intermediate_last_frames_folder_abs, exist_ok=True)
                    print(f"Created/Ensured batch frame output folders in: {batch_base_frame_folder}")
                except Exception as e:
                     print(f"Error creating batch frame folders: {e}")
                     override_paths_needed = False # Disable override if folders can't be made

            # Wrapper function to temporarily change global paths for the worker
            def batch_worker_path_override(*args, **kwargs):
                global individual_frames_folder, intermediate_individual_frames_folder, last_frames_folder, intermediate_last_frames_folder
                # Store original global paths
                orig_individual_frames = individual_frames_folder
                orig_intermediate_individual_frames = intermediate_individual_frames_folder
                orig_last_frames = last_frames_folder
                orig_intermediate_last_frames = intermediate_last_frames_folder

                # Override with batch-specific absolute paths
                individual_frames_folder = batch_individual_frames_folder_abs
                intermediate_individual_frames_folder = batch_intermediate_individual_frames_folder_abs
                last_frames_folder = batch_last_frames_folder_abs
                intermediate_last_frames_folder = batch_intermediate_last_frames_folder_abs
                print(f"Worker override: Using batch frame paths rooted at {output_folder}")
                try:
                    # Call the original worker function with all arguments
                    result = worker(*args, **kwargs)
                    return result
                finally:
                    # Restore original global paths crucial!
                    individual_frames_folder = orig_individual_frames
                    intermediate_individual_frames_folder = orig_intermediate_individual_frames
                    last_frames_folder = orig_last_frames
                    intermediate_last_frames_folder = orig_intermediate_last_frames
                    print("Worker override: Restored global frame paths.")

            worker_function_to_call = batch_worker_path_override if override_paths_needed else worker
            # --- End path override setup ---


            # --- MODIFIED WORKER CALL (BATCH - Add active_model) ---
            async_run(worker_function_to_call,
                    input_image, current_end_image, current_prompt_segment, n_prompt,
                    current_batch_seed_display, # Use the seed calculated for this specific task
                    False, # Use fixed seed for each generation within the same image/prompt
                    total_second_length, latent_window_size, steps, cfg, gs, rs,
                    gpu_memory_preservation, teacache_threshold, video_quality, export_gif,
                    export_apng, export_webp,
                    num_generations=num_generations, # Pass the per-image generation count
                    resolution=resolution, fps=fps,
                    # --- LoRA Args Changed ---
                    adapter_name_to_use=current_adapter_name, # Pass name
                    lora_scale=lora_scale,                   # Keep scale
                    # --- End LoRA Args Changed ---
                    save_individual_frames_flag=batch_save_individual_frames,
                    save_intermediate_frames_flag=batch_save_intermediate_frames,
                    save_last_frame_flag=batch_save_last_frame,
                    # Pass batch multi-line flag to worker to potentially adjust timestamp parsing logic if needed (though it currently doesn't)
                    use_multiline_prompts_flag=batch_use_multiline_prompts,
                    rife_enabled=rife_enabled, rife_multiplier=rife_multiplier,
                    # --- ADDED ---
                    active_model=current_active_model
                   )
            # --- END MODIFIED WORKER CALL (BATCH) ---


            output_filename_from_worker = None
            last_output_for_ui = None
            all_outputs_for_image = {} # Track outputs for this specific image/prompt
            generation_count_this_prompt = 0

            # --- WORKER LISTENING LOOP ---
            while True:
                if batch_stop_requested:
                     print("Batch stop requested while waiting for worker. Ending loop.")
                     if stream: stream.input_queue.push('end') # Signal worker to stop if running
                     break
                flag, data = stream.output_queue.next()

                if flag is None: continue # Should not happen with Queue.get()

                if flag == 'seed_update':
                    # Worker updates seed internally for its generations. Update UI display.
                    current_batch_seed_display = data
                    yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_batch_seed_display, gr.update()

                if flag == 'final_seed':
                     # Worker finished its generations for this image/prompt.
                     # The seed might have incremented if num_generations > 1.
                     # We need this final seed if we continue to the next image/prompt non-randomly.
                     # current_batch_seed = data # Update the main batch seed tracker ONLY if not random
                     pass # Let the outer loop handle seed incrementing based on idx/prompt_idx

                if flag == 'file':
                    output_filename_from_worker = data # MP4 or WebM from worker's *final* step for a generation
                    if output_filename_from_worker:
                        is_intermediate = 'intermediate' in output_filename_from_worker # Check if it's an intermediate save from worker loop
                        if not is_intermediate:
                            generation_count_this_prompt += 1

                            # --- Construct final output filename for batch ---
                            output_file_suffix = ""
                            if batch_use_multiline_prompts:
                                output_file_suffix += f"_p{prompt_idx+1}"
                            if num_generations > 1:
                                # Use the per-prompt generation count
                                output_file_suffix += f"_g{generation_count_this_prompt}"
                            elif not batch_use_multiline_prompts and num_generations > 1: # Original naming convention
                                output_file_suffix += f"_{generation_count_this_prompt}"

                            final_output_filename_base = f"{output_filename_base}{output_file_suffix}"
                            # --- End filename construction ---

                            # Move/Copy the primary file (MP4 or WebM)
                            moved_primary_file = move_and_rename_output_file(
                                output_filename_from_worker,
                                output_folder,
                                f"{final_output_filename_base}{os.path.splitext(output_filename_from_worker)[1]}" # Keep original extension
                            )

                            if moved_primary_file:
                                output_key = f'primary_{generation_count_this_prompt}' if num_generations > 1 else 'primary'
                                all_outputs_for_image[output_key] = moved_primary_file
                                last_output_for_ui = moved_primary_file # Update UI with the latest primary file
                                final_output = moved_primary_file # Track the very last primary file generated

                                prompt_status = f" (Prompt {prompt_idx+1}/{total_prompts_or_loops})" if batch_use_multiline_prompts else ""
                                yield last_output_for_ui, gr.update(visible=False), f"Processing {idx+1}/{len(image_files)}: {start_image_basename} ({current_active_model}) - Generated video {generation_count_this_prompt}/{num_generations}{prompt_status}", "", gr.update(interactive=False), gr.update(interactive=True), current_batch_seed_display, gr.update()

                                # --- Metadata Saving for Batch ---
                                if save_metadata:
                                    gen_time_batch_current = time.time() - gen_start_time_batch
                                    generation_time_seconds = int(gen_time_batch_current)
                                    generation_time_formatted = format_time_human_readable(gen_time_batch_current)
                                    metadata = {
                                        "Model": current_active_model, # <-- ADDED MODEL NAME
                                        "Prompt": current_prompt_segment, # Use the specific prompt segment
                                        "Negative Prompt": n_prompt,
                                        # Use the seed worker actually used for this generation
                                        "Seed": current_batch_seed_display + (generation_count_this_prompt - 1) if not use_random_seed else "Random", # Estimate seed if fixed
                                        "TeaCache": f"Enabled (Threshold: {teacache_threshold})" if teacache_threshold > 0.0 else "Disabled",
                                        "Video Length (seconds)": total_second_length,
                                        "FPS": fps,
                                        "Latent Window Size": latent_window_size,
                                        "Steps": steps,
                                        "CFG Scale": cfg,
                                        "Distilled CFG Scale": gs,
                                        "Guidance Rescale": rs,
                                        "Resolution": resolution,
                                        "Generation Time": generation_time_formatted,
                                        "Total Seconds": f"{generation_time_seconds} seconds",
                                        "Start Frame": image_path,
                                        "End Frame": end_image_path_str if current_end_image is not None else "None",
                                        "Multi-line Prompts Mode": batch_use_multiline_prompts,
                                        "Generation Index": f"{generation_count_this_prompt}/{num_generations}",
                                    }
                                    if not batch_use_multiline_prompts:
                                         metadata["Timestamped Prompts Parsed"] = "[Check Worker Logs]" # Indicate worker handled it
                                    if batch_use_multiline_prompts:
                                        metadata["Prompt Number"] = f"{prompt_idx+1}/{total_prompts_or_loops}"

                                    if current_adapter_name != "None":
                                        metadata["LoRA"] = current_adapter_name
                                        metadata["LoRA Scale"] = lora_scale

                                    save_processing_metadata(moved_primary_file, metadata.copy()) # Save for primary file

                                    # --- Save Metadata for Other Formats ---
                                    worker_output_base = os.path.splitext(output_filename_from_worker)[0]
                                    target_output_base = os.path.join(output_folder, final_output_filename_base)

                                    # GIF
                                    gif_original_path = f"{worker_output_base}.gif"
                                    if export_gif and os.path.exists(gif_original_path):
                                        moved_gif = move_and_rename_output_file(gif_original_path, output_folder, f"{final_output_filename_base}.gif")
                                        if moved_gif: save_processing_metadata(moved_gif, metadata.copy())
                                    # APNG
                                    apng_original_path = f"{worker_output_base}.png"
                                    if export_apng and os.path.exists(apng_original_path):
                                         moved_apng = move_and_rename_output_file(apng_original_path, output_folder, f"{final_output_filename_base}.png")
                                         if moved_apng: save_processing_metadata(moved_apng, metadata.copy())
                                    # WebP
                                    webp_original_path = f"{worker_output_base}.webp"
                                    if export_webp and os.path.exists(webp_original_path):
                                         moved_webp = move_and_rename_output_file(webp_original_path, output_folder, f"{final_output_filename_base}.webp")
                                         if moved_webp: save_processing_metadata(moved_webp, metadata.copy())
                                    # WebM (if not primary)
                                    webm_original_path = f"{worker_output_base}.webm"
                                    if video_quality == 'web_compatible' and moved_primary_file != webm_original_path and os.path.exists(webm_original_path):
                                         moved_webm = move_and_rename_output_file(webm_original_path, output_folder, f"{final_output_filename_base}.webm")
                                         if moved_webm: save_processing_metadata(moved_webm, metadata.copy())
                                    # RIFE
                                    rife_original_path = f"{worker_output_base}_extra_FPS.mp4"
                                    if rife_enabled and os.path.exists(rife_original_path):
                                        rife_target_path = f"{target_output_base}_extra_FPS.mp4"
                                        try:
                                            shutil.copy2(rife_original_path, rife_target_path)
                                            print(f"Copied RIFE enhanced file to batch outputs: {rife_target_path}")
                                            save_processing_metadata(rife_target_path, metadata.copy())
                                            # Update UI only if RIFE is the intended final output
                                            last_output_for_ui = rife_target_path
                                            final_output = rife_target_path
                                        except Exception as e:
                                            print(f"Error copying RIFE enhanced file to {rife_target_path}: {str(e)}")
                                # --- End Metadata Saving for Other Formats ---
                            else:
                                 print(f"ERROR: Failed to move/rename output file {output_filename_from_worker}")
                                 # Keep UI showing previous success if any
                                 yield last_output_for_ui, gr.update(visible=False), f"Error saving output for {idx+1}/{len(image_files)}: {start_image_basename}", "", gr.update(interactive=False), gr.update(interactive=True), current_batch_seed_display, gr.update()

                        else: # Intermediate file from worker - display it but don't move/save metadata yet
                            prompt_status = f" (Prompt {prompt_idx+1}/{total_prompts_or_loops})" if batch_use_multiline_prompts else ""
                            yield output_filename_from_worker, gr.update(visible=False), f"Processing {idx+1}/{len(image_files)}: {start_image_basename} ({current_active_model}) - Generating intermediate result{prompt_status}", "", gr.update(interactive=False), gr.update(interactive=True), current_batch_seed_display, gr.update()

                if flag == 'progress':
                    preview, desc, html = data
                    # Add batch progress info
                    batch_prog_info = f"Processing {idx+1}/{len(image_files)}: {start_image_basename} ({current_active_model}"
                    if batch_use_multiline_prompts:
                        batch_prog_info += f", Prompt {prompt_idx+1}/{total_prompts_or_loops}"
                    batch_prog_info += ")"

                    current_progress_desc = f"{batch_prog_info} - {desc}" if desc else batch_prog_info
                    progress_html = html if html else make_progress_bar_html(0, batch_prog_info)

                    # Add batch info to HTML hint
                    if html:
                         import re
                         hint_match = re.search(r'>(.*?)<br', html)
                         if hint_match:
                              hint = hint_match.group(1)
                              new_hint = f"{hint} [{idx+1}/{len(image_files)}]"
                              if batch_use_multiline_prompts: new_hint += f"[P{prompt_idx+1}]"
                              escaped_hint = re.escape(hint)
                              progress_html = re.sub(f">{escaped_hint}<br", f">{new_hint}<br", html, count=1)
                         else:
                             progress_html += f"<span>{batch_prog_info}</span>"


                    video_update = last_output_for_ui if last_output_for_ui else gr.update()
                    yield video_update, gr.update(visible=True, value=preview), current_progress_desc, progress_html, gr.update(interactive=False), gr.update(interactive=True), current_batch_seed_display, gr.update()

                if flag == 'end':
                    # Worker finished for this image/prompt/(all generations)
                    video_update = last_output_for_ui if last_output_for_ui else gr.update()
                    if prompt_idx == len(prompt_lines_or_fulltext) - 1: # Last prompt for this image
                        yield video_update, gr.update(visible=False), f"Completed {idx+1}/{len(image_files)}: {start_image_basename} ({current_active_model})", "", gr.update(interactive=False), gr.update(interactive=True), current_batch_seed_display, gr.update()
                    else: # More prompts for this image
                        prompt_status = f" (Completed prompt {prompt_idx+1}/{total_prompts_or_loops}, continuing to next prompt)"
                        yield video_update, gr.update(visible=False), f"Processing {idx+1}/{len(image_files)}: {start_image_basename}{prompt_status}", "", gr.update(interactive=False), gr.update(interactive=True), current_batch_seed_display, gr.update()
                    break # End listening loop for this worker instance
            # --- END WORKER LISTENING LOOP ---

            if batch_stop_requested:
                print("Batch stop requested after worker finished. Exiting batch process.")
                yield final_output, gr.update(visible=False), "Batch processing stopped by user.", "", gr.update(interactive=True), gr.update(interactive=False), current_batch_seed_display, ""
                return

            # Increment seed for next prompt if not random (handled inside loop now)

            if not batch_use_multiline_prompts:
                 break # Only one loop if not using multi-line
        # --- END INNER PROMPT LOOP ---

        # Increment seed for the next image if not random
        # Base seed for next image = initial seed + num images processed so far
        # This ensures restarting batch uses predictable seeds if initial seed is the same.
        if not use_random_seed:
             current_batch_seed = seed + (idx + 1) # Increment base seed for next image

        if batch_stop_requested:
            print("Batch stop requested after inner loop. Exiting batch process.")
            yield final_output, gr.update(visible=False), "Batch processing stopped by user.", "", gr.update(interactive=True), gr.update(interactive=False), current_batch_seed_display, ""
            return
    # --- END OUTER BATCH LOOP ---

    if not batch_stop_requested:
        yield final_output, gr.update(visible=False), f"Batch processing complete. Processed {len(image_files)} images using {current_active_model}.", "", gr.update(interactive=True), gr.update(interactive=False), current_batch_seed_display, ""
    else:
         yield final_output, gr.update(visible=False), "Batch processing stopped by user.", "", gr.update(interactive=True), gr.update(interactive=False), current_batch_seed_display, ""


# --- end_process remains unchanged ---
def end_process():
    global batch_stop_requested # Declare intent to modify global flag
    print("\nSending end generation signal...")
    if 'stream' in globals() and stream:
        stream.input_queue.push('end')
        print("End signal sent to current worker.")
    else:
        print("Stream not initialized, cannot send end signal to worker.")

    print("Setting batch stop flag...")
    batch_stop_requested = True # Set the global flag

    # Update buttons immediately
    updates = [gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)]
    return updates


quick_prompts = [
    'A character doing some simple body movements.','A talking man.',
    '[0] A person stands still\n[2] The person waves hello\n[4] The person claps',
    '[0] close up shot, cinematic\n[3] medium shot, cinematic\n[5] wide angle shot, cinematic'
]
quick_prompts = [[x] for x in quick_prompts]

# --- Function to auto-set Latent Window Size --- (Unchanged)
def auto_set_window_size(fps_val: int, current_lws: int):
    """Calculates Latent Window Size for ~1 second sections."""
    if not isinstance(fps_val, int) or fps_val <= 0:
        print("Invalid FPS for auto window size calculation.")
        return gr.update() # No change if FPS is invalid

    try:
        ideal_lws_float = fps_val / 4.0
        target_lws = round(ideal_lws_float)
        min_lws = 1
        max_lws = 33
        calculated_lws = max(min_lws, min(target_lws, max_lws))
        resulting_duration = (calculated_lws * 4) / fps_val

        print(f"Auto-setting LWS: Ideal float LWS for 1s sections={ideal_lws_float:.2f}, Rounded integer LWS={target_lws}, Clamped LWS={calculated_lws}")
        print(f"--> Resulting section duration with LWS={calculated_lws} at {fps_val} FPS will be: {resulting_duration:.3f} seconds")

        if abs(resulting_duration - 1.0) < 0.01: print("This setting provides (near) exact 1-second sections.")
        else: print(f"Note: This is the closest integer LWS to achieve 1-second sections.")

        if calculated_lws != current_lws: return gr.update(value=calculated_lws)
        else:
            print(f"Latent Window Size is already optimal ({current_lws}) for ~1s sections.")
            return gr.update()

    except Exception as e:
        print(f"Error calculating auto window size: {e}")
        traceback.print_exc()
        return gr.update()
# --- End Auto-set Function ---


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# 匹夫 - 微信:AI-pifu') # Updated Title
    with gr.Row():
        # --- Model Selector ---
        model_selector = gr.Radio(
            label="选择FramePack",
            choices=[MODEL_DISPLAY_NAME_ORIGINAL, MODEL_DISPLAY_NAME_F1],
            value=active_model_name, # Use the globally loaded default
            info="选择生成模型。切换模型后将加载新的模型"
        )
        model_status = gr.Markdown(f"Active model: **{active_model_name}**")
        # --- End Model Selector ---
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("加载图像"): # Renamed slightly
                    with gr.Row():
                        with gr.Column():
                            input_image = gr.Image(sources='upload', type="numpy", label="起始帧", height=320)
                        with gr.Column():
                            # End frame might be less effective with F1 but keep for Original
                            end_image = gr.Image(sources='upload', type="numpy", label="尾帧 (只能在原始版下使用)", height=320)

                    with gr.Row():
                        iteration_info_display = gr.Markdown("Calculating generation info...", elem_id="iteration-info-display")
                        auto_set_lws_button = gr.Button(value="设置1秒窗口（只支持原始版）", scale=1) # Clarified button purpose

                    prompt = gr.Textbox(label="提示词仅当“使用多行提示符”处于关闭状态时，才在新行上使用“[秒数] 提示符”格式。例如，[0] 从第 0 秒开始，[2] 在 2 秒后开始，依此类推。适用两种版本。", value='', lines=4, info="")
                    with gr.Row():
                        use_multiline_prompts = gr.Checkbox(label="使用多行提示", value=False, info="开启：每行单独生成一个种子（如果是随机的，则每行使用新的种子）。 关闭：尝试解析“[secs] prompt”格式。")
                        # Latent Window Size - Keep, might affect F1 loop length/speed balance too
                        latent_window_size = gr.Slider(label="潜在窗口大小", minimum=1, maximum=33, value=9, step=1, visible=True, info="潜在窗口大小控制生成块的大小。影响段/循环计数和持续时间，建议默认值为 9。.")
                    example_quick_prompts = gr.Dataset(samples=quick_prompts, label='添加预设关键字', samples_per_page=1000, components=[prompt])
                    example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

                    with gr.Row():
                        save_metadata = gr.Checkbox(label="保存处理元数据", value=True, info="将处理参数与每个视频一起保存在文本文件中")
                        save_individual_frames = gr.Checkbox(label="保存单个帧", value=False, info="将最终视频的每一帧保存为单独的图像")
                        save_intermediate_frames = gr.Checkbox(label="保存中间帧", value=False, info="将中间视频的每一帧保存为单独的图像")
                        save_last_frame = gr.Checkbox(label="保存最后一帧", value=False, info="将生成的最后一帧保存到 last_frames 文件夹")

                    with gr.Row():
                        start_button = gr.Button(value="开始生成", variant='primary')
                        end_button = gr.Button(value="终止生成", interactive=False)

                with gr.Tab("批量处理"):
                    batch_input_folder = gr.Textbox(label="输入文件夹路径（起始帧）", info="包含要处理的起始图像的文件夹（自然排序）")
                    batch_end_frame_folder = gr.Textbox(label="结束帧文件夹路径（可选，使用原始模型）", info="包含匹配的结束帧（与起始帧相同的文件名）的文件夹")
                    batch_output_folder = gr.Textbox(label="输出文件夹路径（可选）", info="留空以使用默认的 batch_outputs 文件夹")
                    batch_prompt = gr.Textbox(label="默认提示词", value='', lines=4, info="如果不存在匹配的 .txt 文件，则使用此选项。仅当“使用多行提示符”关闭时，才在新行上使用“[秒数] 提示符”格式。")

                    with gr.Row():
                        batch_skip_existing = gr.Checkbox(label="跳过现有文件", value=True, info="跳过第一个预期输出视频已经存在的图像")
                        batch_save_metadata = gr.Checkbox(label="保存处理元数据", value=True, info="将处理参数与每个视频一起保存在文本文件中")
                        batch_use_multiline_prompts = gr.Checkbox(label="使用多行提示词", value=False, info="开启：prompt/.txt 中的每一行都是一个单独的 gen。关闭：尝试从完整的 prompt/.txt 解析“[secs] prompt”格式。")

                    with gr.Row():
                        batch_save_individual_frames = gr.Checkbox(label="保存单个帧", value=False, info="将最终视频的每一帧保存为单独的图像（在 batch_output/frames_output 中）")
                        batch_save_intermediate_frames = gr.Checkbox(label="保存中间帧", value=False, info="将中间视频的每一帧保存为单独的图像（在 batch_output/frames_output 中）")
                        batch_save_last_frame = gr.Checkbox(label="保存最后一帧（仅限 MP4）", value=False, info="仅保存每个 MP4 生成的最后一帧（在 batch_output/frames_output 中）")

                    with gr.Row():
                        batch_start_button = gr.Button(value="开始批处理", variant='primary')
                        batch_end_button = gr.Button(value="结束处理", interactive=False)

                    with gr.Row():
                        open_batch_input_folder = gr.Button(value="打开 开始 文件夹")
                        open_batch_end_folder = gr.Button(value="打开 结束  文件夹")
                        open_batch_output_folder = gr.Button(value="打开输出文件夹")


            with gr.Group("Common Settings"): # Group shared settings
                with gr.Row():
                    num_generations = gr.Slider(label="生成数量", minimum=1, maximum=50, value=1, step=1, info="按顺序生成多个视频（每个图像/提示）")
                    resolution = gr.Dropdown(label="分辨率", choices=["1440","1320","1200","1080","960","840","720", "640", "480", "320", "240"], value="640", info="输出分辨率（大于640设置更多保留内存）")

                with gr.Row():
                    # TeaCache - F1 Demo default is 0.15 (True). Let's use slider like original.
                    teacache_threshold = gr.Slider(label='TeaCache 阈值', minimum=0.0, maximum=0.5, value=0.15, step=0.01, info='0 = 禁用，>0 = 启用。值越高，缓存越多，但细节可能越少。两种型号均受影响。')
                    seed = gr.Number(label="种子", value=31337, precision=0)
                    use_random_seed = gr.Checkbox(label="随机种子", value=True, info="使用随机种子而不是固定/递增种子")

                # Negative Prompt - Keep for Original Model, might have minor effect on F1 if CFG > 1?
                n_prompt = gr.Textbox(label="负面提示", value="", visible=True, info="当 CFG > 1.0 时使用（主要用于原始版模型）")

                with gr.Row():
                    fps = gr.Slider(label="FPS", minimum=10, maximum=60, value=30, step=1, info="输出视频的FPS值 - 直接改变生成的帧数")
                    total_second_length = gr.Slider(label="视频总时长（秒）", minimum=1, maximum=120, value=5, step=0.1)

                with gr.Row():
                    steps = gr.Slider(label="步数", minimum=1, maximum=100, value=25, step=1, info='建议两种型号的默认值均为 25.')
                    # GS - F1 Demo uses 10.0. Original uses 10.0. Keep common.
                    gs = gr.Slider(label="蒸馏CFG值", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='建议两种型号的默认值均为 10.0.')

                with gr.Row():
                    # CFG - F1 Demo uses 1.0. Original uses 1.0 but allows > 1 for neg prompt. Keep visible.
                    cfg = gr.Slider(label="CFG 数值", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=True, info='需要 > 1.0 才能进行负面提示。F1 建议我参数是 1.0.')
                    # RS - F1 Demo uses 0.0. Original uses 0.0. Keep common.
                    rs = gr.Slider(label="CFG 缩放", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=True, info='建议两种模型都使用默认值 0.0')

                gr.Markdown("### LoRA 设置（请使用对应模型的lora）")
                with gr.Row():
                    with gr.Column():
                        lora_options = scan_lora_files()
                        selected_lora = gr.Dropdown(label="选择 LoRA", choices=[name for name, _ in lora_options], value="None", info="选择要使用的 LoRA")
                    with gr.Column():
                        with gr.Row():
                             lora_refresh_btn = gr.Button(value="🔄 刷新", scale=1)
                             lora_folder_btn = gr.Button(value="📁 打开文件夹", scale=1)
                        lora_scale = gr.Slider(label="LoRA权重", minimum=0.0, maximum=9.0, value=1.0, step=0.01, info="调整 LoRA 效果的强度")

                with gr.Row():
                    # GPU Memory Preservation - Keep common setting
                    gpu_memory_preservation = gr.Slider(label="GPU 推理保留内存 (GB)", minimum=0, maximum=128, value=8, step=0.1, info="GPU 上保持空闲的内存（低显存模式）。值越大，速度越慢，但有助于防止内存溢出。根据分辨率和显存进行调整。")

                    def update_memory_for_resolution(res):
                        # Adjust defaults based on observation or keep simple mapping
                        res_int = int(res)
                        if res_int >= 1440: return 23
                        elif res_int >= 1320: return 21
                        elif res_int >= 1200: return 19
                        elif res_int >= 1080: return 16
                        elif res_int >= 960: return 14
                        elif res_int >= 840: return 12
                        elif res_int >= 720: return 10
                        elif res_int >= 640: return 8
                        else: return 6 # Default for lower resolutions
                    resolution.change(fn=update_memory_for_resolution, inputs=resolution, outputs=gpu_memory_preservation)

        with gr.Column(): # Right column for preview/results
            preview_image = gr.Image(label="下一个潜在帧", height=200, visible=False)
            result_video = gr.Video(label="成品视频", autoplay=True, show_share_button=True, height=512, loop=True)
            video_info = gr.HTML("<div id='video-info'>查看生成视频信息</div>")
            gr.Markdown('''
            **Notes:**
            - **原始模型：** 按从后向前的顺序生成视频。起始帧出现得晚，结束帧出现得早。使用重叠窗口。时间戳提示 [秒] 与最终视频时间相关，由各部分时长估算（请参阅提示上方的信息）。
            - **FramePack F1 模型：** 从起始帧开始扩展视频。起始帧始终存在。结束帧将被忽略或影响极小。时间戳提示（秒）与最终视频时间相关。
            ''')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            timing_display = gr.Markdown("", label="Time Information", elem_classes='no-generating-animation')

            gr.Markdown("### 预设（包括选定的模型）")
            with gr.Row():
                preset_dropdown = gr.Dropdown(label="选择预设", choices=scan_presets(), value=load_last_used_preset_name() or "Default")
                preset_load_button = gr.Button(value="加载预设")
                preset_refresh_button = gr.Button(value="🔄 刷新")
            with gr.Row():
                preset_save_name = gr.Textbox(label="将预设另存为", placeholder="输入预设名称.")
                preset_save_button = gr.Button(value="保存当前设置")
            preset_status_display = gr.Markdown("")

            gr.Markdown("### 文件夹选项")
            with gr.Row():
                open_outputs_btn = gr.Button(value="打开文件夹")
                open_batch_outputs_btn = gr.Button(value="打开批量输出文件夹")

            # Video Quality - Common setting
            video_quality = gr.Radio(
                label="视频质量",
                choices=["高", "中", "低", "web兼容"],
                value="high",
                info="高：最佳质量，中：平衡，低：文件大小最小，网络兼容：最佳浏览器兼容性（MP4+WebM）"
            )

            gr.Markdown("### RIFE 帧插值（仅限 MP4）")
            with gr.Row():
                rife_enabled = gr.Checkbox(label="启用 RIFE（2 倍/4 倍 FPS）", value=False, info="使用 RIFE 提高生成的 MP4 的 FPS。保存为“[filename]_extra_FPS.mp4”'")
                rife_multiplier = gr.Radio(choices=["2x FPS", "4x FPS"], label="RIFE FPS 乘数", value="2x FPS", info="选择帧率加倍.")

            gr.Markdown("### 导出其他格式")
            gr.Markdown("选择其他格式与MP4/WebM一起导出：:")
            with gr.Row():
                export_gif = gr.Checkbox(label="导出为 GIF", value=False, info="将视频另存为GIF")
                export_apng = gr.Checkbox(label="导出为APNG", value=False, info="将视频另存为PNG")
                export_webp = gr.Checkbox(label="导出为WebP", value=False, info="将视频另存为WebP")


    # --- Update Preset Component Lists ---
    preset_components_list = [
        model_selector, # ADDED
        use_multiline_prompts, save_metadata, save_individual_frames, save_intermediate_frames, save_last_frame,
        batch_skip_existing, batch_save_metadata, batch_use_multiline_prompts, batch_save_individual_frames, batch_save_intermediate_frames, batch_save_last_frame,
        num_generations, resolution, teacache_threshold, seed, use_random_seed, n_prompt, fps, total_second_length,
        latent_window_size, steps, gs, cfg, rs,
        selected_lora,
        lora_scale,
        gpu_memory_preservation, video_quality, rife_enabled, rife_multiplier, export_gif, export_apng, export_webp
    ]
    component_names_for_preset = [
        "model_selector", # ADDED
        "use_multiline_prompts", "save_metadata", "save_individual_frames", "save_intermediate_frames", "save_last_frame",
        "batch_skip_existing", "batch_save_metadata", "batch_use_multiline_prompts", "batch_save_individual_frames", "batch_save_intermediate_frames", "batch_save_last_frame",
        "num_generations", "resolution", "teacache_threshold", "seed", "use_random_seed", "n_prompt", "fps", "total_second_length",
        "latent_window_size", "steps", "gs", "cfg", "rs",
        "selected_lora",
        "lora_scale",
        "gpu_memory_preservation", "video_quality", "rife_enabled", "rife_multiplier", "export_gif", "export_apng", "export_webp"
    ]
    # --- End Preset Component List Update ---


    # --- Preset Action Functions START (Modified Load Action) ---

    def save_preset_action(name: str, *values):
        """Saves the current settings (*values) to a preset file."""
        if not name:
            return gr.update(), gr.update(value="Preset name cannot be empty.")

        preset_data = {}
        if len(values) != len(component_names_for_preset):
             msg = f"Error: Mismatched number of values ({len(values)}) and component names ({len(component_names_for_preset)})."
             print(msg)
             return gr.update(), gr.update(value=msg)

        for i, comp_name in enumerate(component_names_for_preset):
             preset_data[comp_name] = values[i]

        preset_path = get_preset_path(name)
        try:
            with open(preset_path, 'w', encoding='utf-8') as f:
                json.dump(preset_data, f, indent=4)
            save_last_used_preset_name(name)
            presets = scan_presets()
            status_msg = f"Preset '{name}' saved successfully."
            print(status_msg)
            # Return updates for dropdown and status message
            return gr.update(choices=presets, value=name), gr.update(value=status_msg)
        except Exception as e:
            error_msg = f"Error saving preset '{name}': {e}"
            print(error_msg)
            # Return updates only for status message
            return gr.update(), gr.update(value=error_msg)


    def load_preset_action(name: str, progress=gr.Progress()):
        """Loads settings from a preset file, switches model if needed, and updates UI."""
        global active_model_name # Need to potentially update this

        progress(0, desc=f"Loading preset '{name}'...")
        preset_data = load_preset_data(name)
        if preset_data is None:
             error_msg = f"Failed to load preset '{name}'."
             print(error_msg)
             # Return updates for status message and model status (no change)
             return [gr.update() for _ in preset_components_list] + [gr.update(value=error_msg), gr.update(value=f"默认加载: **{active_model_name}**")] + [gr.update()]

        # --- Model Switching from Preset ---
        model_status_update = f"当前加载: **{active_model_name}**" # Default status
        preset_model = preset_data.get("model_selector", DEFAULT_MODEL_NAME)
        if preset_model != active_model_name:
             progress(0.1, desc=f"Preset requires model '{preset_model}'. Switching...")
             print(f"Preset '{name}' requires model switch to '{preset_model}'.")
             try:
                 # Call the switch function
                 new_active_model, switch_status_msg = switch_active_model(preset_model, progress=progress)
                 # Update the global active_model_name if switch was successful (or reverted)
                 active_model_name = new_active_model
                 model_status_update = f"Status: {switch_status_msg}" # Show switch status
                 # Update the preset data in case switch failed and reverted
                 preset_data["model_selector"] = active_model_name
                 print(f"Switch status: {switch_status_msg}")
                 if "Error" in switch_status_msg or "CRITICAL" in switch_status_msg:
                      # If switch failed, maybe don't apply the rest of the preset?
                      # For now, proceed but the model might be wrong.
                      pass
             except Exception as switch_err:
                  error_msg = f"Error switching model for preset: {switch_err}"
                  print(error_msg)
                  traceback.print_exc()
                  model_status_update = f"Error switching model: {error_msg}. Model remains '{active_model_name}'."
                  preset_data["model_selector"] = active_model_name # Ensure preset reflects reality
        else:
             progress(0.5, desc="Model already correct. Loading settings...")
             print(f"Preset '{name}' uses the currently active model '{active_model_name}'.")
             model_status_update = f"Active model: **{active_model_name}**" # Confirm active model
        # --- End Model Switching ---


        # --- Apply Preset Values to UI Components ---
        updates = []
        loaded_values: Dict[str, Any] = {} # Store loaded values for iteration info update
        available_loras = [lora_name for lora_name, _ in scan_lora_files()] # Get current LoRAs

        for i, comp_name in enumerate(component_names_for_preset):
            target_component = preset_components_list[i]
            # Get the default value from the component instance if possible
            comp_initial_value = getattr(target_component, 'value', None)

            if comp_name in preset_data:
                value = preset_data[comp_name]
                # Special handling for dropdowns to ensure value exists
                if comp_name == "selected_lora":
                    if value not in available_loras:
                        print(f"Preset Warning: Saved LoRA '{value}' not found. Setting LoRA to 'None'.")
                        value = "None"
                    updates.append(gr.update(value=value))
                elif comp_name == "model_selector":
                     # Ensure the value matches the *actual* active model after potential switch attempt
                     updates.append(gr.update(value=active_model_name))
                     value = active_model_name # Use the confirmed active model name
                elif comp_name == "resolution":
                     # Ensure resolution value is valid
                     # Explicitly define valid choices to avoid potential issues with getattr during load
                     _VALID_RESOLUTIONS = ["1440","1320","1200","1080","960","840","720", "640", "480", "320", "240"]
                     if value not in _VALID_RESOLUTIONS:
                          print(f"Preset Warning: Saved resolution '{value}' not valid. Using default ('640').")
                          value = "640" # Use hardcoded default
                     updates.append(gr.update(value=value))
                else:
                    # For other components, assume the value is valid
                    updates.append(gr.update(value=value))
                loaded_values[comp_name] = value # Store the value applied/used
            else:
                 # Key missing in preset, keep component's current value
                 print(f"Preset Warning: Key '{comp_name}' not found in preset '{name}'. Keeping current value.")
                 updates.append(gr.update()) # No change update
                 loaded_values[comp_name] = comp_initial_value # Store the default/current value

        # Verify update list length
        if len(updates) != len(preset_components_list):
             error_msg = f"Error applying preset '{name}': Mismatch in component update count."
             print(error_msg)
             # Return error status and don't change components
             return [gr.update() for _ in preset_components_list] + [gr.update(value=error_msg), gr.update(value=f"Active model: **{active_model_name}**")] + [gr.update()]


        # --- Update Iteration Info Display based on loaded values ---
        vid_len = loaded_values.get('total_second_length', 5)
        fps_val = loaded_values.get('fps', 30)
        win_size = loaded_values.get('latent_window_size', 9)
        # update_iteration_info uses the global active_model_name, which should be correct now
        info_text = update_iteration_info(vid_len, fps_val, win_size)
        info_update = gr.update(value=info_text)
        # --- End Iteration Info Update ---

        save_last_used_preset_name(name) # Save preset name as last used
        status_msg = f"Preset '{name}' loaded."
        print(status_msg)
        preset_status_update = gr.update(value=status_msg)
        model_status_update_gr = gr.update(value=model_status_update) # Use the status from model switching

        # Return component updates, preset status, model status, iteration info
        return updates + [preset_status_update, model_status_update_gr, info_update]


    def refresh_presets_action():
        """Refreshes the preset dropdown list."""
        presets = scan_presets()
        last_used = load_last_used_preset_name()
        selected = last_used if last_used in presets else "Default"
        return gr.update(choices=presets, value=selected)

    # --- Preset Action Functions END ---


    # --- Gradio Event Wiring START ---
    lora_refresh_btn.click(fn=refresh_loras, outputs=[selected_lora])
    lora_folder_btn.click(fn=lambda: open_folder(loras_folder), inputs=None, outputs=None) # Simple lambda

    # --- Update ips list for process function (ADD model_selector) ---
    ips = [input_image, end_image, prompt, n_prompt, seed, use_random_seed, num_generations, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality, export_gif, export_apng, export_webp, save_metadata, resolution, fps,
           lora_scale,
           use_multiline_prompts, save_individual_frames, save_intermediate_frames, save_last_frame, rife_enabled, rife_multiplier,
           selected_lora, # LoRA dropdown component
           model_selector # ADDED Model selector component
           ]
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, seed, timing_display])

    end_button.click(fn=end_process, inputs=None, outputs=[start_button, end_button, batch_start_button, batch_end_button], cancels=[]) # Add cancels=[]

    open_outputs_btn.click(fn=lambda: open_folder(outputs_folder), inputs=None, outputs=None)
    open_batch_outputs_btn.click(fn=lambda: open_folder(outputs_batch_folder), inputs=None, outputs=None)

    batch_folder_status_text = gr.Textbox(visible=False) # Use Textbox for status messages
    open_batch_input_folder.click(fn=lambda x: open_folder(x) if x and os.path.isdir(x) else f"Folder not found or invalid: {x}", inputs=[batch_input_folder], outputs=[batch_folder_status_text])
    open_batch_end_folder.click(fn=lambda x: open_folder(x) if x and os.path.isdir(x) else f"Folder not found or invalid: {x}", inputs=[batch_end_frame_folder], outputs=[batch_folder_status_text])
    open_batch_output_folder.click(fn=lambda x: open_folder(x if x and os.path.isdir(x) else outputs_batch_folder), inputs=[batch_output_folder], outputs=[batch_folder_status_text])


    # --- Update batch_ips list for batch_process function (ADD model_selector) ---
    batch_ips = [batch_input_folder, batch_output_folder, batch_end_frame_folder, batch_prompt, n_prompt, seed, use_random_seed,
                 total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
                 teacache_threshold, video_quality, export_gif, export_apng, export_webp, batch_skip_existing,
                 batch_save_metadata, num_generations, resolution, fps,
                 lora_scale,
                 batch_use_multiline_prompts,
                 batch_save_individual_frames, batch_save_intermediate_frames, batch_save_last_frame,
                 rife_enabled, rife_multiplier,
                 selected_lora, # LoRA dropdown component
                 model_selector # ADDED Model selector component
                 ]
    batch_start_button.click(fn=batch_process, inputs=batch_ips, outputs=[result_video, preview_image, progress_desc, progress_bar, batch_start_button, batch_end_button, seed, timing_display])

    batch_end_button.click(fn=end_process, inputs=None, outputs=[start_button, end_button, batch_start_button, batch_end_button], cancels=[]) # Add cancels=[]

    # --- Preset Event Wiring START ---
    preset_save_button.click(
        fn=save_preset_action,
        inputs=[preset_save_name] + preset_components_list,
        outputs=[preset_dropdown, preset_status_display]
    )
    # Update preset load to include model_status and iteration_info_display outputs
    preset_load_button.click(
        fn=load_preset_action,
        inputs=[preset_dropdown],
        outputs=preset_components_list + [preset_status_display, model_status, iteration_info_display]
    )
    preset_refresh_button.click(
        fn=refresh_presets_action,
        inputs=[],
        outputs=[preset_dropdown]
    )
    # --- Preset Event Wiring END ---

    # --- Auto Set Latent Window Size Button Wiring --- (Unchanged)
    auto_set_lws_button.click(
        fn=auto_set_window_size,
        inputs=[fps, latent_window_size],
        outputs=[latent_window_size]
    )
    # --- End Auto Set Wiring ---

    # --- Change Listeners for Iteration Info START ---
    # Update iteration info when model changes, or relevant params change
    iteration_info_inputs = [total_second_length, fps, latent_window_size]
    # Function to trigger update
    def update_iter_info_ui(vid_len, fps_val, win_size, current_model): # Pass model just to trigger change
        # The update_iteration_info function uses the global active_model_name
        return update_iteration_info(vid_len, fps_val, win_size)

    # Trigger on model change
    model_selector.change(
        fn=update_iter_info_ui,
        inputs=iteration_info_inputs + [model_selector], # Include model selector as input
        outputs=iteration_info_display,
        queue=False
    )
    # Trigger on parameter change
    for comp in iteration_info_inputs:
        comp.change(
            fn=update_iter_info_ui,
            inputs=iteration_info_inputs + [model_selector], # Include model selector here too
            outputs=iteration_info_display,
            queue=False
        )
    # --- Change Listeners for Iteration Info END ---

    # --- Model Switch Wiring ---
    model_selector.change(
        fn=switch_active_model,
        inputs=[model_selector],
        outputs=[model_selector, model_status], # Update selector value (in case switch failed) and status
        show_progress="full" # Show progress during model switch
    )
    # --- End Model Switch Wiring ---

    # --- Gradio Event Wiring END ---


    # --- video_info_js remains unchanged ---
    video_info_js = """
    function updateVideoInfo() {
        const videoResultDiv = document.querySelector('#result_video');
        if (!videoResultDiv) return;
        const videoElement = videoResultDiv.querySelector('video');

        if (videoElement && videoElement.currentSrc && videoElement.currentSrc.startsWith('http')) { // Check if src is loaded
            const infoDiv = document.getElementById('video-info');
            if (!infoDiv) return;
            const displayInfo = () => {
                if (videoElement.videoWidth && videoElement.videoHeight && videoElement.duration && isFinite(videoElement.duration)) {
                     const format = videoElement.currentSrc ? videoElement.currentSrc.split('.').pop().toUpperCase().split('?')[0] : 'N/A'; // Handle potential query strings
                     infoDiv.innerHTML = `<p>Resolution: ${videoElement.videoWidth}x${videoElement.videoHeight} | Duration: ${videoElement.duration.toFixed(2)}s | Format: ${format}</p>`;
                } else if (videoElement.readyState < 1) {
                     infoDiv.innerHTML = '<p>Loading video info...</p>';
                } else {
                     // Sometimes duration might be Infinity initially
                     infoDiv.innerHTML = `<p>Resolution: ${videoElement.videoWidth}x${videoElement.videoHeight} | Duration: Loading... | Format: ${videoElement.currentSrc ? videoElement.currentSrc.split('.').pop().toUpperCase().split('?')[0] : 'N/A'}</p>`;
                }
            };
            // Use 'loadeddata' or 'durationchange' as they often fire when duration is known
            videoElement.removeEventListener('loadeddata', displayInfo);
            videoElement.addEventListener('loadeddata', displayInfo);
            videoElement.removeEventListener('durationchange', displayInfo);
            videoElement.addEventListener('durationchange', displayInfo);

            // Initial check if data is already available
            if (videoElement.readyState >= 2) { // HAVE_CURRENT_DATA or more
                displayInfo();
            } else {
                 infoDiv.innerHTML = '<p>Loading video info...</p>';
            }
        } else {
             const infoDiv = document.getElementById('video-info');
             if (infoDiv) infoDiv.innerHTML = "<div>Generate a video to see information</div>";
        }
    }
    // Debounce function
    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    };

    const debouncedUpdateVideoInfo = debounce(updateVideoInfo, 250); // Update max 4 times per second

    // Use MutationObserver to detect when the video src changes
    const observerCallback = function(mutationsList, observer) {
        for(const mutation of mutationsList) {
            // Check if the result_video div or its children changed, or if attributes changed (like src)
             if (mutation.type === 'childList' || mutation.type === 'attributes') {
                 const videoResultDiv = document.querySelector('#result_video');
                 // Check if the mutation target is the video element itself or its container
                 if (videoResultDiv && (mutation.target === videoResultDiv || videoResultDiv.contains(mutation.target) || mutation.target.tagName === 'VIDEO')) {
                    debouncedUpdateVideoInfo(); // Use debounced update
                    break; // No need to check other mutations for this change
                 }
            }
        }
    };
    const observer = new MutationObserver(observerCallback);
    // Observe changes in the body, looking for subtree modifications and attribute changes
    observer.observe(document.body, { childList: true, subtree: true, attributes: true, attributeFilter: ['src'] });

    // Initial call on load
    if (document.readyState === 'complete') {
      // Already loaded
      // debouncedUpdateVideoInfo(); // Call directly if loaded
    } else {
      // Wait for DOM content
      document.addEventListener('DOMContentLoaded', debouncedUpdateVideoInfo);
    }
    // Add listener for Gradio updates as well
    if (typeof window.gradio_config !== 'undefined') {
        window.addEventListener('gradio:update', debouncedUpdateVideoInfo);
    }
    """
    result_video.elem_id = "result_video"

    # --- Startup Loading START (Combined Preset & Iteration Info) ---
    def apply_preset_and_init_info_on_startup():
        global active_model_name, transformer # Allow modification of globals on startup

        print("Applying preset and initializing info on startup...")
        initial_values = {}
        for i, comp in enumerate(preset_components_list):
             # Try to get default value from component definition
             default_value = getattr(comp, 'value', None)
             initial_values[component_names_for_preset[i]] = default_value

        # Ensure Default preset exists with current defaults (including default model)
        initial_values["model_selector"] = active_model_name # Set initial model
        create_default_preset_if_needed(initial_values)

        # Load last used preset name, fallback to Default
        preset_to_load = load_last_used_preset_name()
        available_presets = scan_presets()
        if preset_to_load not in available_presets:
            print(f"Last used preset '{preset_to_load}' not found or invalid, loading 'Default'.")
            preset_to_load = "Default"
        else:
            print(f"Loading last used preset: '{preset_to_load}'")

        # Load preset data
        preset_data = load_preset_data(preset_to_load)
        if preset_data is None and preset_to_load != "Default":
             print(f"Failed to load '{preset_to_load}', attempting to load 'Default'.")
             preset_to_load = "Default"
             preset_data = load_preset_data(preset_to_load)
        elif preset_data is None and preset_to_load == "Default":
             print(f"Critical Error: Failed to load 'Default' preset data. Using hardcoded component defaults.")
             # Use the initial_values gathered from components
             preset_data = initial_values

        # --- Switch Model Based on Preset ---
        startup_model_status = f"Active model: **{active_model_name}**"
        preset_model = preset_data.get("model_selector", DEFAULT_MODEL_NAME)
        if preset_model != active_model_name:
             print(f"Startup: Preset '{preset_to_load}' requires model '{preset_model}'. Switching...")
             # Use a simple progress indicator for startup
             startup_progress = gr.Progress()
             startup_progress(0, desc=f"Loading model '{preset_model}' for startup preset...")
             try:
                 # Call switch_active_model directly - modifies globals
                 new_active_model, switch_status_msg = switch_active_model(preset_model, progress=startup_progress)
                 active_model_name = new_active_model # Update global name
                 startup_model_status = f"Status: {switch_status_msg}"
                 preset_data["model_selector"] = active_model_name # Update data with actual model
                 print(f"Startup model switch status: {switch_status_msg}")
             except Exception as startup_switch_err:
                  error_msg = f"Startup Error switching model: {startup_switch_err}"
                  print(error_msg)
                  traceback.print_exc()
                  startup_model_status = f"Error: {error_msg}. Model remains '{active_model_name}'."
                  preset_data["model_selector"] = active_model_name # Reflect reality
        else:
             print(f"Startup: Preset '{preset_to_load}' uses initially loaded model '{active_model_name}'.")
             startup_model_status = f"Active model: **{active_model_name}**" # Confirm active model
        # --- End Model Switch ---

        # --- Generate UI Updates from Preset Data ---
        preset_updates = []
        loaded_values_startup: Dict[str, Any] = {}
        available_loras_startup = [lora_name for lora_name, _ in scan_lora_files()]

        for i, comp_name in enumerate(component_names_for_preset):
            comp_initial_value = initial_values.get(comp_name) # Default from component
            value_to_set = comp_initial_value # Start with default
            if comp_name in preset_data:
                 value_from_preset = preset_data[comp_name]
                 # Validate specific components
                 if comp_name == "model_selector":
                      value_to_set = active_model_name # Always use the confirmed active model
                 elif comp_name == "selected_lora":
                      if value_from_preset not in available_loras_startup:
                           print(f"Startup Warning: Preset LoRA '{value_from_preset}' not found. Setting LoRA to 'None'.")
                           value_to_set = "None"
                      else:
                           value_to_set = value_from_preset
                 elif comp_name == "resolution":
                      # Explicitly define valid choices to avoid potential issues with getattr during load
                      _VALID_RESOLUTIONS = ["1440","1320","1200","1080","960","840","720", "640", "480", "320", "240"]
                      if value_from_preset not in _VALID_RESOLUTIONS:
                           print(f"Startup Warning: Preset resolution '{value_from_preset}' not valid. Using default ('640').")
                           value_to_set = "640" # Use hardcoded default
                      else:
                           value_to_set = value_from_preset
                 else:
                      value_to_set = value_from_preset # Use preset value directly
            else:
                 print(f"Startup Warning: Key '{comp_name}' missing in '{preset_to_load}'. Using component's default.")
                 value_to_set = comp_initial_value # Keep default

            preset_updates.append(gr.update(value=value_to_set))
            loaded_values_startup[comp_name] = value_to_set # Store applied value

        # --- Calculate Initial Iteration Info ---
        initial_vid_len = loaded_values_startup.get('total_second_length', 5)
        initial_fps = loaded_values_startup.get('fps', 30)
        initial_win_size = loaded_values_startup.get('latent_window_size', 9)
        # update_iteration_info uses the global active_model_name set during model switch
        initial_info_text = update_iteration_info(initial_vid_len, initial_fps, initial_win_size)

        # Return updates for: preset dropdown, all components, model status, iteration info
        return [gr.update(choices=available_presets, value=preset_to_load)] + preset_updates + [startup_model_status, initial_info_text]

    block.load(
        fn=apply_preset_and_init_info_on_startup,
        inputs=[],
        outputs=[preset_dropdown] + preset_components_list + [model_status, iteration_info_display]
    )
    # --- Startup Loading END ---

    block.load(None, None, None, js=video_info_js)


# --- get_available_drives remains unchanged ---
def get_available_drives():
    """Detect available drives on the system regardless of OS"""
    available_paths = []
    if platform.system() == "Windows":
        import string
        from ctypes import windll
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1: drives.append(f"{letter}:\\")
            bitmask >>= 1
        available_paths = drives
    elif platform.system() == "Darwin":
         available_paths = ["/", "/Volumes"] # Add standard mount point
    else: # Linux/Other Unix
        available_paths = ["/", "/mnt", "/media"] # Common mount points
        # Also add home directory for convenience
        home_dir = os.path.expanduser("~")
        if home_dir not in available_paths:
            available_paths.append(home_dir)

    # Filter out paths that don't actually exist
    existing_paths = [p for p in available_paths if os.path.exists(p) and os.path.isdir(p)]

    # Add the current working directory and script directory for relative paths
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if cwd not in existing_paths: existing_paths.append(cwd)
    if script_dir not in existing_paths: existing_paths.append(script_dir)

    # Add specific output/input folders relative to script dir
    for folder in [outputs_folder, loras_folder, presets_folder, outputs_batch_folder]:
         abs_folder = os.path.abspath(folder)
         if abs_folder not in existing_paths and os.path.exists(abs_folder):
             existing_paths.append(abs_folder)

    print(f"Allowed Gradio paths: {list(set(existing_paths))}") # Use set to remove duplicates
    return list(set(existing_paths))


block.launch(
    share=args.share,
    inbrowser=True,
    allowed_paths=get_available_drives() # Use dynamically generated paths
)
