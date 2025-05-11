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
from typing import Optional # Added for type hinting
import sys # Added for RIFE
import cv2 # Added for RIFE
import json # <-- ADDED FOR PRESETS

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

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = False

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()

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

transformer.to(dtype=torch.bfloat16)
vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)
transformer.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    transformer.to(gpu)

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
    """Get all image files from a folder."""
    if not folder_path or not os.path.exists(folder_path):
        return []

    # Get dynamically supported image formats
    image_extensions = print_supported_image_formats()
    images = []

    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in image_extensions:
            images.append(file_path)

    return sorted(images)

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
            return True
        # Try peft's adapter handling if available
        elif hasattr(model, "peft_config") and model.peft_config:
            if hasattr(model, "disable_adapters"):
                print("Unloading LoRA using disable_adapters method")
                model.disable_adapters()
                return True
            # For PEFT models without disable_adapters method
            elif hasattr(model, "active_adapters") and model.active_adapters:
                print("Clearing active adapters list")
                model.active_adapters = []
                return True
        # Special handling for DynamicSwap models
        elif is_dynamic_swap:
            print("DynamicSwap model detected, attempting to reset internal model state")

            # For DynamicSwap models, try to check if there's an internal model that has LoRA attributes
            if hasattr(model, "model"):
                internal_model = model.model
                if hasattr(internal_model, "unload_lora_weights"):
                    print("Unloading LoRA from internal model")
                    internal_model.unload_lora_weights()
                    return True
                elif hasattr(internal_model, "peft_config") and internal_model.peft_config:
                    if hasattr(internal_model, "disable_adapters"):
                        print("Disabling adapters on internal model")
                        internal_model.disable_adapters()
                        return True

            # If all else fails with DynamicSwap, try to directly remove LoRA modules
            print("Attempting direct LoRA module removal as fallback")
            return force_remove_lora_modules(model)
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
        for name, module in list(model.named_modules()):
            # Check for typical PEFT/LoRA module names
            if 'lora' in name.lower():
                print(f"Found LoRA module: {name}")
                lora_removed = True

                # Get parent module and attribute name
                parent_name, _, attr_name = name.rpartition('.')
                if parent_name:
                    try:
                        parent = model.get_submodule(parent_name)
                        if hasattr(parent, attr_name):
                            # Try to restore original module if possible
                            if hasattr(module, 'original_module'):
                                setattr(parent, attr_name, module.original_module)
                                print(f"Restored original module for {name}")
                            # Otherwise just try to reset the module
                            else:
                                print(f"Could not restore original module for {name}")
                    except Exception as e:
                        print(f"Error accessing parent module {parent_name}: {str(e)}")

        # Clear PEFT configuration
        if hasattr(model, "peft_config"):
            model.peft_config = None
            print("Cleared peft_config")
            lora_removed = True

        # Clear LoRA adapter references
        if hasattr(model, "active_adapters"):
            model.active_adapters = []
            print("Cleared active_adapters")
            lora_removed = True

        return lora_removed
    except Exception as e:
        print(f"Error during force LoRA removal: {str(e)}")
        traceback.print_exc()
        return False

print_supported_image_formats()

# --- Preset Functions START --- (Added for Presets)

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

# --- Updated Function ---
def update_iteration_info(vid_len_s, fps_val, win_size):
    """Calculates and formats information about generation sections."""
    # (Keep the initial validation and calculation parts the same)
    if not all([isinstance(vid_len_s, (int, float)), isinstance(fps_val, int), isinstance(win_size, int)]):
         return "Calculating..."
    if fps_val <= 0 or win_size <= 0:
         return "Invalid FPS or Latent Window Size."

    try:
        # Calculate total sections using the same logic as the worker
        total_frames_needed = vid_len_s * fps_val
        frames_per_section_calc = win_size * 4 # Used for section count and duration timing

        # Calculate total sections needed (ensure division by zero doesn't happen)
        total_latent_sections = 0
        if frames_per_section_calc > 0:
            total_latent_sections = int(max(round(total_frames_needed / frames_per_section_calc), 1))
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

        info_text = (
            f"**Generation Info:** Approx. **{total_latent_sections}** section(s) will be generated.\n"
            f"Each section represents {timing_description} of the final video time.\n"
            f"(One section processes {frames_in_one_section} frames internally with overlap at {fps_val} FPS).\n"
            f"*Use the **{section_duration_seconds:.2f}s per section** estimate for '[seconds] prompt' timings.*"
        )

        # --- CORRECTED TIP ---
        # Calculate the ideal LWS for 1s sections at this FPS
        ideal_lws_float = fps_val / 4.0
        ideal_lws_int = round(ideal_lws_float)
        ideal_lws_clamped = max(1, min(ideal_lws_int, 33)) # Clamp to slider range [1, 33]

        # Check if the *current* setting is NOT already the ideal one for ~1s sections
        if win_size != ideal_lws_clamped:
            # Check if the ideal clamped LWS *would* result in near 1s sections
            ideal_duration = (ideal_lws_clamped * 4) / fps_val
            if abs(ideal_duration - 1.0) < 0.01:
                 info_text += f"\n\n*Tip: Set Latent Window Size to **{ideal_lws_clamped}** for (near) exact 1-second sections at {fps_val} FPS.*"
            # else: # Optional: Add note if even the ideal int LWS isn't close to 1s
            #    info_text += f"\n\n*Note: Exact 1-second sections may not be achievable with integer Latent Window Sizes at {fps_val} FPS.*"

        return info_text
    except Exception as e:
        print(f"Error calculating iteration info: {e}")
        traceback.print_exc() # Add traceback
        return "Error calculating info."
# --- End Updated Function ---


@torch.no_grad()
def worker(input_image, end_image, prompt, n_prompt, seed, use_random_seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality='high', export_gif=False, export_apng=False, export_webp=False, num_generations=1, resolution="640", fps=30, selected_lora="none", lora_scale=1.0, save_individual_frames_flag=False, save_intermediate_frames_flag=False, save_last_frame_flag=False, use_multiline_prompts_flag=False, rife_enabled=False, rife_multiplier="2x FPS"): # Added RIFE params
    # Removed convert_lora from signature
    # Declare global variables at the beginning of the function
    global transformer, text_encoder, text_encoder_2, image_encoder, vae
    global individual_frames_folder, intermediate_individual_frames_folder, last_frames_folder, intermediate_last_frames_folder # Ensure these are accessible if modified

    total_latent_sections = (total_second_length * fps) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    # Clean up any previously loaded LoRA at the start of a new worker session
    if hasattr(transformer, "peft_config") and transformer.peft_config:
        print("Cleaning up previous LoRA weights at worker start")
        safe_unload_lora(transformer, gpu)

    # --- MODIFICATION START ---
    # Check for timestamped prompts ONLY if multi-line is disabled
    parsed_prompts = None
    encoded_prompts = {} # Dictionary to store encoded prompts {prompt_text: (tensors)}
    using_timestamped_prompts = False

    if not use_multiline_prompts_flag: # Check the flag passed from process/batch_process
        parsed_prompts = parse_simple_timestamped_prompt(prompt, total_second_length, latent_window_size, fps)
        if parsed_prompts:
            using_timestamped_prompts = True
            print("Using timestamped prompts.")
        else:
            print("Timestamped prompt format not detected or invalid, using the entire prompt as one.")
    else:
        print("Multi-line prompts enabled, skipping timestamp parsing.")
    # --- MODIFICATION END ---


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
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Starting generation {gen_idx+1}/{num_generations} with seed {current_seed}...'))))

        try:
            # Clean GPU
            if not high_vram:
                unload_complete_models(
                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                )

            # Text encoding
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

            if not high_vram:
                fake_diffusers_current_device(text_encoder, gpu)
                load_model_as_complete(text_encoder_2, target_device=gpu)

            # --- MODIFICATION START ---
            # Pre-encode prompts based on whether we parsed timestamps or not
            if using_timestamped_prompts:
                unique_prompts = set(p[1] for p in parsed_prompts)
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Encoding {len(unique_prompts)} unique timestamped prompts...'))))
                for p_text in unique_prompts:
                    if p_text not in encoded_prompts:
                         llama_vec_p, clip_l_pooler_p = encode_prompt_conds(p_text, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                         llama_vec_p, llama_attention_mask_p = crop_or_pad_yield_mask(llama_vec_p, length=512)
                         encoded_prompts[p_text] = (llama_vec_p, llama_attention_mask_p, clip_l_pooler_p)
                print(f"Pre-encoded {len(encoded_prompts)} unique prompts.")
                # Use the tensors from the *first* parsed prompt for negative prompt shape matching if needed
                # Ensure parsed_prompts is not empty before accessing
                if not parsed_prompts:
                    raise ValueError("Timestamped prompts were detected but parsing resulted in an empty list.")
                llama_vec, llama_attention_mask, clip_l_pooler = encoded_prompts[parsed_prompts[0][1]]
            else:
                # Original single prompt encoding
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Encoding single prompt...'))))
                llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
                # Store it for consistency, although not strictly needed if not timestamped
                encoded_prompts[prompt] = (llama_vec, llama_attention_mask, clip_l_pooler)

            # Negative prompt encoding (remains mostly the same)
            if cfg == 1.0: # Check against float
                # Use zero tensors matching the shape of the (first) positive prompt
                first_prompt_key = list(encoded_prompts.keys())[0]
                ref_llama_vec = encoded_prompts[first_prompt_key][0]
                ref_clip_l = encoded_prompts[first_prompt_key][2]
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(ref_llama_vec), torch.zeros_like(ref_clip_l)
                # Need a zero mask too
                ref_llama_mask = encoded_prompts[first_prompt_key][1]
                llama_attention_mask_n = torch.zeros_like(ref_llama_mask)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)
                llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)

            # Dtype conversions
            for p_text in encoded_prompts:
                 l_vec, l_mask, c_pool = encoded_prompts[p_text]
                 encoded_prompts[p_text] = (l_vec.to(transformer.dtype), l_mask, c_pool.to(transformer.dtype))

            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
            # --- MODIFICATION END ---


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
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]

            # Processing end image (if provided)
            has_end_image = end_image is not None
            end_image_np = None
            end_image_pt = None
            end_latent = None # Initialize end_latent
            if has_end_image:
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing end frame ...'))))
                # Use the same bucket size as the start image
                end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
                try:
                    os.makedirs(used_images_folder, exist_ok=True)
                    Image.fromarray(end_image_np).save(os.path.join(used_images_folder, f'{job_id}_end.png'))
                    print(f"Saved end image to {os.path.join(used_images_folder, f'{job_id}_end.png')}")
                except Exception as e:
                    print(f"Error saving end image: {str(e)}")
                end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1
                end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]

            # VAE encoding
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))
            if not high_vram:
                load_model_as_complete(vae, target_device=gpu)
            start_latent = vae_encode(input_image_pt, vae)
            if has_end_image:
                end_latent = vae_encode(end_image_pt, vae) # Encode end latent

            # CLIP Vision
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))
            if not high_vram:
                load_model_as_complete(image_encoder, target_device=gpu)

            # Encode start image
            image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state

            # Encode end image and combine if present
            if has_end_image:
                end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
                end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
                # Combine embeddings (simple average)
                image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2.0
                print("Combined start and end frame CLIP vision embeddings.")

            # Dtype conversion for image encoder (moved after potential combination)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)


            # Apply LoRA if selected
            using_lora = False
            previous_lora_loaded = hasattr(transformer, "peft_config") and transformer.peft_config

            if selected_lora != "none":
                stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Loading LoRA {os.path.basename(selected_lora)}...'))))
                try:
                    lora_path, lora_name = os.path.split(selected_lora)
                    if not lora_path:
                        lora_path = loras_folder

                    if previous_lora_loaded:
                        print("Unloading previously loaded LoRA before loading new one")
                        lora_unload_success = safe_unload_lora(transformer, gpu)

                        if not lora_unload_success and 'DynamicSwap' in transformer.__class__.__name__:
                            print("LoRA unloading failed for DynamicSwap model - need to reload model")
                            if not high_vram:
                                unload_complete_models(
                                    text_encoder, text_encoder_2, image_encoder, vae, transformer
                                )
                                from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
                                transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
                                transformer.high_quality_fp32_output_for_inference = True
                                transformer.requires_grad_(False)
                                if not high_vram:
                                    from diffusers_helper.memory import DynamicSwapInstaller
                                    DynamicSwapInstaller.install_model(transformer, device=gpu)
                                else:
                                     transformer.to(gpu)
                                print("Successfully reloaded transformer model")

                    transformer.to(gpu)

                    # Load LoRA - conversion is handled internally by load_lora -> _convert_hunyuan_video_lora_to_diffusers
                    current_transformer = load_lora(transformer, lora_path, lora_name)
                    adapter_name = os.path.splitext(lora_name)[0] # Get adapter name from filename
                    print(f"LoRA '{adapter_name}' loaded. Applying scale: {lora_scale}")

                    print("Verifying all LoRA components are on GPU...")
                    for name, module in transformer.named_modules():
                        if 'lora_' in name.lower():
                            module.to(gpu)

                    for name, param in transformer.named_parameters():
                        if 'lora' in name.lower():
                            if param.device.type != 'cuda':
                                print(f"Force moving LoRA parameter {name} from {param.device} to {gpu}")
                                param.data = param.data.to(gpu)

                    # Apply the scale using set_adapters
                    # Pass adapter_name and lora_scale within lists as expected by set_adapters
                    set_adapters(transformer, [adapter_name], [lora_scale])
                    print(f"Scale {lora_scale} applied to LoRA adapter '{adapter_name}'.")

                    using_lora = True
                    print(f"Successfully loaded and configured LoRA: {lora_name} with scale: {lora_scale}")
                except Exception as e:
                    print(f"Error loading LoRA {selected_lora}: {str(e)}")
                    traceback.print_exc()

            # Sampling
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, f'Start sampling generation {gen_idx+1}/{num_generations}...'))))

            rnd = torch.Generator("cpu").manual_seed(current_seed)
            num_frames = latent_window_size * 4 - 3

            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
            history_pixels = None
            total_generated_latent_frames = 0

            # Make latent_paddings a list for indexing
            base_latent_paddings = reversed(range(total_latent_sections))
            if total_latent_sections > 4:
                latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            else:
                latent_paddings = list(base_latent_paddings) # Convert iterator to list

            # --- MODIFICATION for latent_paddings loop ---
            duration_per_section = (latent_window_size * 4 - 3) / fps
            current_prompt_text_for_callback = prompt # Default for callback

            for i, latent_padding in enumerate(latent_paddings):
                is_last_section = latent_padding == 0
                # is_first_section determines if we are generating the END of the video (highest padding)
                is_first_section = (i == 0) # Check if it's the first item in the list/iterator

                latent_padding_size = latent_padding * latent_window_size

                if stream.input_queue.top() == 'end':
                    stream.output_queue.push(('end', None))
                    print("Worker detected end signal during latent processing")
                    try:
                        if not high_vram:
                            unload_complete_models(
                                text_encoder, text_encoder_2, image_encoder, vae, transformer
                            )
                    except Exception as cleanup_error:
                        print(f"Error during cleanup: {str(cleanup_error)}")
                    return

                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

                # --- MODIFICATION START ---
                # Determine current prompt tensors based on time if using timestamped prompts
                first_prompt_key = list(encoded_prompts.keys())[0]
                active_llama_vec = encoded_prompts[first_prompt_key][0] # Default to first prompt
                active_llama_mask = encoded_prompts[first_prompt_key][1]
                active_clip_pooler = encoded_prompts[first_prompt_key][2]
                current_prompt_text_for_callback = first_prompt_key # Default text

                if using_timestamped_prompts:
                    # ===>>> START OF THE REVISED CHANGE <<<===
                    # Calculate the approximate video time corresponding to the *start* of the current latent section being generated.
                    # Use the loop iteration index 'i' for progress, as 'latent_padding' is not linear w.r.t time.
                    # Generation runs backward (i=0 is end frame, i=N-1 is start frame), so map 'i' to time accordingly.

                    if total_latent_sections > 1:
                        # Calculate exact seconds per section based on frame relationship
                        # Each section corresponds to (latent_window_size * 4) / fps seconds of video time
                        section_duration_seconds = (latent_window_size * 4) / fps

                        # Map loop iteration index 'i' (0 to N-1) to time (total_length -> 0)
                        # With proper settings (LWS = fps/4), each section represents exactly 1 second
                        current_video_time = total_second_length - (i * section_duration_seconds)

                        # Ensure we don't go below 0
                        if current_video_time < 0:
                            current_video_time = 0.0
                    else:
                        # If only one section, it essentially generates based on the initial state (time 0).
                        current_video_time = 0.0
                    # ===>>> END OF THE REVISED CHANGE <<<===


                    # DEBUG: Print all available prompts and current position
                    print(f"\n===== PROMPT DEBUG INFO =====")
                    print(f"Iteration: {i} / {total_latent_sections - 1}") # Use i for clarity
                    print(f"Latent padding value for this iteration: {latent_padding}") # Keep for info
                    print(f"Current video time (mapped from iteration): {current_video_time:.2f}s")
                    print(f"Available prompts: {parsed_prompts}")

                    # Find the appropriate prompt for the current time
                    # The prompt active AT or BEFORE the current_video_time
                    selected_prompt_text = parsed_prompts[0][1]  # Default to the first prompt ([0] time)
                    last_matching_time = parsed_prompts[0][0]

                    print(f"Checking against prompts...")
                    # Iterate through prompts sorted by time [0s, 10s, 20s]
                    for start_time_prompt, p_text in parsed_prompts:
                        # If the current video time we are generating FOR
                        # is >= the timestamp of the prompt, that prompt is potentially active.
                        # We take the LAST one that matches this condition.
                        print(f"  - Checking time {start_time_prompt:.2f}s ('{p_text[:20]}...') vs current_video_time {current_video_time:.2f}s")
                        # Add a small epsilon to handle floating point comparisons near timestamps
                        epsilon = 1e-4
                        if current_video_time >= (start_time_prompt - epsilon):
                             selected_prompt_text = p_text
                             last_matching_time = start_time_prompt
                             print(f"    - MATCH: Current time {current_video_time:.2f}s >= {start_time_prompt}s. Tentative selection: '{selected_prompt_text[:20]}...'")
                        else:
                            # Since prompts are sorted, once we find a timestamp > current_video_time,
                            # we know the previous one was the correct one to use.
                            print(f"    - NO MATCH: Current time {current_video_time:.2f}s < {start_time_prompt}s. Stopping search.")
                            break # Stop searching

                    print(f"Final selected prompt active at/before {current_video_time:.2f}s is the one from {last_matching_time}s: '{selected_prompt_text}'")
                    print(f"===== END DEBUG INFO =====\n")

                    # Retrieve the encoded tensors
                    active_llama_vec, active_llama_mask, active_clip_pooler = encoded_prompts[selected_prompt_text]
                    current_prompt_text_for_callback = selected_prompt_text # Update for callback

                    # Print the current time and selected prompt
                    print(f'---> Generating section corresponding to video time >= {last_matching_time:.2f}s, Using prompt: "{selected_prompt_text[:50]}..."')


                else:
                     # If not using timestamped prompts, use the single encoded prompt
                     active_llama_vec, active_llama_mask, active_clip_pooler = encoded_prompts[prompt]
                     current_prompt_text_for_callback = prompt
                # --- MODIFICATION END ---


                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

                # Determine clean latents
                clean_latents_pre = start_latent.to(history_latents)
                # Get original post latents from history
                clean_latents_post_orig, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)

                # Use end_latent for the post-latents only during the first section (end frame generation)
                if has_end_image and is_first_section:
                    clean_latents_post = end_latent.to(history_latents)
                    print("Using end_latent for clean_latents_post in the first section.")
                else:
                    clean_latents_post = clean_latents_post_orig

                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)


                if not high_vram:
                    unload_complete_models()
                    move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                    if using_lora:
                        print("Ensuring LoRA adapters are on correct device (cuda:0)...")
                        for name, module in transformer.named_modules():
                            if 'lora_' in name.lower():
                                try: module.to(gpu)
                                except Exception as e: print(f"Error moving LoRA module {name}: {str(e)}")

                if teacache_threshold > 0:
                    print(f"Teacache: {teacache_threshold}")
                    transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_threshold)
                else:
                    print("Teacache disabled")
                    transformer.initialize_teacache(enable_teacache=False)

                sampling_start_time = time.time() # Start time for this specific sampling step

                # --- MODIFICATION for callback ---
                def callback(d):
                    preview = d['denoised']
                    preview = vae_decode_fake(preview)

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

                    elapsed_time = time.time() - sampling_start_time # Elapsed time for *this specific sampling step*
                    time_per_step = elapsed_time / current_step if current_step > 0 else 0
                    remaining_steps = steps - current_step
                    eta_seconds = time_per_step * remaining_steps

                    expected_frames = latent_window_size * 4 - 3

                    if current_step == steps:
                        post_processing_eta = expected_frames * estimated_vae_time_per_frame + estimated_save_time
                        eta_seconds = post_processing_eta
                    else:
                        post_processing_eta = expected_frames * estimated_vae_time_per_frame + estimated_save_time
                        eta_seconds += post_processing_eta

                    eta_str = format_time_human_readable(eta_seconds)
                    total_elapsed = time.time() - gen_start_time # <-- CORRECTED: Use current gen start time
                    elapsed_str = format_time_human_readable(total_elapsed)

                    hint = f'Sampling {current_step}/{steps} (Gen {gen_idx+1}/{num_generations}, Seed {current_seed})'

                    # --- MODIFICATION START for desc ---
                    desc = f'Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / fps) :.2f} seconds (FPS-{fps}).'
                    if using_timestamped_prompts:
                        desc += f' Current Prompt: "{current_prompt_text_for_callback[:50]}..."'
                    # --- MODIFICATION END for desc ---

                    time_info = f'Elapsed: {elapsed_str} | ETA: {eta_str}'

                    print(f"\rProgress: {percentage}% | {hint} | {time_info}     ", end="")
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, f"{hint}<br/>{time_info}"))))
                    return
                # --- END MODIFICATION for callback ---

                try:
                    if using_lora:
                        print("Final device check for all tensors before sampling...")
                        devices_found = set()
                        for name, param in transformer.named_parameters():
                            if param.requires_grad or 'lora' in name.lower():
                                devices_found.add(str(param.device))
                                if param.device.type != 'cuda':
                                    print(f"Moving {name} from {param.device} to {gpu}")
                                    param.data = param.data.to(gpu)
                        print(f"Devices found for parameters: {devices_found}")

                    # --- MODIFICATION for sample_hunyuan call ---
                    generated_latents = sample_hunyuan(
                        transformer=transformer,
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=num_frames,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
                        num_inference_steps=steps,
                        generator=rnd,
                        # Use the active tensors determined earlier
                        prompt_embeds=active_llama_vec,
                        prompt_embeds_mask=active_llama_mask,
                        prompt_poolers=active_clip_pooler,
                        # Negative prompts remain the same
                        negative_prompt_embeds=llama_vec_n,
                        negative_prompt_embeds_mask=llama_attention_mask_n,
                        negative_prompt_poolers=clip_l_pooler_n,
                        device=gpu,
                        dtype=torch.bfloat16,
                        image_embeddings=image_encoder_last_hidden_state,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=callback,
                    )
                    # --- END MODIFICATION for sample_hunyuan call ---
                except ConnectionResetError as e:
                    print(f"Connection Reset Error caught during sampling: {str(e)}")
                    print("Continuing with the process anyway...")
                    if stream.input_queue.top() == 'end':
                        stream.output_queue.push(('end', None))
                        return
                    empty_shape = (1, 16, latent_window_size, height // 8, width // 8)
                    generated_latents = torch.zeros(empty_shape, dtype=torch.float32).cpu()
                    print("Skipping to next generation due to connection error")
                    break # Skip current generation

                section_time = time.time() - sampling_start_time
                print(f"\nSection completed in {section_time:.2f} seconds")
                print(f"VAE decoding started (takes longer on higher resolution)... {'Using standard decoding' if high_vram else 'Using memory optimization: VAE offloading enabled'}")

                if is_last_section:
                    generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

                if not high_vram:
                    offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)
                    load_model_as_complete(vae, target_device=gpu)

                vae_start_time = time.time()
                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

                if history_pixels is None:
                    history_pixels = vae_decode(real_history_latents, vae).cpu()
                else:
                    section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                    overlapped_frames = latent_window_size * 4 - 3
                    current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

                vae_time = time.time() - vae_start_time
                num_frames_decoded = real_history_latents.shape[2]
                vae_time_per_frame = vae_time / num_frames_decoded if num_frames_decoded > 0 else estimated_vae_time_per_frame
                vae_time_history.append(vae_time_per_frame)
                if len(vae_time_history) > 0:
                    estimated_vae_time_per_frame = sum(vae_time_history) / len(vae_time_history)
                print(f"VAE decoding completed in {vae_time:.2f} seconds ({vae_time_per_frame:.3f} sec/frame)")

                if not high_vram:
                    unload_complete_models()

                is_intermediate = not is_last_section
                if is_intermediate:
                    output_filename = os.path.join(intermediate_videos_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
                    webm_output_filename = os.path.join(intermediate_webm_videos_folder, f'{job_id}_{total_generated_latent_frames}.webm')
                else:
                    output_filename = os.path.join(outputs_folder, f'{job_id}.mp4')  # Use only the timestamp as filename for final output
                    webm_output_filename = os.path.join(webm_videos_folder, f'{job_id}.webm')

                save_start_time = time.time()
                try:
                    # --- Existing MP4 Saving ---
                    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
                    save_bcthw_as_mp4(history_pixels, output_filename, fps=fps, video_quality=video_quality)
                    print(f"Saved MP4 video to {output_filename}")

                    # --- SAVE LAST FRAME (MP4 ONLY) - Moved Here ---
                    # Save last frame from the ORIGINAL history_pixels BEFORE RIFE
                    if save_last_frame_flag is True and output_filename and os.path.exists(output_filename): # Check flag is True and MP4 exists
                        try:
                            print(f"Attempting to save last frame for {output_filename}")
                            last_frame_base_name = os.path.splitext(os.path.basename(output_filename))[0]
                            frames_output_dir = os.path.join(
                                intermediate_last_frames_folder if is_intermediate else last_frames_folder,
                                last_frame_base_name # Use video base name for subfolder
                            )
                            # Ensure the specific output directory for this frame exists
                            os.makedirs(frames_output_dir, exist_ok=True)
                            # Pass the specific dir and a simple filename base
                            save_last_frame_to_file(history_pixels, frames_output_dir, f"{last_frame_base_name}_lastframe")
                        except Exception as lf_err:
                            print(f"Error saving last frame for {output_filename}: {str(lf_err)}")
                            traceback.print_exc()
                    # --- END SAVE LAST FRAME ---

                    # --- START OF RIFE INTEGRATION ---
                    if rife_enabled and output_filename and os.path.exists(output_filename):
                        print(f"RIFE Enabled: Processing {output_filename}")
                        try:
                            # 1. Check source FPS (Optional but recommended)
                            cap = cv2.VideoCapture(output_filename)
                            source_fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                            print(f"Source MP4 FPS: {source_fps:.2f}")

                            # Only apply RIFE if source FPS is not excessively high (e.g., <= 60)
                            if source_fps <= 60:
                                # 2. Determine multiplier
                                multiplier_val = "4" if rife_multiplier == "4x FPS" else "2"
                                print(f"Using RIFE multiplier: {multiplier_val}x")

                                # 3. Construct output filename
                                rife_output_filename = os.path.splitext(output_filename)[0] + '_extra_FPS.mp4'
                                print(f"RIFE output filename: {rife_output_filename}")

                                # 4. Construct RIFE command
                                rife_script_path = os.path.abspath(os.path.join(current_dir, "Practical-RIFE", "inference_video.py"))
                                rife_model_path = os.path.abspath(os.path.join(current_dir, "Practical-RIFE", "train_log")) # Directory containing model files

                                # Check if script and model dir exist
                                if not os.path.exists(rife_script_path):
                                     print(f"ERROR: RIFE script not found at {rife_script_path}")
                                elif not os.path.exists(rife_model_path):
                                    print(f"ERROR: RIFE model directory not found at {rife_model_path}")
                                else:
                                    # Use full paths and quotes for safety
                                    cmd = (
                                        f'"{sys.executable}" "{rife_script_path}" '
                                        f'--model="{rife_model_path}" '
                                        f'--multi={multiplier_val} '
                                        f'--video="{os.path.abspath(output_filename)}" '
                                        f'--output="{os.path.abspath(rife_output_filename)}"'
                                    )
                                    print(f"Executing RIFE command: {cmd}")

                                    # 5. Execute command
                                    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ)

                                    if result.returncode == 0:
                                        if os.path.exists(rife_output_filename):
                                            print(f"Successfully applied RIFE. Saved as: {rife_output_filename}")
                                            # Display the RIFE-enhanced video instead of the original
                                            stream.output_queue.push(('rife_file', rife_output_filename))
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

                        # Save last frame of webm video if enabled --> REMOVED (Only MP4)

                    # Save individual frames if enabled
                    if ((is_intermediate and save_intermediate_frames_flag) or # Use the flag
                        (not is_intermediate and save_individual_frames_flag)): # Use the flag
                        # Create a subfolder with the video filename
                        frames_output_dir = os.path.join(
                            intermediate_individual_frames_folder if is_intermediate else individual_frames_folder,
                            os.path.splitext(os.path.basename(output_filename))[0]
                        )
                        # Use the filename base for individual frames
                        from diffusers_helper.utils import save_individual_frames
                        save_individual_frames(history_pixels, frames_output_dir, job_id)
                        print(f"Saved individual frames to {frames_output_dir}")

                except ConnectionResetError as e:
                    print(f"Connection Reset Error during video saving: {str(e)}")
                    print("Continuing with the process anyway...")
                    output_filename = None
                    webm_output_filename = None
                except Exception as e:
                    # MODIFIED: Added traceback
                    print(f"Error saving MP4/WebM video or associated last frame: {str(e)}") # Clarify potential error source
                    traceback.print_exc() # Add traceback for better debugging
                    output_filename = None
                    webm_output_filename = None

                # Metadata saving logic
                save_metadata_enabled = True # Assume true for now, should be passed ideally
                # --- MODIFICATION for metadata ---
                if save_metadata_enabled and is_last_section:
                    gen_time_current = time.time() - gen_start_time # Use gen_start_time for current gen time
                    generation_time_seconds = int(gen_time_current)
                    generation_time_formatted = format_time_human_readable(gen_time_current)

                    # Determine the prompt to save in metadata
                    metadata_prompt = prompt # Original full prompt by default
                    if using_timestamped_prompts:
                        # Save the original multi-line prompt with timestamps
                        metadata_prompt = prompt

                    metadata = {
                        "Prompt": metadata_prompt, # Use the chosen prompt representation
                        "Seed": current_seed,
                        "TeaCache": f"Enabled (Threshold: {teacache_threshold})" if teacache_threshold > 0 else "Disabled",
                        "Video Length (seconds)": total_second_length,
                        "FPS": fps,
                        "Latent Window Size": latent_window_size, # <-- ADDED
                        "Steps": steps,
                        "CFG Scale": cfg, # Add explicit CFG Scale
                        "Distilled CFG Scale": gs,
                        "Guidance Rescale": rs, # Add Guidance Rescale
                        "Resolution": resolution,
                        "Generation Time": generation_time_formatted,
                        "Total Seconds": f"{generation_time_seconds} seconds",
                        "Start Frame Provided": True,
                        "End Frame Provided": has_end_image,
                        "Timestamped Prompts Used": using_timestamped_prompts, # Add flag
                    }

                    if selected_lora != "none":
                        lora_name = os.path.basename(selected_lora)
                        metadata["LoRA"] = lora_name
                        metadata["LoRA Scale"] = lora_scale
                        # metadata["LoRA Conversion"] = "Enabled" if convert_lora else "Disabled" # Removed

                    # Only save metadata for the original MP4 file
                    if output_filename: save_processing_metadata(output_filename, metadata)
                    # Metadata saving for other formats (no change needed for RIFE here)
                    if export_gif and os.path.exists(os.path.splitext(output_filename)[0] + '.gif'):
                        save_processing_metadata(os.path.splitext(output_filename)[0] + '.gif', metadata)
                    if export_apng and os.path.exists(os.path.splitext(output_filename)[0] + '.png'):
                        save_processing_metadata(os.path.splitext(output_filename)[0] + '.png', metadata)
                    if export_webp and os.path.exists(os.path.splitext(output_filename)[0] + '.webp'):
                        save_processing_metadata(os.path.splitext(output_filename)[0] + '.webp', metadata)
                    if video_quality == 'web_compatible' and webm_output_filename and os.path.exists(webm_output_filename):
                        save_processing_metadata(webm_output_filename, metadata)
                # --- END MODIFICATION for metadata ---


                # Save additional formats (no change needed for RIFE)
                try:
                    if export_gif:
                        gif_filename = os.path.join(intermediate_gif_videos_folder if is_intermediate else gif_videos_folder, f'{job_id}_{total_generated_latent_frames}.gif' if is_intermediate else f'{job_id}.gif')
                        try:
                            os.makedirs(os.path.dirname(gif_filename), exist_ok=True)
                            save_bcthw_as_gif(history_pixels, gif_filename, fps=fps)
                            print(f"Saved GIF animation to {gif_filename}")

                            # REMOVED Last frame saving for GIF

                        except Exception as e: print(f"Error saving GIF: {str(e)}")

                    if export_apng:
                        apng_filename = os.path.join(intermediate_apng_videos_folder if is_intermediate else apng_videos_folder, f'{job_id}_{total_generated_latent_frames}.png' if is_intermediate else f'{job_id}.png')
                        try:
                            os.makedirs(os.path.dirname(apng_filename), exist_ok=True)
                            save_bcthw_as_apng(history_pixels, apng_filename, fps=fps)
                            print(f"Saved APNG animation to {apng_filename}")

                            # REMOVED Last frame saving for APNG

                        except Exception as e: print(f"Error saving APNG: {str(e)}")

                    if export_webp:
                        webp_filename = os.path.join(intermediate_webp_videos_folder if is_intermediate else webp_videos_folder, f'{job_id}_{total_generated_latent_frames}.webp' if is_intermediate else f'{job_id}.webp')
                        try:
                            os.makedirs(os.path.dirname(webp_filename), exist_ok=True)
                            save_bcthw_as_webp(history_pixels, webp_filename, fps=fps)
                            print(f"Saved WebP animation to {webp_filename}")

                            # REMOVED Last frame saving for WebP

                        except Exception as e: print(f"Error saving WebP: {str(e)}")
                except ConnectionResetError as e:
                    print(f"Connection Reset Error during additional format saving: {str(e)}")
                    print("Continuing with the process anyway...")

                save_time = time.time() - save_start_time
                save_time_history.append(save_time)
                if len(save_time_history) > 0:
                    estimated_save_time = sum(save_time_history) / len(save_time_history)
                print(f"Saving operations completed in {save_time:.2f} seconds")
                print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

                # Yield the original filename to the UI
                stream.output_queue.push(('file', output_filename))

                if is_last_section:
                    break # End of generation loop

            # --- Generation Loop Timing ---
            gen_time_completed = time.time() - gen_start_time # Use gen_start_time for correct duration
            generation_times.append(gen_time_completed)
            avg_gen_time = sum(generation_times) / len(generation_times)
            remaining_gens = num_generations - (gen_idx + 1)
            estimated_remaining_time = avg_gen_time * remaining_gens

            print(f"\nGeneration {gen_idx+1}/{num_generations} completed in {gen_time_completed:.2f} seconds")
            if remaining_gens > 0:
                print(f"Estimated time for remaining generations: {estimated_remaining_time/60:.1f} minutes")
                if using_lora and hasattr(transformer, "peft_config") and transformer.peft_config:
                    print("Cleaning up LoRA weights before next generation")
                    safe_unload_lora(transformer, gpu)

            stream.output_queue.push(('timing', {'gen_time': gen_time_completed, 'avg_time': avg_gen_time, 'remaining_time': estimated_remaining_time}))
            # --- End Generation Loop Timing ---

        except KeyboardInterrupt as e:
            if str(e) == 'User ends the task.':
                print("\n" + "="*50 + "\nGENERATION ENDED BY USER\n" + "="*50)
                if using_lora and hasattr(transformer, "peft_config") and transformer.peft_config:
                    print("Cleaning up LoRA weights after user interruption")
                    safe_unload_lora(transformer, gpu)
                if not high_vram:
                    print("Unloading models from memory...")
                    unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                stream.output_queue.push(('end', None))
                return
            else: raise
        except ConnectionResetError as e:
            print(f"Connection Reset Error outside main processing loop: {str(e)}")
            print("Trying to continue with next generation...")
            if using_lora and hasattr(transformer, "peft_config") and transformer.peft_config:
                print("Cleaning up LoRA weights after connection error")
                safe_unload_lora(transformer, gpu)
            if not high_vram:
                try: unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                except Exception as cleanup_error: print(f"Error during memory cleanup: {str(cleanup_error)}")
            if gen_idx == num_generations - 1:
                stream.output_queue.push(('end', None))
                return
            continue # Continue to next generation
        except Exception as e:
            print("\n" + "="*50 + f"\nERROR DURING GENERATION: {str(e)}\n" + "="*50)
            traceback.print_exc()
            print("="*50)
            if using_lora and hasattr(transformer, "peft_config") and transformer.peft_config:
                print("Cleaning up LoRA weights after generation error")
                safe_unload_lora(transformer, gpu)
            if not high_vram:
                try: unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)
                except Exception as cleanup_error: print(f"Error during memory cleanup after exception: {str(cleanup_error)}")
            if gen_idx == num_generations - 1:
                stream.output_queue.push(('end', None))
                return
            continue # Continue to next generation

    total_time_worker = time.time() - start_time # Use overall worker start time for total duration
    print(f"\nTotal worker time: {total_time_worker:.2f} seconds ({total_time_worker/60:.2f} minutes)")

    if hasattr(transformer, "peft_config") and transformer.peft_config:
        print("Final cleanup of LoRA weights at worker completion")
        safe_unload_lora(transformer, gpu)

    stream.output_queue.push(('final_timing', {'total_time': total_time_worker, 'generation_times': generation_times}))
    stream.output_queue.push(('final_seed', last_used_seed))
    stream.output_queue.push(('end', None))
    return


# Modified process function signature
def process(input_image, end_image, prompt, n_prompt, seed, use_random_seed, num_generations, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality='high', export_gif=False, export_apng=False, export_webp=False, save_metadata=True, resolution="640", fps=30, selected_lora="None", lora_scale=1.0, use_multiline_prompts=False, save_individual_frames=False, save_intermediate_frames=False, save_last_frame=False, rife_enabled=False, rife_multiplier="2x FPS"): # Added RIFE params
    # Removed convert_lora from signature
    global stream
    assert input_image is not None, 'No start input image!' # Changed assertion message

    lora_path = get_lora_path_from_name(selected_lora)
    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True), seed, ''

    # --- MODIFICATION START ---
    # Decide which prompt text to use based on the multi-line flag
    if use_multiline_prompts and prompt.strip():
        # Split prompt by lines, filter empty lines and very short prompts (less than 2 chars)
        prompt_lines = [line.strip() for line in prompt.split('\n')]
        prompt_lines = [line for line in prompt_lines if len(line) >= 2]

        if not prompt_lines:
            # If no valid prompts after filtering, use the original prompt as one line
            prompt_lines = [prompt.strip()]
        print(f"Multi-line enabled: Processing {len(prompt_lines)} prompts individually.")
    else:
        # Use the regular prompt as a single line (timestamp parsing will happen in worker if applicable)
        prompt_lines = [prompt.strip()]
        if not use_multiline_prompts:
             print("Multi-line disabled: Passing full prompt to worker for potential timestamp parsing.")
        else:
             print("Multi-line enabled, but prompt seems empty or invalid, using as single line.")

    total_prompts_or_loops = len(prompt_lines)
    # --- MODIFICATION END ---

    final_video = None

    # Loop through each prompt OR just once if multi-line is disabled
    for prompt_idx, current_prompt_line in enumerate(prompt_lines):
        stream = AsyncStream()

        print(f"Starting processing loop {prompt_idx+1}/{total_prompts_or_loops}")
        status_msg = f"Processing prompt {prompt_idx+1}/{total_prompts_or_loops}" if use_multiline_prompts else "Starting generation"
        yield None, None, status_msg, '', gr.update(interactive=False), gr.update(interactive=True), seed, ''

        # --- MODIFICATION START ---
        # Pass the CURRENT prompt line if multi-line is ON
        # Pass the ORIGINAL full prompt if multi-line is OFF
        prompt_to_worker = prompt if not use_multiline_prompts else current_prompt_line

        # Pass the use_multiline_prompts flag correctly to the worker
        # Also pass the other boolean flags correctly
        # Pass RIFE params
        async_run(worker, input_image, end_image, prompt_to_worker, n_prompt, seed, use_random_seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality, export_gif, export_apng, export_webp, num_generations, resolution, fps, lora_path, lora_scale, # Removed convert_lora
                  save_individual_frames_flag=save_individual_frames, # Pass flags with correct names
                  save_intermediate_frames_flag=save_intermediate_frames,
                  save_last_frame_flag=save_last_frame,
                  use_multiline_prompts_flag=use_multiline_prompts,
                  rife_enabled=rife_enabled, rife_multiplier=rife_multiplier) # Pass RIFE params
        # --- MODIFICATION END ---

        output_filename = None
        webm_filename = None
        gif_filename = None
        apng_filename = None
        webp_filename = None
        current_seed = seed
        timing_info = ""
        last_output = None

        while True:
            flag, data = stream.output_queue.next()

            if flag == 'seed_update':
                current_seed = data
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, timing_info

            if flag == 'final_seed':
                current_seed = data

            if flag == 'timing':
                gen_time = data['gen_time']
                avg_time = data['avg_time']
                remaining_time = data['remaining_time']
                eta_str = f"{remaining_time/60:.1f} minutes" if remaining_time > 60 else f"{remaining_time:.1f} seconds"
                timing_info = f"Last generation: {gen_time:.2f}s | Average: {avg_time:.2f}s | ETA: {eta_str}"
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, timing_info

            if flag == 'final_timing':
                total_time = data['total_time']
                timing_info = f"Total generation time: {total_time:.2f}s ({total_time/60:.2f} min)"
                yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, timing_info

            if flag == 'file':
                output_filename = data
                if output_filename is None:
                    print("Warning: No output file was generated due to an error")
                    yield None, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), current_seed, timing_info
                    continue

                last_output = output_filename
                final_video = output_filename  # Update final video with the latest output
                base_name = os.path.basename(output_filename)
                base_name_no_ext = os.path.splitext(base_name)[0]
                is_intermediate = 'intermediate' in output_filename

                if is_intermediate:
                    webm_filename = os.path.join(intermediate_webm_videos_folder, f"{base_name_no_ext}.webm")
                    gif_filename = os.path.join(intermediate_gif_videos_folder, f"{base_name_no_ext}.gif")
                    apng_filename = os.path.join(intermediate_apng_videos_folder, f"{base_name_no_ext}.png")
                    webp_filename = os.path.join(intermediate_webp_videos_folder, f"{base_name_no_ext}.webp")
                else:
                    webm_filename = os.path.join(webm_videos_folder, f"{base_name_no_ext}.webm")
                    gif_filename = os.path.join(gif_videos_folder, f"{base_name_no_ext}.gif")
                    apng_filename = os.path.join(apng_videos_folder, f"{base_name_no_ext}.png")
                    webp_filename = os.path.join(webp_videos_folder, f"{base_name_no_ext}.webp")

                if not os.path.exists(webm_filename): webm_filename = None
                if not os.path.exists(gif_filename): gif_filename = None
                if not os.path.exists(apng_filename): apng_filename = None
                if not os.path.exists(webp_filename): webp_filename = None

                video_file = output_filename if output_filename is not None else None
                if output_filename is not None and video_quality == 'web_compatible' and webm_filename and os.path.exists(webm_filename):
                    video_file = webm_filename

                prompt_info = f" (Prompt {prompt_idx+1}/{total_prompts_or_loops})" if use_multiline_prompts else ""
                yield video_file, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True), current_seed, timing_info + prompt_info

            if flag == 'rife_file':
                # This is a RIFE-enhanced video file
                rife_video_file = data
                print(f"Displaying RIFE-enhanced video: {rife_video_file}")
                prompt_info = f" (Prompt {prompt_idx+1}/{total_prompts_or_loops})" if use_multiline_prompts else ""
                final_video = rife_video_file  # Update final video with RIFE version
                yield rife_video_file, gr.update(), gr.update(value=f"RIFE-enhanced video ready ({rife_multiplier})"), gr.update(), gr.update(interactive=False), gr.update(interactive=True), current_seed, timing_info + prompt_info

            if flag == 'progress':
                preview, desc, html = data

                # Add prompt info to progress display if using multi-line prompts
                if use_multiline_prompts:
                    if html:
                        # Extract the existing hint from html if possible
                        import re
                        hint_match = re.search(r'>(.*?)<br', html)
                        if hint_match:
                            hint = hint_match.group(1)
                            new_hint = f"{hint} (Prompt {prompt_idx+1}/{total_prompts_or_loops}: {current_prompt_line[:30]}{'...' if len(current_prompt_line) > 30 else ''})"
                            html = html.replace(hint, new_hint)

                    # Add prompt info to the description
                    if desc:
                        desc += f" (Prompt {prompt_idx+1}/{total_prompts_or_loops})"

                # If NOT multi-line, the desc from worker already contains timestamp info if used
                yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True), current_seed, timing_info


            if flag == 'end':
                if prompt_idx == len(prompt_lines) - 1:  # If this is the last prompt loop
                    # Check if there's a RIFE-enhanced version of the final video
                    if rife_enabled and final_video and os.path.exists(os.path.splitext(final_video)[0] + '_extra_FPS.mp4'):
                        rife_final_video = os.path.splitext(final_video)[0] + '_extra_FPS.mp4'
                        print(f"Using RIFE-enhanced final video: {rife_final_video}")
                        yield rife_final_video, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False), current_seed, timing_info
                    else:
                        yield final_video, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False), current_seed, timing_info
                else:
                    # For intermediate prompts (only happens if multi-line is ON)
                    yield final_video, gr.update(visible=False), f"Completed prompt {prompt_idx+1}/{total_prompts_or_loops}", '', gr.update(interactive=False), gr.update(interactive=True), current_seed, timing_info
                break # Exit the inner while loop

        # If multi-line is disabled, we only run the outer loop once
        if not use_multiline_prompts:
            break

    # Only reach this point if all prompts (if multi-line) are processed
    # Check if there's a RIFE-enhanced version of the final video
    if rife_enabled and final_video and os.path.exists(os.path.splitext(final_video)[0] + '_extra_FPS.mp4'):
        rife_final_video = os.path.splitext(final_video)[0] + '_extra_FPS.mp4'
        print(f"Using RIFE-enhanced final video at process end: {rife_final_video}")
        yield rife_final_video, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False), current_seed, timing_info
    else:
        yield final_video, gr.update(visible=False), '', '', gr.update(interactive=True), gr.update(interactive=False), current_seed, timing_info


# Modified batch_process function signature
def batch_process(input_folder, output_folder, batch_end_frame_folder, prompt, n_prompt, seed, use_random_seed, total_second_length,
                  latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold,
                  video_quality='high', export_gif=False, export_apng=False, export_webp=False,
                  skip_existing=True, save_metadata=True, num_generations=1, resolution="640", fps=30,
                  selected_lora="None", lora_scale=1.0, batch_use_multiline_prompts=False, # Removed convert_lora
                  batch_save_individual_frames=False, batch_save_intermediate_frames=False, batch_save_last_frame=False,
                  rife_enabled=False, rife_multiplier="2x FPS"): # Added RIFE params
    global stream
    global batch_stop_requested # Declare intent to use global flag

    # --- Reset stop flag at the beginning of a new batch job ---
    print("Resetting batch stop flag.")
    batch_stop_requested = False
    # --- End Reset ---

    lora_path = get_lora_path_from_name(selected_lora)

    if not input_folder or not os.path.exists(input_folder):
        return None, f"Input folder does not exist: {input_folder}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""

    if not output_folder:
        output_folder = outputs_batch_folder
    else:
        try: os.makedirs(output_folder, exist_ok=True)
        except Exception as e: return None, f"Error creating output folder: {str(e)}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""

    # Check if end frame folder is provided and exists
    use_end_frames = batch_end_frame_folder and os.path.isdir(batch_end_frame_folder)
    if batch_end_frame_folder and not use_end_frames:
         print(f"Warning: End frame folder provided but not found or not a directory: {batch_end_frame_folder}. Proceeding without end frames.")

    image_files = get_images_from_folder(input_folder)
    if not image_files:
        return None, f"No image files found in {input_folder}", "", "", gr.update(interactive=True), gr.update(interactive=False), seed, ""

    yield None, None, f"Found {len(image_files)} images to process. End frames {'enabled' if use_end_frames else 'disabled'}.", "", gr.update(interactive=False), gr.update(interactive=True), seed, ""

    final_output = None
    current_seed = seed

    # --- OUTER BATCH LOOP ---
    for idx, image_path in enumerate(image_files):
        # --- Check stop flag at start of outer loop ---
        if batch_stop_requested:
            print("Batch stop requested. Exiting batch process.")
            yield final_output, gr.update(visible=False), "Batch processing stopped by user.", "", gr.update(interactive=True), gr.update(interactive=False), current_seed, ""
            return # Exit the batch function completely
        # --- End Check ---

        start_image_basename = os.path.basename(image_path)
        output_filename_base = os.path.splitext(start_image_basename)[0]

        # --- MODIFICATION START ---
        # Determine the base prompt text for this image
        current_prompt_text = prompt # Default prompt from UI
        custom_prompt = get_prompt_from_txt_file(image_path)
        if custom_prompt:
            current_prompt_text = custom_prompt
            print(f"Using custom prompt from txt file for {image_path}")

        # Decide if we process lines separately or pass the whole text
        if batch_use_multiline_prompts:
            # Split into lines if multi-line is enabled
            potential_lines = current_prompt_text.split('\n')
            prompt_lines_or_fulltext = [line.strip() for line in potential_lines if line.strip()]
            prompt_lines_or_fulltext = [line for line in prompt_lines_or_fulltext if len(line) >= 2]
            if not prompt_lines_or_fulltext:
                 prompt_lines_or_fulltext = [current_prompt_text.strip()] # Fallback
            print(f"Batch multi-line enabled: Processing {len(prompt_lines_or_fulltext)} prompts for {start_image_basename}")
        else:
            # If multi-line is disabled, use the whole text as a single item list
            # The worker will handle potential timestamp parsing within this text
            prompt_lines_or_fulltext = [current_prompt_text.strip()] # Pass the full text
            print(f"Batch multi-line disabled: Passing full prompt text to worker for {start_image_basename}")

        total_prompts_or_loops = len(prompt_lines_or_fulltext)
        # --- MODIFICATION END ---


        # Skip check logic - simplified for now
        skip_this_image = False
        if skip_existing:
            # Check if the first generation output exists (might need refinement for multi-prompt/multi-gen)
            output_check_base = os.path.join(output_folder, f"{output_filename_base}")
            if batch_use_multiline_prompts:
                output_check_path = f"{output_check_base}_p1{'_g1' if num_generations > 1 else ''}.mp4"
            else: # Single prompt (potentially timestamped)
                 output_check_path = f"{output_check_base}{'_1' if num_generations > 1 else ''}.mp4"
            skip_this_image = os.path.exists(output_check_path)


        if skip_this_image:
            print(f"Skipping {image_path} - output already exists")
            yield None, None, f"Skipping {idx+1}/{len(image_files)}: {start_image_basename} - already processed", "", gr.update(interactive=False), gr.update(interactive=True), seed, ""
            continue

        # Load start image
        try:
            img = Image.open(image_path)
            if img.mode != 'RGB': img = img.convert('RGB')
            input_image = np.array(img)
            if len(input_image.shape) == 2: input_image = np.stack([input_image]*3, axis=2)
            print(f"Loaded start image {image_path} with shape {input_image.shape} and dtype {input_image.dtype}")
        except Exception as e:
            print(f"Error loading start image {image_path}: {str(e)}")
            yield None, None, f"Error processing {idx+1}/{len(image_files)}: {start_image_basename} - {str(e)}", "", gr.update(interactive=False), gr.update(interactive=True), seed, ""
            continue

        # Try to load corresponding end image if folder provided
        current_end_image = None
        end_image_path_str = "None"
        if use_end_frames:
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
            else:
                 print(f"No matching end frame found for {start_image_basename} in {batch_end_frame_folder}")

        # --- MODIFICATION START ---
        # --- INNER PROMPT LOOP (if multi-line enabled) ---
        for prompt_idx, current_prompt_segment in enumerate(prompt_lines_or_fulltext):
            # --- Check stop flag at start of inner loop ---
            if batch_stop_requested:
                print("Batch stop requested during prompt loop. Exiting batch process.")
                yield final_output, gr.update(visible=False), "Batch processing stopped by user.", "", gr.update(interactive=True), gr.update(interactive=False), current_seed, ""
                return # Exit the batch function completely
            # --- End Check ---

            # Reset generation counter for each prompt line/text
            generation_count_for_image = 0
            # Reset current_seed for each prompt line (important to maintain deterministic behavior)
            if use_random_seed:
                current_seed = random.randint(1, 2147483647)
                yield None, None, f"Using new random seed {current_seed} for prompt {prompt_idx+1}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, ""
            elif prompt_idx > 0: # Only increment seed for subsequent prompts if not random
                current_seed += 1 # Should this happen per prompt or per image? Per prompt seems more consistent here.

            prompt_info = ""
            if batch_use_multiline_prompts:
                prompt_info = f" (Prompt {prompt_idx+1}/{total_prompts_or_loops}: {current_prompt_segment[:30]}{'...' if len(current_prompt_segment) > 30 else ''})"
            elif not batch_use_multiline_prompts:
                 # Indicate potential timestamp parsing if multi-line is off
                 prompt_info = " (Processing full text - potential timestamps)"

            yield None, None, f"Processing {idx+1}/{len(image_files)}: {start_image_basename} (End: {os.path.basename(end_image_path_str) if current_end_image is not None else 'No'}) with {num_generations} generation(s){prompt_info}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, ""


            gen_start_time_batch = time.time() # Track start time for metadata

            # Reset stream and run worker
            stream = AsyncStream()

            # Create individual frames folders for batch output if needed
            batch_individual_frames_folder = None
            batch_intermediate_individual_frames_folder = None
            batch_last_frames_folder = None
            batch_intermediate_last_frames_folder = None
            if batch_save_individual_frames or batch_save_intermediate_frames or batch_save_last_frame:
                batch_individual_frames_folder = os.path.join(output_folder, 'individual_frames')
                batch_intermediate_individual_frames_folder = os.path.join(batch_individual_frames_folder, 'intermediate_videos')
                batch_last_frames_folder = os.path.join(output_folder, 'last_frames')
                batch_intermediate_last_frames_folder = os.path.join(batch_last_frames_folder, 'intermediate_videos')
                os.makedirs(batch_individual_frames_folder, exist_ok=True)
                os.makedirs(batch_intermediate_individual_frames_folder, exist_ok=True)
                os.makedirs(batch_last_frames_folder, exist_ok=True)
                os.makedirs(batch_intermediate_last_frames_folder, exist_ok=True)
                print(f"Created frames folders for batch output in: {output_folder}")

            # Custom function for batch worker that overrides the individual_frames_folder path
            def batch_worker_override(*args, **kwargs):
                # Save original paths
                global individual_frames_folder, intermediate_individual_frames_folder, last_frames_folder, intermediate_last_frames_folder

                orig_individual_frames = individual_frames_folder
                orig_intermediate_individual_frames = intermediate_individual_frames_folder
                orig_last_frames = last_frames_folder
                orig_intermediate_last_frames = intermediate_last_frames_folder

                # Override with batch paths if they exist
                if batch_individual_frames_folder:
                    individual_frames_folder = batch_individual_frames_folder
                    intermediate_individual_frames_folder = batch_intermediate_individual_frames_folder
                    last_frames_folder = batch_last_frames_folder
                    intermediate_last_frames_folder = batch_intermediate_last_frames_folder

                try:
                    # Call original worker
                    result = worker(*args, **kwargs)
                    return result
                finally:
                    # Restore original paths
                    if batch_individual_frames_folder:
                        individual_frames_folder = orig_individual_frames
                        intermediate_individual_frames_folder = orig_intermediate_individual_frames
                        last_frames_folder = orig_last_frames
                        intermediate_last_frames_folder = orig_intermediate_last_frames

            # Determine if override is needed
            override_needed = batch_save_individual_frames or batch_save_intermediate_frames or batch_save_last_frame

            # --- Run Worker ---
            # Pass the correct prompt segment and the flag to the worker
            # Pass RIFE params
            async_run(batch_worker_override if override_needed else worker,
                    input_image, current_end_image, current_prompt_segment, n_prompt, current_seed, use_random_seed,
                    total_second_length, latent_window_size, steps, cfg, gs, rs,
                    gpu_memory_preservation, teacache_threshold, video_quality, export_gif, # Removed convert_lora
                    export_apng, export_webp, num_generations=num_generations, resolution=resolution, fps=fps,
                    selected_lora=lora_path, lora_scale=lora_scale,
                    save_individual_frames_flag=batch_save_individual_frames, # Pass flags
                    save_intermediate_frames_flag=batch_save_intermediate_frames,
                    save_last_frame_flag=batch_save_last_frame,
                    use_multiline_prompts_flag=batch_use_multiline_prompts, # Pass the flag
                    rife_enabled=rife_enabled, rife_multiplier=rife_multiplier) # Pass RIFE params


            output_filename = None
            last_output = None
            all_outputs = {}

            # --- WORKER LISTENING LOOP ---
            while True:
                # --- Check stop flag inside worker listening loop (optional but safer) ---
                # This helps if the worker finishes but the user clicked stop just before 'end' flag
                if batch_stop_requested:
                     print("Batch stop requested while waiting for worker. Ending loop.")
                     # We might have already pushed 'end' to worker via end_process,
                     # but breaking here ensures the batch loop stops promptly.
                     break # Exit the 'while True' listening loop
                # --- End Check ---

                flag, data = stream.output_queue.next() # Add timeout to prevent blocking indefinitely if worker hangs

                if flag is None: # Timeout occurred
                    continue

                if flag == 'seed_update':
                    # Update the seed for the *next* potential generation within this prompt loop
                    current_seed = data
                    yield gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), current_seed, gr.update()

                if flag == 'final_seed':
                     # This seed was the last one used for the *previous* completed generation
                     # We might want to use the seed from 'seed_update' if that's more relevant for the *next* step
                     pass # Maybe ignore final_seed here, current_seed should be correct

                if flag == 'file':
                    output_filename = data
                    if output_filename:
                        is_intermediate = 'intermediate' in output_filename
                        if not is_intermediate:
                            generation_count_for_image += 1 # Increment only for final outputs

                            # Determine suffix based on multi-line and generation count
                            if batch_use_multiline_prompts:
                                suffix = f"_p{prompt_idx+1}" + (f"_g{generation_count_for_image}" if num_generations > 1 else "")
                            else: # Single prompt text (potentially timestamped)
                                suffix = f"_{generation_count_for_image}" if num_generations > 1 else ""

                            modified_image_filename_base = f"{output_filename_base}{suffix}"

                            # Move MP4
                            moved_file = move_and_rename_output_file(output_filename, output_folder, f"{modified_image_filename_base}.mp4")
                            if moved_file:
                                output_key = f'mp4_{generation_count_for_image}' if num_generations > 1 else 'mp4'
                                all_outputs[output_key] = moved_file
                                last_output = moved_file
                                final_output = moved_file

                                prompt_status = f" (Prompt {prompt_idx+1}/{total_prompts_or_loops})" if batch_use_multiline_prompts else ""
                                yield last_output, gr.update(visible=False), f"Processing {idx+1}/{len(image_files)}: {start_image_basename} - Generated video {generation_count_for_image}/{num_generations}{prompt_status}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()

                                # Save metadata for the original MP4
                                if save_metadata:
                                    gen_time_batch_current = time.time() - gen_start_time_batch # Use batch start time for this image/prompt combo
                                    generation_time_seconds = int(gen_time_batch_current)
                                    generation_time_formatted = format_time_human_readable(gen_time_batch_current)

                                    metadata = {
                                        "Prompt": current_prompt_segment, # Save the specific prompt/text used
                                        "Seed": current_seed, # Use the seed that was just used for this generation
                                        "TeaCache": f"Enabled (Threshold: {teacache_threshold})" if teacache_threshold > 0 else "Disabled",
                                        "Video Length (seconds)": total_second_length,
                                        "FPS": fps,
                                        "Latent Window Size": latent_window_size, # <-- ADDED
                                        "Steps": steps,
                                        "CFG Scale": cfg, # Add explicit CFG Scale
                                        "Distilled CFG Scale": gs,
                                        "Guidance Rescale": rs, # Add Guidance Rescale
                                        "Resolution": resolution,
                                        "Generation Time": generation_time_formatted,
                                        "Total Seconds": f"{generation_time_seconds} seconds",
                                        "Start Frame": image_path,
                                        "End Frame": end_image_path_str if current_end_image is not None else "None",
                                        "Multi-line Prompts Mode": batch_use_multiline_prompts, # Record mode
                                    }
                                    # Indicate if timestamps might have been parsed (only possible if multi-line is off)
                                    if not batch_use_multiline_prompts:
                                         metadata["Timestamped Prompts Parsed"] = "[Check Worker Logs]" # Indicate potential use

                                    if batch_use_multiline_prompts:
                                        metadata["Prompt Number"] = f"{prompt_idx+1}/{total_prompts_or_loops}"

                                    if lora_path != "none":
                                        lora_name = os.path.basename(lora_path)
                                        metadata["LoRA"] = lora_name
                                        metadata["LoRA Scale"] = lora_scale
                                    save_processing_metadata(moved_file, metadata)

                                # Handle other formats (no change for RIFE here)
                                if export_gif:
                                    gif_filename = os.path.splitext(output_filename)[0] + '.gif'
                                    moved_gif = move_and_rename_output_file(gif_filename, output_folder, f"{modified_image_filename_base}.gif")
                                    if moved_gif and save_metadata: save_processing_metadata(moved_gif, metadata)
                                if export_apng:
                                    apng_filename = os.path.splitext(output_filename)[0] + '.png'
                                    moved_apng = move_and_rename_output_file(apng_filename, output_folder, f"{modified_image_filename_base}.png")
                                    if moved_apng and save_metadata: save_processing_metadata(moved_apng, metadata)
                                if export_webp:
                                    webp_filename = os.path.splitext(output_filename)[0] + '.webp'
                                    moved_webp = move_and_rename_output_file(webp_filename, output_folder, f"{modified_image_filename_base}.webp")
                                    if moved_webp and save_metadata: save_processing_metadata(moved_webp, metadata)
                                if video_quality == 'web_compatible':
                                    webm_filename = os.path.splitext(output_filename)[0] + '.webm'
                                    moved_webm = move_and_rename_output_file(webm_filename, output_folder, f"{modified_image_filename_base}.webm")
                                    if moved_webm:
                                        if save_metadata: save_processing_metadata(moved_webm, metadata)
                                        last_output = moved_webm
                                        final_output = moved_webm

                                # Move the RIFE-enhanced file if it exists (copy, don't rename to base)
                                rife_original_path = os.path.splitext(output_filename)[0] + '_extra_FPS.mp4'
                                if os.path.exists(rife_original_path):
                                    # Create a filename like 'base_p1_g1_extra_FPS.mp4'
                                    rife_target_filename = f"{modified_image_filename_base}_extra_FPS.mp4"
                                    rife_target_path = os.path.join(output_folder, rife_target_filename)
                                    try:
                                        shutil.copy2(rife_original_path, rife_target_path)
                                        print(f"Copied RIFE enhanced file to batch outputs: {rife_target_path}")
                                        # Use the RIFE-enhanced video as the output if RIFE is enabled
                                        if rife_enabled:
                                            last_output = rife_target_path
                                            final_output = rife_target_path
                                            print(f"Using RIFE-enhanced video as the display output: {rife_target_path}")
                                    except Exception as e:
                                        print(f"Error copying RIFE enhanced file to {rife_target_path}: {str(e)}")


                        else: # Intermediate file
                            prompt_status = f" (Prompt {prompt_idx+1}/{total_prompts_or_loops})" if batch_use_multiline_prompts else ""
                            yield output_filename, gr.update(visible=False), f"Processing {idx+1}/{len(image_files)}: {start_image_basename} - Generating intermediate result{prompt_status}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()


                if flag == 'progress':
                    preview, desc, html = data # Desc now comes from worker potentially with prompt info
                    current_progress = f"Processing {idx+1}/{len(image_files)}: {start_image_basename}"

                    # Add prompt info to the progress text if using multi-line prompts
                    if batch_use_multiline_prompts:
                        prompt_status = f" (Prompt {prompt_idx+1}/{total_prompts_or_loops})"
                        current_progress += prompt_status

                    # If not multi-line, desc from worker already contains timestamp info if used
                    if desc: current_progress += f" - {desc}"

                    progress_html = html if html else make_progress_bar_html(0, f"Processing file {idx+1} of {len(image_files)}")

                    # Add prompt info to HTML if using multi-line prompts
                    if batch_use_multiline_prompts and html:
                        import re
                        hint_match = re.search(r'>(.*?)<br', html)
                        if hint_match:
                            hint = hint_match.group(1)
                            new_hint = f"{hint} (Prompt {prompt_idx+1}/{total_prompts_or_loops})"
                            progress_html = html.replace(hint, new_hint)

                    video_update = last_output if last_output else gr.update()
                    yield video_update, gr.update(visible=True, value=preview), current_progress, progress_html, gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()


                if flag == 'end':
                    video_update = last_output if last_output else gr.update()

                    if prompt_idx == len(prompt_lines_or_fulltext) - 1:  # If this is the last prompt loop for this image
                        yield video_update, gr.update(visible=False), f"Completed {idx+1}/{len(image_files)}: {start_image_basename}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()
                    else: # Only happens if batch_use_multiline_prompts is True
                        prompt_status = f" (Completed prompt {prompt_idx+1}/{total_prompts_or_loops}, continuing to next prompt)"
                        yield video_update, gr.update(visible=False), f"Processing {idx+1}/{len(image_files)}: {start_image_basename}{prompt_status}", "", gr.update(interactive=False), gr.update(interactive=True), current_seed, gr.update()
                    break # End loop for this prompt/text
            # --- END WORKER LISTENING LOOP ---

            # --- Check stop flag AGAIN after worker loop finishes ---
            # This catches the case where stop was requested *during* the last worker processing
            if batch_stop_requested:
                print("Batch stop requested after worker finished. Exiting batch process.")
                yield final_output, gr.update(visible=False), "Batch processing stopped by user.", "", gr.update(interactive=True), gr.update(interactive=False), current_seed, ""
                return # Exit the batch function completely
            # --- End Check ---

            # If multi-line is disabled, we break after the first loop iteration
            if not batch_use_multiline_prompts:
                 break
        # --- END INNER PROMPT LOOP ---

        # --- Check stop flag after inner loop (redundant if checked inside, but safe) ---
        if batch_stop_requested:
            print("Batch stop requested after inner loop. Exiting batch process.")
            yield final_output, gr.update(visible=False), "Batch processing stopped by user.", "", gr.update(interactive=True), gr.update(interactive=False), current_seed, ""
            return # Exit the batch function completely
        # --- End Check ---
    # --- END OUTER BATCH LOOP ---

    # Final yield if loop completes normally
    if not batch_stop_requested:
        yield final_output, gr.update(visible=False), f"Batch processing complete. Processed {len(image_files)} images.", "", gr.update(interactive=True), gr.update(interactive=False), current_seed, ""
    else:
        # Ensure buttons are reset even if stopped at the very end
         yield final_output, gr.update(visible=False), "Batch processing stopped by user.", "", gr.update(interactive=True), gr.update(interactive=False), current_seed, ""


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
    # Make Start buttons interactive again, End buttons non-interactive
    updates = [gr.update(interactive=True), gr.update(interactive=False), gr.update(interactive=True), gr.update(interactive=False)]
    return updates


quick_prompts = [
    'A character doing some simple body movements.','A talking man.',
    '[0] A person stands still\n[2] The person waves hello\n[4] The person claps',
    '[0] close up shot, cinematic\n[3] medium shot, cinematic\n[5] wide angle shot, cinematic'
]
quick_prompts = [[x] for x in quick_prompts]

# --- Function to auto-set Latent Window Size ---
def auto_set_window_size(fps_val: int, current_lws: int):
    """Calculates Latent Window Size for ~1 second sections."""
    if not isinstance(fps_val, int) or fps_val <= 0:
        print("Invalid FPS for auto window size calculation.")
        return gr.update() # No change if FPS is invalid

    try:
        # Calculate the ideal float value for LWS to get exactly 1s sections
        # section_duration = (LWS * 4) / fps = 1.0  => LWS = fps / 4.0
        ideal_lws_float = fps_val / 4.0

        # Round to the nearest integer, as LWS must be integer
        target_lws = round(ideal_lws_float)

        # Get min/max from the actual slider component (safer)
        min_lws = 1
        max_lws = 33 # Hardcoded based on slider definition in UI

        # Clamp the value within the slider's range
        calculated_lws = max(min_lws, min(target_lws, max_lws))

        # Calculate the actual duration this LWS will give
        resulting_duration = (calculated_lws * 4) / fps_val

        print(f"Auto-setting LWS: Ideal float LWS for 1s sections={ideal_lws_float:.2f}, Rounded integer LWS={target_lws}, Clamped LWS={calculated_lws}")
        print(f"--> Resulting section duration with LWS={calculated_lws} at {fps_val} FPS will be: {resulting_duration:.3f} seconds")

        # Provide feedback on exactness
        if abs(resulting_duration - 1.0) < 0.01: # Allow small tolerance
            print("This setting provides (near) exact 1-second sections.")
        else:
            print(f"Note: This is the closest integer LWS to achieve 1-second sections.")


        # Only update if the value is different
        if calculated_lws != current_lws:
            return gr.update(value=calculated_lws)
        else:
            # If the value is already correct, don't trigger an unnecessary update loop
            print(f"Latent Window Size is already optimal ({current_lws}) for ~1s sections.")
            return gr.update() # No change needed

    except Exception as e:
        print(f"Error calculating auto window size: {e}")
        traceback.print_exc() # Add traceback
        return gr.update() # No change on error
# --- End Auto-set Function ---


css = make_progress_bar_css()
block = gr.Blocks(css=css).queue()
with block:
    gr.Markdown('# FramePack Improved SECourses App V42 - https://www.patreon.com/posts/126855226')
    with gr.Row():
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Single Image"):
                    # Modified Image Input Section
                    with gr.Row():
                        with gr.Column():
                            input_image = gr.Image(sources='upload', type="numpy", label="Start Frame", height=320)
                        with gr.Column():
                            end_image = gr.Image(sources='upload', type="numpy", label="End Frame (Optional)", height=320) # Added end_image

                    # --- ADDED Iteration Info Display + Button ---
                    with gr.Row():
                        iteration_info_display = gr.Markdown("Calculating generation info...", elem_id="iteration-info-display") # Give more space
                        auto_set_lws_button = gr.Button(value="Set Window for ~1s Sections", scale=1) # Add button
                    # --- END ADDED ---

                    prompt = gr.Textbox(label="Prompt", value='', lines=4, info="Use '[seconds] prompt' format on new lines ONLY when 'Use Multi-line Prompts' is OFF. Example [0] starts second 0, [2] starts after 2 seconds passed and so on") # Changed lines to 4
                    with gr.Row():
                        use_multiline_prompts = gr.Checkbox(label="Use Multi-line Prompts", value=False, info="ON: Each line is a separate gen. OFF: Try parsing '[secs] prompt' format.")
                        latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True, info="Controls generation chunks. Affects section count and duration (see info above prompt).") # Added info here
                    example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                    example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

                    with gr.Row():
                        save_metadata = gr.Checkbox(label="Save Processing Metadata", value=True, info="Save processing parameters in a text file alongside each video")
                        save_individual_frames = gr.Checkbox(label="Save Individual Frames", value=False, info="Save each frame of the final video as an individual image")
                        save_intermediate_frames = gr.Checkbox(label="Save Intermediate Frames", value=False, info="Save each frame of intermediate videos as individual images")
                        save_last_frame = gr.Checkbox(label="Save Last Frame Of Generations (MP4 Only)", value=False, info="Save only the last frame of each MP4 generation to the last_frames folder") # Renamed variable & updated info

                    with gr.Row():
                        start_button = gr.Button(value="Start Generation", variant='primary')
                        end_button = gr.Button(value="End Generation", interactive=False)

                with gr.Tab("Batch Processing"):
                    batch_input_folder = gr.Textbox(label="Input Folder Path (Start Frames)", info="Folder containing starting images to process")
                    # Added End Frame Folder for Batch
                    batch_end_frame_folder = gr.Textbox(label="End Frame Folder Path (Optional)", info="Folder containing matching end frames (same filename as start frame)")
                    batch_output_folder = gr.Textbox(label="Output Folder Path (optional)", info="Leave empty to use the default batch_outputs folder")
                    batch_prompt = gr.Textbox(label="Default Prompt", value='', lines=4, info="Used if no matching .txt file exists. Use '[seconds] prompt' format on new lines ONLY when 'Use Multi-line Prompts' is OFF.") # Changed lines to 4

                    with gr.Row():
                        batch_skip_existing = gr.Checkbox(label="Skip Existing Files", value=True, info="Skip files that already exist in the output folder")
                        batch_save_metadata = gr.Checkbox(label="Save Processing Metadata", value=True, info="Save processing parameters in a text file alongside each video")
                        batch_use_multiline_prompts = gr.Checkbox(label="Use Multi-line Prompts", value=False, info="ON: Each line in prompt/.txt is a separate gen. OFF: Try parsing '[secs] prompt' format from full prompt/.txt.")

                    with gr.Row():
                        batch_save_individual_frames = gr.Checkbox(label="Save Individual Frames", value=False, info="Save each frame of the final video as an individual image")
                        batch_save_intermediate_frames = gr.Checkbox(label="Save Intermediate Frames", value=False, info="Save each frame of intermediate videos as individual images")
                        batch_save_last_frame = gr.Checkbox(label="Save Last Frame Of Generations (MP4 Only)", value=False, info="Save only the last frame of each MP4 generation to the last_frames folder") # Renamed variable & updated info

                    with gr.Row():
                        batch_start_button = gr.Button(value="Start Batch Processing", variant='primary')
                        batch_end_button = gr.Button(value="End Processing", interactive=False)

                    with gr.Row():
                        open_batch_input_folder = gr.Button(value="Open Start Folder")
                        # Added button for end frame folder
                        open_batch_end_folder = gr.Button(value="Open End Folder")
                        open_batch_output_folder = gr.Button(value="Open Output Folder")


            with gr.Group():
                with gr.Row():
                    num_generations = gr.Slider(label="Number of Generations", minimum=1, maximum=50, value=1, step=1, info="Generate multiple videos in sequence (per prompt if multi-line is ON)")
                    resolution = gr.Dropdown(label="Resolution", choices=["1440","1320","1200","1080","960","840","720", "640", "480", "320", "240"], value="640", info="Output Resolution (bigger than 640 set more Preserved Memory)")

                with gr.Row():
                    teacache_threshold = gr.Slider(label='TeaCache Threshold', minimum=0.0, maximum=0.5, value=0.15, step=0.01, info='0 = Disabled, 0.15 = Default. Higher values = more caching but potentially less detail.')
                    seed = gr.Number(label="Seed", value=31337, precision=0)
                    use_random_seed = gr.Checkbox(label="Random Seed", value=True, info="Use random seeds instead of incrementing")

                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=True) # Made visible for preset saving/loading

                with gr.Row():
                    fps = gr.Slider(label="FPS", minimum=10, maximum=60, value=30, step=1, info="Output Videos FPS - Directly changes how many frames are generated, 60 will make double frames")
                    total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)



                with gr.Row():
                    steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                    gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')

                with gr.Row():
                    cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=True, info='You need more than CFG 1 for negative prompts')
                    rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=True)

                gr.Markdown("### LoRA Settings")
                with gr.Row():
                    with gr.Column():
                        lora_options = scan_lora_files()
                        selected_lora = gr.Dropdown(label="Select LoRA", choices=[name for name, _ in lora_options], value="None", info="Select a LoRA to apply")
                    with gr.Column():
                        with gr.Row():
                             lora_refresh_btn = gr.Button(value="🔄 Refresh", scale=1)
                             lora_folder_btn = gr.Button(value="📁 Open Folder", scale=1)
                        lora_scale = gr.Slider(label="LoRA Scale", minimum=0.0, maximum=2.0, value=1.0, step=0.01, info="Adjust the strength of the LoRA effect (0-2)")

                with gr.Row():
                    gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=0, maximum=128, value=8, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                    # --- Start of Resolution -> Memory Update ---
                    def update_memory_for_resolution(res):
                        if res == "1440": return 23
                        if res == "1320": return 21
                        if res == "1200": return 19
                        if res == "1080": return 16
                        elif res == "960": return 14
                        elif res == "840": return 12
                        elif res == "720": return 10
                        elif res == "640": return 8
                        else: return 6
                    resolution.change(fn=update_memory_for_resolution, inputs=resolution, outputs=gpu_memory_preservation)
                    # --- End of Resolution -> Memory Update ---

        with gr.Column(): # Right column for preview/results
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=True, height=512, loop=True)
            video_info = gr.HTML("<div id='video-info'>Generate a video to see information</div>")
            # Updated Markdown Text
            gr.Markdown('''
            Note on Sampling: Due to inverted sampling, the end part of the video is generated first, and the start part last.
            - **Start Frame Only:** If the start action isn't in the video initially, wait for the full generation.
            - **Start and End Frames:** The model attempts a smooth transition. The end frame's influence appears early in the generation process.
            - **Timestamp Prompts (Multi-line OFF):** Prompts like `[2] wave hello` trigger *after* 2 seconds have passed in the *final* video (meaning they are generated earlier in the process). Use the **Generation Info** (duration per section) above for timing estimates.
            ''') # Added pointer to Generation Info
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            timing_display = gr.Markdown("", label="Time Information", elem_classes='no-generating-animation')

            # --- Preset UI Section START --- (Inserted BEFORE Video Quality)
            gr.Markdown("### Presets")
            with gr.Row():
                preset_dropdown = gr.Dropdown(label="Select Preset", choices=scan_presets(), value=load_last_used_preset_name() or "Default")
                preset_load_button = gr.Button(value="Load Preset")
                preset_refresh_button = gr.Button(value="🔄 Refresh")
            with gr.Row():
                preset_save_name = gr.Textbox(label="Save Preset As", placeholder="Enter preset name...")
                preset_save_button = gr.Button(value="Save Current Settings")
            preset_status_display = gr.Markdown("") # To show feedback
            # --- Preset UI Section END ---

            gr.Markdown("### Folder Options")
            with gr.Row():
                open_outputs_btn = gr.Button(value="Open Generations Folder")
                open_batch_outputs_btn = gr.Button(value="Open Batch Outputs Folder")

            video_quality = gr.Radio( # <-- This comes AFTER the preset UI
                label="Video Quality",
                choices=["high", "medium", "low", "web_compatible"],
                value="high",
                info="High: Best quality, Medium: Balanced, Low: Smallest file size, Web Compatible: Best browser compatibility"
            )

            # --- Start of RIFE UI Addition ---
            gr.Markdown("### RIFE Frame Interpolation (MP4 Only)")
            with gr.Row():
                rife_enabled = gr.Checkbox(label="Enable RIFE (2x/4x FPS)", value=False, info="Increases FPS of generated MP4s using RIFE. Saves as '[filename]_extra_FPS.mp4'")
                rife_multiplier = gr.Radio(choices=["2x FPS", "4x FPS"], label="RIFE FPS Multiplier", value="2x FPS", info="Choose the frame rate multiplication factor.")
            # --- End of RIFE UI Addition ---

            gr.Markdown("### Additional Export Formats")
            gr.Markdown("Select additional formats to export alongside MP4:")
            with gr.Row():
                export_gif = gr.Checkbox(label="Export as GIF", value=False, info="Save animation as GIF (larger file size but widely compatible)")
                export_apng = gr.Checkbox(label="Export as APNG", value=False, info="Save animation as Animated PNG (better quality than GIF but less compatible)")
                export_webp = gr.Checkbox(label="Export as WebP", value=False, info="Save animation as WebP (good balance of quality and file size)")


    # --- LIST OF COMPONENTS FOR PRESETS --- (Added for Presets)
    # Define this list AFTER all relevant components have been created in the UI definition
    # IMPORTANT: The order here MUST match the order in save_preset inputs and load_preset outputs!
    # Exclude prompts and batch folders. Include n_prompt.
    preset_components_list = [
        use_multiline_prompts,          # Checkbox
        save_metadata,                  # Checkbox
        save_individual_frames,         # Checkbox
        save_intermediate_frames,       # Checkbox
        save_last_frame,                # Checkbox
        batch_skip_existing,            # Checkbox
        batch_save_metadata,            # Checkbox
        batch_use_multiline_prompts,    # Checkbox
        batch_save_individual_frames,   # Checkbox
        batch_save_intermediate_frames, # Checkbox
        batch_save_last_frame,          # Checkbox
        num_generations,                # Slider (int)
        resolution,                     # Dropdown (str)
        teacache_threshold,             # Slider (float)
        seed,                           # Number (int) - Note: Loading seed might often be overridden by 'Random Seed' checkbox
        use_random_seed,                # Checkbox
        n_prompt,                       # Textbox (str)
        fps,                            # Slider (int)
        total_second_length,            # Slider (float)
        latent_window_size,             # Slider (int) # <-- Already included
        steps,                          # Slider (int)
        gs,                             # Slider (float)
        cfg,                            # Slider (float)
        rs,                             # Slider (float)
        selected_lora,                  # Dropdown (str - name)
        lora_scale,                     # Slider (float)
        # convert_lora removed
        gpu_memory_preservation,        # Slider (float)
        video_quality,                  # Radio (str)
        rife_enabled,                   # Checkbox
        rife_multiplier,                # Radio (str)
        export_gif,                     # Checkbox
        export_apng,                    # Checkbox
        export_webp                     # Checkbox
    ]
    # Give components names/ids for easier debugging if needed (optional)
    component_names_for_preset = [
        "use_multiline_prompts", "save_metadata", "save_individual_frames", "save_intermediate_frames", "save_last_frame",
        "batch_skip_existing", "batch_save_metadata", "batch_use_multiline_prompts", "batch_save_individual_frames", "batch_save_intermediate_frames", "batch_save_last_frame",
        "num_generations", "resolution", "teacache_threshold", "seed", "use_random_seed", "n_prompt", "fps", "total_second_length",
        "latent_window_size", "steps", "gs", "cfg", "rs", "selected_lora", "lora_scale",
        # convert_lora removed
        "gpu_memory_preservation", "video_quality", "rife_enabled", "rife_multiplier", "export_gif", "export_apng", "export_webp"
    ]
    # --------------------------------------


    # --- Preset Action Functions START --- (Added for Presets)

    def save_preset_action(name: str, *values):
        """Saves the current settings (*values) to a preset file."""
        if not name:
            return gr.update(), gr.update(value="Preset name cannot be empty.") # Update dropdown, update status

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
            save_last_used_preset_name(name) # Remember this as last used
            presets = scan_presets()
            status_msg = f"Preset '{name}' saved successfully."
            print(status_msg)
            # Refresh dropdown and select the newly saved preset
            return gr.update(choices=presets, value=name), gr.update(value=status_msg)
        except Exception as e:
            error_msg = f"Error saving preset '{name}': {e}"
            print(error_msg)
            return gr.update(), gr.update(value=error_msg) # Update status only

    def load_preset_action(name: str):
        """Loads settings from a preset file and updates the UI components."""
        preset_data = load_preset_data(name)
        if preset_data is None:
             # Keep current values if loading fails, just show error
             return [gr.update() for _ in preset_components_list] + [gr.update(value=f"Failed to load preset '{name}'.")]

        # Prepare the list of updates
        updates = []
        found_lora = False
        available_loras = [lora_name for lora_name, _ in scan_lora_files()] # Get current LoRA names
        loaded_values = {} # Store loaded values for iteration info update

        for i, comp_name in enumerate(component_names_for_preset):
            comp_initial_value = getattr(preset_components_list[i], 'value', None) # Get component default
            if comp_name in preset_data:
                value = preset_data[comp_name]
                # Special handling for LoRA dropdown: check if the saved LoRA still exists
                if comp_name == "selected_lora":
                    if value in available_loras:
                        updates.append(gr.update(value=value))
                        found_lora = True
                    else:
                        print(f"Warning: Saved LoRA '{value}' not found in current LoRA list. Setting to 'None'.")
                        value = "None" # Update value before adding update
                        updates.append(gr.update(value="None")) # Default to None if not found
                else:
                    updates.append(gr.update(value=value))
                loaded_values[comp_name] = value # Store the value that will be used
            else:
                # If key exists in older preset but not current list (like convert_lora), it's ignored here
                # If key is missing from preset file but exists in current list, use default
                if comp_name not in preset_data:
                     print(f"Warning: Key '{comp_name}' not found in preset '{name}'. Using component's current/default value.")
                     updates.append(gr.update()) # No change for missing keys
                     loaded_values[comp_name] = getattr(preset_components_list[i], 'value', None) # Store current/default value

        # Check if the number of updates matches the component list (should always match now)
        if len(updates) != len(preset_components_list):
             print(f"Error: Number of updates ({len(updates)}) does not match number of components ({len(preset_components_list)}).")
             # Return no updates on critical error
             return [gr.update() for _ in preset_components_list] + [gr.update(value=f"Error applying preset '{name}'. Mismatch in component count.")] + [gr.update()] # Add update for info display


        save_last_used_preset_name(name) # Remember this as last used
        status_msg = f"Preset '{name}' loaded."
        print(status_msg)

        # Calculate iteration info based on loaded values
        vid_len = loaded_values.get('total_second_length', 5)
        fps_val = loaded_values.get('fps', 30)
        win_size = loaded_values.get('latent_window_size', 9)
        info_text = update_iteration_info(vid_len, fps_val, win_size)

        # Return updates for all components + status + iteration info display
        return updates + [gr.update(value=status_msg)] + [gr.update(value=info_text)]


    def refresh_presets_action():
        """Refreshes the preset dropdown list."""
        presets = scan_presets()
        last_used = load_last_used_preset_name()
        selected = last_used if last_used in presets else "Default"
        return gr.update(choices=presets, value=selected)

    # --- Preset Action Functions END ---


    # --- Gradio Event Wiring START ---
    lora_refresh_btn.click(fn=refresh_loras, outputs=[selected_lora])
    lora_folder_btn.click(fn=lambda: open_folder(loras_folder), outputs=[gr.Text(visible=False)])

    # Update inputs list for single image processing - match names with process function
    ips = [input_image, end_image, prompt, n_prompt, seed, use_random_seed, num_generations, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, teacache_threshold, video_quality, export_gif, export_apng, export_webp, save_metadata, resolution, fps, selected_lora, lora_scale, use_multiline_prompts, save_individual_frames, save_intermediate_frames, save_last_frame, rife_enabled, rife_multiplier] # Removed convert_lora
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button, seed, timing_display])
    # End button needs to update both sets of start/end buttons
    end_button.click(fn=end_process, outputs=[start_button, end_button, batch_start_button, batch_end_button])

    open_outputs_btn.click(fn=lambda: open_folder(outputs_folder), outputs=gr.Text(visible=False))
    open_batch_outputs_btn.click(fn=lambda: open_folder(outputs_batch_folder), outputs=gr.Text(visible=False))

    # Connect batch folder buttons
    batch_folder_status_text = gr.Text(visible=False) # Shared status text for folder opening
    open_batch_input_folder.click(fn=lambda x: open_folder(x) if x else "No input folder specified", inputs=[batch_input_folder], outputs=[batch_folder_status_text])
    open_batch_end_folder.click(fn=lambda x: open_folder(x) if x else "No end frame folder specified", inputs=[batch_end_frame_folder], outputs=[batch_folder_status_text]) # Added for end frame folder
    open_batch_output_folder.click(fn=lambda x: open_folder(x if x else outputs_batch_folder), inputs=[batch_output_folder], outputs=[batch_folder_status_text])

    # Update inputs list for batch processing - match names with batch_process function
    batch_ips = [batch_input_folder, batch_output_folder, batch_end_frame_folder, batch_prompt, n_prompt, seed, use_random_seed,
                total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation,
                teacache_threshold, video_quality, export_gif, export_apng, export_webp, batch_skip_existing, # Removed convert_lora
                batch_save_metadata, num_generations, resolution, fps, selected_lora, lora_scale, batch_use_multiline_prompts,
                batch_save_individual_frames, batch_save_intermediate_frames, batch_save_last_frame,
                rife_enabled, rife_multiplier] # Added RIFE UI components
    batch_start_button.click(fn=batch_process, inputs=batch_ips, outputs=[result_video, preview_image, progress_desc, progress_bar, batch_start_button, batch_end_button, seed, timing_display])
    # End button needs to update both sets of start/end buttons
    batch_end_button.click(fn=end_process, outputs=[start_button, end_button, batch_start_button, batch_end_button])

    # --- Preset Event Wiring START --- (Added for Presets)
    preset_save_button.click(
        fn=save_preset_action,
        inputs=[preset_save_name] + preset_components_list, # Pass name and all component values
        outputs=[preset_dropdown, preset_status_display] # Update dropdown and status
    )

    preset_load_button.click(
        fn=load_preset_action,
        inputs=[preset_dropdown], # Pass selected preset name
        outputs=preset_components_list + [preset_status_display] + [iteration_info_display] # Update ALL components + status + info display
    )

    preset_refresh_button.click(
        fn=refresh_presets_action,
        inputs=[],
        outputs=[preset_dropdown] # Update dropdown
    )
    # --- Preset Event Wiring END ---

    # --- Auto Set Latent Window Size Button Wiring ---
    auto_set_lws_button.click(
        fn=auto_set_window_size,
        inputs=[fps, latent_window_size], # Pass current FPS and LWS
        outputs=[latent_window_size]      # Update the LWS slider
    )
    # --- End Auto Set Wiring ---

    # --- Change Listeners for Iteration Info ---
    # (fps and latent_window_size changes trigger this)
    iteration_info_inputs = [total_second_length, fps, latent_window_size]
    for comp in iteration_info_inputs:
        comp.change(
            fn=update_iteration_info,
            inputs=iteration_info_inputs,
            outputs=iteration_info_display,
            queue=False # No need to queue this simple update
        )
    # --- END Iteration Info Listeners ---

    # --- Gradio Event Wiring END ---


    video_info_js = """
    function updateVideoInfo() {
        // Select the video element within the specific Gradio output component
        const videoResultDiv = document.querySelector('#result_video'); // Assuming default ID or add one
        if (!videoResultDiv) return;
        const videoElement = videoResultDiv.querySelector('video');

        if (videoElement) {
            const infoDiv = document.getElementById('video-info');
            if (!infoDiv) return;

            // Function to update info, called on loadedmetadata or if already loaded
            const displayInfo = () => {
                if (videoElement.videoWidth && videoElement.videoHeight && videoElement.duration) {
                     const format = videoElement.currentSrc ? videoElement.currentSrc.split('.').pop().toUpperCase() : 'N/A';
                     infoDiv.innerHTML = `<p>Resolution: ${videoElement.videoWidth}x${videoElement.videoHeight} | Duration: ${videoElement.duration.toFixed(2)}s | Format: ${format}</p>`;
                } else {
                     infoDiv.innerHTML = '<p>Loading video info...</p>';
                }
            };

            // Check if metadata is already loaded
            if (videoElement.readyState >= 1) { // HAVE_METADATA or higher
                displayInfo();
            } else {
                // Add event listener if not loaded yet
                videoElement.removeEventListener('loadedmetadata', displayInfo); // Remove previous listeners
                videoElement.addEventListener('loadedmetadata', displayInfo);
            }
        } else {
             const infoDiv = document.getElementById('video-info');
             if (infoDiv) infoDiv.innerHTML = "<div>Generate a video to see information</div>";
        }
    }

    // Use MutationObserver to detect when the video component updates
    const observerCallback = function(mutationsList, observer) {
        for(const mutation of mutationsList) {
            if (mutation.type === 'childList') {
                 // Check if the video element or its container was added/changed
                 const videoResultDiv = document.querySelector('#result_video'); // Re-select if needed
                 if (videoResultDiv && (mutation.target === videoResultDiv || videoResultDiv.contains(mutation.target))) {
                    updateVideoInfo();
                    break; // No need to check other mutations if we found the relevant one
                 }
            }
        }
    };

    // Observe the body for changes, adjust target if needed for performance
    const observer = new MutationObserver(observerCallback);
    observer.observe(document.body, { childList: true, subtree: true });

    // Also run on initial load
    if (document.readyState === 'complete') {
      updateVideoInfo();
    } else {
      document.addEventListener('DOMContentLoaded', updateVideoInfo);
    }
    """
    # Add ID to the result_video component for easier JS selection
    result_video.elem_id = "result_video"

    # --- Startup Loading START (Combined Preset & Iteration Info) ---
    # This function will run once when the Gradio app loads
    def apply_preset_and_init_info_on_startup():
        print("Applying preset and initializing info on startup...")
        # 1. Get current default values from UI definition to create Default.json if needed
        initial_values = {}
        for i, comp in enumerate(preset_components_list):
             # Use getattr to safely access the 'value' attribute if it exists
             default_value = getattr(comp, 'value', None)
             initial_values[component_names_for_preset[i]] = default_value

        create_default_preset_if_needed(initial_values)

        # 2. Determine which preset to load (last used or default)
        preset_to_load = load_last_used_preset_name()
        available_presets = scan_presets() # Get current list

        if preset_to_load not in available_presets:
            print(f"Last used preset '{preset_to_load}' not found or invalid, loading 'Default'.")
            preset_to_load = "Default"
        else:
            print(f"Loading last used preset: '{preset_to_load}'")

        # 3. Load the preset data
        preset_data = load_preset_data(preset_to_load)
        if preset_data is None and preset_to_load != "Default":
             print(f"Failed to load '{preset_to_load}', attempting to load 'Default'.")
             preset_to_load = "Default"
             preset_data = load_preset_data(preset_to_load)

        # 4. Prepare updates based on loaded data or defaults
        preset_updates = []
        loaded_values = {} # Store loaded values to pass to info update

        if preset_data:
            available_loras = [lora_name for lora_name, _ in scan_lora_files()]
            for i, comp_name in enumerate(component_names_for_preset):
                comp_initial_value = initial_values.get(comp_name) # Get initial value for fallback
                if comp_name in preset_data:
                    value = preset_data[comp_name]
                    if comp_name == "selected_lora" and value not in available_loras:
                        print(f"Startup Warning: Saved LoRA '{value}' not found. Setting to 'None'.")
                        value = "None"
                    preset_updates.append(gr.update(value=value))
                    loaded_values[comp_name] = value # Store loaded value
                else:
                    # If a key is missing in the preset, keep the component's initial value
                    print(f"Startup Warning: Key '{comp_name}' missing in '{preset_to_load}'. Using component's default.")
                    preset_updates.append(gr.update(value=comp_initial_value)) # Use initial value if missing
                    loaded_values[comp_name] = comp_initial_value # Store initial value
        else: # Failed to load Default preset
             print("Critical Error: Failed to load 'Default' preset data. Using hardcoded defaults.")
             preset_updates = [gr.update(value=initial_values.get(name)) for name in component_names_for_preset]
             loaded_values = initial_values # Use initial values

        # 5. Calculate initial iteration info using loaded/default values
        initial_vid_len = loaded_values.get('total_second_length', 5) # Use loaded or default
        initial_fps = loaded_values.get('fps', 30)
        initial_win_size = loaded_values.get('latent_window_size', 9)
        initial_info_text = update_iteration_info(initial_vid_len, initial_fps, initial_win_size)

        # 6. Return updates for the dropdown, all components, and the info display
        # The first output corresponds to preset_dropdown
        return [gr.update(choices=available_presets, value=preset_to_load)] + preset_updates + [initial_info_text]

    block.load(
        fn=apply_preset_and_init_info_on_startup,
        inputs=[],
        # Update dropdown, all preset components, AND the new info display
        outputs=[preset_dropdown] + preset_components_list + [iteration_info_display]
    )
    # --- Startup Loading END ---

    # Separate load for JS (remains the same)
    block.load(None, None, None, js=video_info_js)


def get_available_drives():
    """Detect available drives on the system regardless of OS"""
    available_paths = []
    if platform.system() == "Windows":
        import string
        from ctypes import windll
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1: drives.append(f"{letter}:\\") # Use backslash for Windows paths
            bitmask >>= 1
        available_paths = drives
    elif platform.system() == "Darwin": # macOS
         available_paths = ["/", "/Volumes"] # Check root and mounted volumes
    else: # Linux
        available_paths = ["/", "/mnt", "/media"] # Common mount points

    # Filter out paths that don't actually exist
    existing_paths = [p for p in available_paths if os.path.exists(p)]
    print(f"Allowed Gradio paths: {existing_paths}")
    return existing_paths


# Don't add port args when modifying
block.launch(
    share=args.share,
    inbrowser=True,
    allowed_paths=get_available_drives()
)
