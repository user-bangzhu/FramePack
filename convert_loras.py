import os
import sys
import glob
import time
import platform
import argparse
import traceback
import subprocess
import gradio as gr
from typing import List, Tuple

# Add proper paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))

# Import the conversion functionality
from convert_hunyuan_video_to_diffusers import convert

# Parse command-line arguments
parser = argparse.ArgumentParser(description="LoRA Conversion Tool")
parser.add_argument('--share', action='store_true', help='Enable sharing the Gradio app')
parser.add_argument('--inbrowser', action='store_true', default=True, help='Open the Gradio app in browser')
parser.add_argument('--server', type=str, default='0.0.0.0', help='Server address')
parser.add_argument('--port', type=int, default=7860, help='Server port')
args = parser.parse_args()

# Setup default paths
loras_folder = os.path.join(current_dir, 'loras')
os.makedirs(loras_folder, exist_ok=True)
print(f"Default loras folder: {loras_folder}")

# Function to get all available drives (copied from app.py)
def get_available_drives():
    """Detect available drives on the system regardless of OS"""
    available_paths = []
    
    if platform.system() == "Windows":
        import string
        from ctypes import windll
        
        # Check each drive letter
        drives = []
        bitmask = windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drives.append(f"{letter}:/")
            bitmask >>= 1
            
        available_paths = drives
    else:
        # For Linux/Mac, just use root
        available_paths = ["/"]
        
    print(f"Available drives detected: {available_paths}")
    return available_paths

# Function to open folder in file explorer/finder
def open_folder(folder_path):
    """Opens the specified folder in the file explorer/manager in a cross-platform way."""
    try:
        folder_path = os.path.abspath(folder_path)
        if platform.system() == "Windows":
            os.startfile(folder_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", folder_path])
        else:  # Linux
            subprocess.run(["xdg-open", folder_path])
        return f"Opened {os.path.basename(folder_path)} folder"
    except Exception as e:
        return f"Error opening folder: {str(e)}"

# Get supported LoRA file extensions
def get_supported_extensions():
    """Return supported LoRA file extensions"""
    return ['.pt', '.ckpt', '.safetensors']

# Get all LoRA files from a folder
def get_lora_files(folder_path: str) -> List[str]:
    """Get all LoRA files from a folder."""
    if not folder_path or not os.path.exists(folder_path):
        return []
    
    extensions = get_supported_extensions()
    lora_files = []
    
    for ext in extensions:
        files = glob.glob(os.path.join(folder_path, f"**/*{ext}"), recursive=True)
        lora_files.extend(files)
    
    return sorted(lora_files)

# Convert a single LoRA file
def convert_lora_file(
    input_file: str, 
    output_path: str, 
    target_format: str = "default"
) -> Tuple[bool, str]:
    """
    Convert a single LoRA file between formats
    
    Args:
        input_file: Path to the input LoRA file
        output_path: Path where to save the converted model
        target_format: Target format ("default" or "other")
    
    Returns:
        Tuple of (success, message)
    """
    try:
        print(f"Converting {input_file} to {target_format} format")
        
        # Create output filename
        file_basename = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join(output_path, f"{file_basename}_{target_format}.safetensors")
        
        # Convert the file
        convert(input_file, output_file, target_format)
        
        return True, f"Successfully converted {os.path.basename(input_file)} to {output_file}"
    
    except Exception as e:
        error_msg = f"Error converting {os.path.basename(input_file)}: {str(e)}"
        traceback.print_exc()
        return False, error_msg

# Batch conversion function
def batch_convert(
    input_folder: str, 
    output_folder: str,
    target_format: str,
    progress=gr.Progress()
) -> str:
    """
    Convert all LoRA files in the input folder
    
    Args:
        input_folder: Folder containing LoRA files
        output_folder: Folder where to save converted models
        target_format: Target format ("default" or "other")
        progress: Gradio progress object
    
    Returns:
        Message with conversion results
    """
    if not input_folder or not os.path.exists(input_folder):
        return f"Input folder does not exist: {input_folder}"
    
    # Set default output folder if not provided
    if not output_folder:
        output_folder = loras_folder
    else:
        try:
            os.makedirs(output_folder, exist_ok=True)
        except Exception as e:
            return f"Error creating output folder: {str(e)}"
    
    # Get all LoRA files from the input folder
    lora_files = get_lora_files(input_folder)
    
    if not lora_files:
        return f"No LoRA files found in {input_folder}"
    
    results = []
    success_count = 0
    
    # Process each LoRA file
    for idx, lora_file in enumerate(progress.tqdm(lora_files)):
        progress_text = f"Processing {idx+1}/{len(lora_files)}: {os.path.basename(lora_file)}"
        progress(idx/len(lora_files), desc=progress_text)
        
        success, message = convert_lora_file(
            lora_file, 
            output_folder, 
            target_format
        )
        
        if success:
            success_count += 1
        
        results.append(message)
    
    # Create final result message
    final_message = f"Conversion complete. Processed {len(lora_files)} files with {success_count} successes and {len(lora_files) - success_count} failures.\n\n"
    final_message += "\n".join(results)
    
    return final_message

# Create the Gradio interface
with gr.Blocks(title="LoRA Converter") as app:
    gr.Markdown("# HunyuanVideo LoRA Converter")
    gr.Markdown("Convert LoRA files between formats")
    
    with gr.Row():
        with gr.Column():
            input_folder = gr.Textbox(
                label="Input Folder Path",
                placeholder="Path to folder containing LoRA files to convert",
                info="Folder containing LoRA files to convert (.pt, .ckpt, .safetensors)"
            )
            
            output_folder = gr.Textbox(
                label="Output Folder Path",
                placeholder="Leave empty to use the default loras folder",
                value=loras_folder,
                info="Folder where to save converted models"
            )
            
            target_format = gr.Dropdown(
                label="Target Format",
                choices=["default", "other"],
                value="default",
                info="Convert to 'default' (from Diffusers) or 'other' (to Diffusers) format"
            )
            
            with gr.Row():
                convert_btn = gr.Button("Start Conversion", variant="primary")
                open_input_btn = gr.Button("Open Input Folder")
                open_output_btn = gr.Button("Open Output Folder")
        
        with gr.Column():
            result_text = gr.Textbox(
                label="Conversion Results",
                placeholder="Conversion results will appear here",
                lines=20
            )

    # Connect the buttons
    convert_btn.click(
        fn=batch_convert,
        inputs=[input_folder, output_folder, target_format],
        outputs=result_text
    )
    
    open_input_btn.click(
        fn=lambda x: open_folder(x) if x else "No input folder specified",
        inputs=[input_folder],
        outputs=result_text
    )
    
    open_output_btn.click(
        fn=lambda x: open_folder(x if x else loras_folder),
        inputs=[output_folder],
        outputs=result_text
    )

# Launch the app
if __name__ == "__main__":
    # Print startup information
    print("\n=== LoRA CONVERSION TOOL ===")
    print(f"Supported extensions: {', '.join(get_supported_extensions())}")
    print(f"Default output folder: {loras_folder}")
    print("===========================\n")
    
    # Launch with dynamically detected drives
    app.launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_name=args.server,
        server_port=args.port,
        allowed_paths=get_available_drives()
    ) 