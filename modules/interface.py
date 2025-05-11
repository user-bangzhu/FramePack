import gradio as gr
import time
import datetime
from typing import List, Dict, Any, Optional

from modules.video_queue import JobStatus
from modules.prompt_handler import get_section_boundaries, get_quick_prompts
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html


def create_interface(
    process_fn, 
    monitor_fn, 
    end_process_fn, 
    update_queue_status_fn,
    default_prompt: str = 'The girl dances gracefully, with clear movements, full of charm.'
):
    """
    Create the Gradio interface for the video generation application
    
    Args:
        process_fn: Function to process a new job
        monitor_fn: Function to monitor an existing job
        end_process_fn: Function to cancel the current job
        update_queue_status_fn: Function to update the queue status display
        default_prompt: Default prompt text
        
    Returns:
        Gradio Blocks interface
    """
    # Get section boundaries and quick prompts
    section_boundaries = get_section_boundaries()
    quick_prompts = get_quick_prompts()
    
    # Create the interface
    css = make_progress_bar_css()
    block = gr.Blocks(css=css).queue()
    
    with block:
        gr.Markdown('# FramePack with Timestamped Prompts')
        
        with gr.Tabs():
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(sources='upload', type="numpy", label="Image", height=320)   
                        
                        prompt = gr.Textbox(label="Prompt", value=default_prompt)
                        example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
                        example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

                        with gr.Row():
                            start_button = gr.Button(value="Add to Queue")
                            monitor_button = gr.Button(value="Monitor Selected Job")
                            end_button = gr.Button(value="Cancel Current Job", interactive=True)

                        with gr.Group():
                            use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')

                            n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=False)  # Not used
                            seed = gr.Number(label="Seed", value=31337, precision=0)

                            total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                            latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=False)  # Should not change
                            steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')

                            cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=False)  # Should not change
                            gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                            rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change

                            gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=128, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")

                    with gr.Column():
                        current_job_id = gr.Textbox(label="Current Job ID", visible=True, interactive=True)
                        preview_image = gr.Image(label="Next Latents", height=200, visible=False)
                        result_video = gr.Video(label="Finished Frames", autoplay=True, show_share_button=False, height=512, loop=True)
                        gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
                        progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
                        progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            
            with gr.TabItem("Queue Status"):
                queue_status = gr.DataFrame(
                    headers=["Job ID", "Status", "Created", "Started", "Completed", "Elapsed", "Queue Position"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    label="Job Queue"
                )
                refresh_button = gr.Button("Refresh Queue Status")
                refresh_button.click(update_queue_status_fn, outputs=[queue_status])
        
        # Connect the main process
        ips = [input_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache]
        start_button.click(fn=process_fn, inputs=ips, outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button])
        
        # Add monitor functionality
        monitor_button.click(fn=monitor_fn, inputs=[current_job_id], outputs=[result_video, current_job_id, preview_image, progress_desc, progress_bar, start_button, end_button])
        
        # Add cancel functionality
        end_button.click(fn=end_process_fn)
    
    return block


def format_queue_status(jobs):
    """Format job data for display in the queue status table"""
    rows = []
    for job in jobs:
        created = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.created_at)) if job.created_at else ""
        started = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.started_at)) if job.started_at else ""
        completed = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(job.completed_at)) if job.completed_at else ""

        # Calculate elapsed time
        elapsed_time = ""
        if job.started_at and job.completed_at:
            start_datetime = datetime.datetime.fromtimestamp(job.started_at)
            complete_datetime = datetime.datetime.fromtimestamp(job.completed_at)
            elapsed_seconds = (complete_datetime - start_datetime).total_seconds()
            elapsed_time = str(elapsed_seconds) # Convert to string

        position = job.queue_position if hasattr(job, 'queue_position') else ""

        rows.append([
            job.id,
            job.status.value,
            created,
            started,
            completed,
            elapsed_time,
            str(position) if position is not None else ""
        ])
    return rows
