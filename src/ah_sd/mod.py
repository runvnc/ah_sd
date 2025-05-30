import torch
import sys
import asyncio
import random
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline # Only SDXL needed
from nanoid import generate
import os
from typing import Optional, Dict, Any, List # Added
import traceback

# Updated imports for MindRoot plugin system
from lib.providers.commands import command
from lib.providers.services import service
from lib.providers.hooks import hook # Assuming this is the correct path for @hook

pipeline = None

# Global model configuration, typically set by environment variables via warmup
if os.environ.get('AH_DEFAULT_SD_MODEL'):
    current_model = os.environ.get('AH_DEFAULT_SD_MODEL') # Assumed to be an SDXL model
    local_model = not current_model.startswith('http') # Basic check if it's a local path vs URL
    from_huggingface = current_model.startswith('http') or '/' in current_model # Heuristic for HF model ID
else:
    current_model = 'stabilityai/stable-diffusion-xl-base-1.0'
    local_model = False
    from_huggingface = True

def random_img_fname():
    img_dir = "imgs"
    if not os.path.exists(img_dir):
        try:
            os.makedirs(img_dir)
        except OSError as e:
            print(f"Error creating directory {img_dir}: {e}", file=sys.stderr)
            img_dir = "."
    return os.path.join(img_dir, generate() + ".png")

def use_model(model_id: str, local: bool = True):
    global current_model, local_model, from_huggingface, pipeline
    current_model = model_id
    local_model = local
    from_huggingface = not local
    pipeline = None # Force reinitialization

@hook()
async def warmup(context: Optional[Any] = None):
    global from_huggingface, current_model, pipeline, local_model
 
    if pipeline is not None:
        print(f"Pipeline for {current_model} already initialized.")
        return

    print(f"Warmup: Initializing SDXL pipeline. Model: {current_model}, Local: {local_model}, HF: {from_huggingface}")

    try:
        print(f"Initializing StableDiffusionXLPipeline for {current_model}...")
        # For local files, determine if it's a single file or a directory
        if local_model and os.path.isfile(current_model):
            print(f"Loading local single file model: {current_model}")
            pipeline = StableDiffusionXLPipeline.from_single_file(current_model, torch_dtype=torch.float16, use_safetensors=current_model.endswith('.safetensors'))
        else: # HuggingFace ID or local directory
            print(f"Loading model from pretrained: {current_model}")
            pipeline = StableDiffusionXLPipeline.from_pretrained(current_model, torch_dtype=torch.float16, use_safetensors=True)
        
        pipeline = pipeline.to("cuda")
        print(f"Pipeline for {current_model} loaded to CUDA.")
        if not local_model and hasattr(pipeline, 'safety_checker') and pipeline.safety_checker is not None:
            print("Disabling safety checker for HuggingFace model.")
            pipeline.safety_checker = lambda images, **kwargs: (images, [False]*len(images))
    except Exception as e:
        trace = traceback.format_exc()
        print(f"Error during pipeline initialization: {e} {trace}", file=sys.stderr)
        # sys.exit(1) # Avoid exiting the whole process in a plugin
        raise RuntimeError(f"Pipeline initialization failed: {e}")

@service()
async def text_to_image(prompt: str, negative_prompt: str = '', 
                        model_id: Optional[str] = None, 
                        from_huggingface_flag: Optional[bool] = None, 
                        count: int = 1, context: Optional[Any] = None, 
                        save_to: Optional[str] = None, 
                        w: int = 1024, h: int = 1024, 
                        steps: int = 20, cfg: float = 8.0,
                        seed: int = 12345) -> Optional[List[str]]:
    global pipeline, current_model, local_model, from_huggingface
    try:
        print('text_to_image service called (SDXL only)')
        if model_id is not None:
            is_local_model_request = not from_huggingface_flag if from_huggingface_flag is not None else not model_id.startswith('http') # Basic inference
            if model_id != current_model or (is_local_model_request != local_model):
                print(f"Model change requested. New model: {model_id}, Current: {current_model}")
                use_model(model_id, local=is_local_model_request)
        
        if pipeline is None:
            print("Pipeline not initialized. Calling warmup...")
            await warmup(context=context)
            if pipeline is None:
                print("Pipeline initialization failed. Cannot generate image.", file=sys.stderr)
                return None

        images_fnames = []
        
        # Handle seed: -1 means random, otherwise use specified seed
        actual_seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
        generator = torch.Generator(device="cuda").manual_seed(actual_seed)
        print(f"Using seed: {actual_seed}")
        
        (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(prompt, device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=negative_prompt)
        print("SDXL prompt encoded.")

        print("count = ", count)
        for n in range(count):
            actual_w = w
            actual_h = h

            print(f"Generating SDXL image {n+1}/{count} with prompt: '{prompt[:50]}...' model: {current_model} W: {actual_w} H: {actual_h}")
            
            image_obj = pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                width=actual_w, height=actual_h,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                num_inference_steps=steps, guidance_scale=cfg,
                                generator=generator).images[0]
            
            fname_to_save = save_to if save_to and count == 1 else random_img_fname()
            
            try:
                image_obj.save(fname_to_save)
                print(f"Image saved to {fname_to_save}")
                images_fnames.append(fname_to_save)
            except Exception as e:
                print(f"Error saving image to {fname_to_save}: {e}", file=sys.stderr)

        return images_fnames if images_fnames else None
    except Exception as e:
        trace = traceback.format_exc()
        print(f"Error in text_to_image: {e}\n{trace}", file=sys.stderr)
        # sys.exit(1)
        raise RuntimeError(f"Error generating image: {e}")

@command()
async def image(prompt: str, negative_prompt: str = "", 
                steps: int = 20, cfg: float = 8.0, 
                w: int = 1024, h: int = 1024,
                seed: int = 12345,
                context: Optional[Any] = None) -> Optional[Dict[str, str]]: # Updated signature
    """image: Generate an image from a prompt using the currently configured model.
    This plugin is configured for SDXL models only. Default size is 1024x1024.

    Args:
        prompt (str): The text prompt to generate the image from.
        negative_prompt (str, optional): Text prompt to guide what not to include. Defaults to ''.
        steps (int, optional): Number of inference steps. Defaults to 20.
        cfg (float, optional): Classifier-Free Guidance scale. Defaults to 8.0.
        w (int, optional): Width of the generated image. Defaults to 1024.
        h (int, optional): Height of the generated image. Defaults to 1024.
        seed (int, optional): Random seed for generation. Use -1 for random seed. Defaults to 12345.
        context (Optional[Any]): The execution context.

    Example:
    { "image": { "prompt": "A cute tabby cat in the forest" } }

    Example with more parameters:
    { "image": { 
        "prompt": "A futuristic cityscape at sunset", 
        "negative_prompt": "ugly, blurry, cars", 
        "w": 768, 
        "h": 768,
        "seed": 42,
        "steps": 25,
        "cfg": 7.0
      }
    }
    """
    print(f'image command (SDXL only) called with prompt: "{prompt[:50]}..." negative_prompt: "{negative_prompt[:50]}..."')
    fnames = await text_to_image(prompt=prompt, negative_prompt=negative_prompt,
                                model_id=None, from_huggingface_flag=None, 
                                w=w, h=h, steps=steps, cfg=cfg, seed=seed, context=context)
    
    if fnames and isinstance(fnames, list) and len(fnames) > 0:
        fname = fnames[0] # For now, command handles the first image if multiple were made by service
        print(f"Image command output to file: {fname}")
        if hasattr(context, 'insert_image'):
            img_filename = os.path.basename(fname)
            rel_url = f"/imgs/{img_filename}"
            await context.insert_image(rel_url)
            return { "status": "generated", "file_path": rel_url, "message": f"Image generated at {rel_url} and inserted."}
        else:
            print("Context does not have insert_image method.", file=sys.stderr)
            return { "status": "generated_no_insert", "file_path": fname, "message": "Image generated but could not be inserted into chat."}
    else:
        print("Image generation failed or returned no filename.", file=sys.stderr)
        if hasattr(context, 'send_message'):
            await context.send_message("Sorry, I couldn't generate the image.")
        return { "status": "error", "message": "Image generation failed." }
