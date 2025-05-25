import torch
import sys
import asyncio
from diffusers import DiffusionPipeline
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
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
    current_model = 'models/' + os.environ.get('AH_DEFAULT_SD_MODEL')
    local_model = True
    use_sdxl = True # Defaulting to True if model is local, can be overridden by AH_USE_SDXL
    from_huggingface = False
else:
    current_model = 'stabilityai/stable-diffusion-xl-base-1.0'
    local_model = False
    use_sdxl = True # Defaulting to SDXL for HuggingFace
    from_huggingface = True

if os.environ.get('AH_USE_SDXL') == 'True':
    use_sdxl = True
elif os.environ.get('AH_USE_SDXL') == 'False':
    use_sdxl = False
    if not os.environ.get('AH_DEFAULT_SD_MODEL'): # If not local and SDXL is false, use a non-XL model
        current_model = 'runwayml/stable-diffusion-v1-5'

def random_img_fname():
    img_dir = "imgs"
    if not os.path.exists(img_dir):
        try:
            os.makedirs(img_dir)
        except OSError as e:
            print(f"Error creating directory {img_dir}: {e}", file=sys.stderr)
            img_dir = "."
    return os.path.join(img_dir, generate() + ".png")

def use_model(model_id: str, local: bool = True, is_sdxl: Optional[bool] = None):
    global current_model, local_model, use_sdxl, from_huggingface, pipeline
    current_model = model_id
    local_model = local
    from_huggingface = not local
    if is_sdxl is not None:
        use_sdxl = is_sdxl
    pipeline = None # Force reinitialization

def get_pipeline_embeds(pipe, prompt: str, negative_prompt: str, device: str):
    max_length = pipe.tokenizer.model_max_length

    input_ids = pipe.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
    negative_ids = pipe.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)

    shape_max_length = max(input_ids.shape[-1], negative_ids.shape[-1])

    if input_ids.shape[-1] < shape_max_length:
        input_ids = pipe.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length", max_length=shape_max_length).input_ids.to(device)
    if negative_ids.shape[-1] < shape_max_length:
        negative_ids = pipe.tokenizer(negative_prompt, return_tensors="pt", truncation=False, padding="max_length", max_length=shape_max_length).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipe.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipe.text_encoder(negative_ids[:, i: i + max_length])[0])
    
    prompt_embeds = torch.cat(concat_embeds, dim=1)
    negative_prompt_embeds = torch.cat(neg_embeds, dim=1)
    return prompt_embeds, negative_prompt_embeds

@hook()
async def warmup(context: Optional[Any] = None):
    global from_huggingface, current_model, pipeline, local_model, use_sdxl
 
    if pipeline is not None:
        print(f"Pipeline for {current_model} already initialized.")
        return

    print(f"Warmup: Initializing pipeline. Model: {current_model}, SDXL: {use_sdxl}, Local: {local_model}, HF: {from_huggingface}")

    try:
        if True or use_sdxl:
            print(f"Initializing StableDiffusionXLPipeline for {current_model}...")
            pipeline = DiffusionPipeline.from_pretrained(
                current_model, torch_dtype=torch.float16, safety_checker=None
            ).to("cuda")

            #if not from_huggingface:
            #   pipeline = StableDiffusionXLPipeline.from_single_file(current_model, torch_dtype=torch.float16, use_safetensors=True if current_model.endswith('.safetensors') else False)
            #else:
            #    pipeline = StableDiffusionXLPipeline.from_pretrained(current_model, torch_dtype=torch.float16, use_safetensors=True)
        else:
            print(f"Initializing StableDiffusionPipeline for {current_model}...")
            if not from_huggingface:
                pipeline = StableDiffusionPipeline.from_single_file(current_model, torch_dtype=torch.float16, use_safetensors=True if current_model.endswith('.safetensors') else False)
            else:
                pipeline = StableDiffusionPipeline.from_pretrained(current_model, torch_dtype=torch.float16, use_safetensors=True)
        
        print(f"Pipeline for {current_model} loaded to CUDA.")

        if not local_model and hasattr(pipeline, 'safety_checker') and pipeline.safety_checker is not None:
            print("Disabling safety checker for HuggingFace model.")
            pipeline.safety_checker = lambda images, **kwargs: (images, [False]*len(images))
    except Exception as e:
        trace = traceback.format_exc()
        print(f"Error during pipeline initialization: {e} {trace}", file=sys.stderr)
        sys.exit(1)
        pipeline = None

@service()
async def text_to_image(prompt: str, negative_prompt: str = '', 
                        model_id: Optional[str] = None, 
                        from_huggingface_flag: Optional[bool] = None, 
                        is_sdxl_flag: Optional[bool] = None,
                        count: int = 1, context: Optional[Any] = None, 
                        save_to: Optional[str] = None, 
                        w: int = 1024, h: int = 1024, 
                        steps: int = 20, cfg: float = 8.0) -> Optional[str]: # Updated signature
    global pipeline, current_model, use_sdxl, local_model, from_huggingface # Ensure all relevant globals are accessible
    try:
        print('text to image')
        if model_id is not None:
            is_local_model_request = not from_huggingface_flag if from_huggingface_flag is not None else not model_id.startswith('http') # Basic inference
            if model_id != current_model or \
            (is_sdxl_flag is not None and is_sdxl_flag != use_sdxl) or \
            (is_local_model_request != local_model):
                print(f"Model change requested by service. New model: {model_id}, Current: {current_model}, New SDXL: {is_sdxl_flag}, Current SDXL: {use_sdxl}")
                use_model(model_id, local=is_local_model_request, is_sdxl=is_sdxl_flag)
        
        print("A.")
        if pipeline is None:
            print("Pipeline not initialized. Calling warmup...")
            await warmup(context=context)
            if pipeline is None:
                print("Pipeline initialization failed. Cannot generate image.", file=sys.stderr)
                return None
        else:
            print("pipeline already initialized?")
        images_fnames = []
        (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
        ) = pipeline.encode_prompt(prompt, "cuda", num_images_per_prompt=1, negative_prompt=negative_prompt)
        print("encoded prompt?")
        print(pipeline)
        print("count = ", count)
        for n in range(count):
            actual_w = w if w != 1024 else (1024 if use_sdxl else 512)
            actual_h = h if h != 1024 else (1024 if use_sdxl else 512)
            if w != 1024 : actual_w = w # If user specified w, use it
            if h != 1024 : actual_h = h # If user specified h, use it

            print(f"Generating image {n+1}/{count} with prompt: '{prompt[:50]}...' model: {current_model} SDXL: {use_sdxl} W: {actual_w} H: {actual_h}")
            
            image_obj = pipeline(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds,
                                width=actual_w, height=actual_h,
                                pooled_prompt_embeds=pooled_prompt_embeds,
                                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
                                num_inference_steps=steps, guidance_scale=cfg).images[0]
            
            fname_to_save = save_to if save_to and count == 1 else random_img_fname()
            
            try:
                image_obj.save(fname_to_save)
                print(f"Image saved to {fname_to_save}")
                images_fnames.append(fname_to_save)
            except Exception as e:
                print(f"Error saving image to {fname_to_save}: {e}", file=sys.stderr)

        return images_fnames[0] if count == 1 and images_fnames else None # Simplified return for single image
    except Exception as e:
        trace = traceback.format_exc()
        print(e)
        print(trace)
        sys.exit(1)
        raise Error(f"Error generating image: {e}\nTraceback:\n{trace}")

@command()
async def image(prompt: str, negative_prompt: str = "", 
                steps: int = 20, cfg: float = 8.0, 
                w: int = 1024, h: int = 1024, 
                context: Optional[Any] = None) -> Optional[Dict[str, str]]: # Updated signature
    """image: Generate an image from a prompt using the currently configured model.

    Args:
        prompt (str): The text prompt to generate the image from.
        negative_prompt (str, optional): Text prompt to guide what not to include. Defaults to ''.
        steps (int, optional): Number of inference steps. Defaults to 20.
        cfg (float, optional): Classifier-Free Guidance scale. Defaults to 8.0.
        w (int, optional): Width of the generated image. Defaults to 1024 (or 512 for non-SDXL default model).
        h (int, optional): Height of the generated image. Defaults to 1024 (or 512 for non-SDXL default model).
        context (Optional[Any]): The execution context.

    Example:
    { "image": { "prompt": "A cute tabby cat in the forest" } }

    Example with more parameters:
    { "image": { 
        "prompt": "A futuristic cityscape at sunset", 
        "negative_prompt": "ugly, blurry, cars", 
        "w": 768, 
        "h": 768,
        "steps": 25,
        "cfg": 7.0
      }
    }
    """
    print('image command called with prompt:', prompt, 'negative_prompt:', negative_prompt )
    fname = await text_to_image(prompt=prompt, negative_prompt=negative_prompt,
                                model_id=None, from_huggingface_flag=None, is_sdxl_flag=None, 
                                w=w, h=h, steps=steps, cfg=cfg, context=context)
    print("Image generation completed, filename:", fname)
    #sys.exit(1)
    if fname and isinstance(fname, str):
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

