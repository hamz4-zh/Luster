import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image


base_model_path = "stabilityai/stable-diffusion-xl-base-1.0"

lora_checkpoints = [
    "jewelry_lora_model/checkpoint-500",
    "jewelry_lora_model/checkpoint-1000",
    "jewelry_lora_model/checkpoint-1500"
]

positive_prompt = "cushion_cut_emerald_rose_gold_solitaire, professional jewelry macro photography, pure white studio background"

negative_prompt = "side stones, diamonds on band, pave, textured band, halo, engraving, cluttered background"

test_seed = 42 


print("Loading base SDXL model (this takes a moment)...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path, 
    torch_dtype=torch.float16, 
    variant="fp16",
    use_safetensors=True
).to("cuda")

generated_images = []


for checkpoint in lora_checkpoints:
    print(f"\nTesting {checkpoint}")
    pipe.unload_lora_weights()
    
    #  specific checkpoint
    try:
        pipe.load_lora_weights(checkpoint)
    except Exception as e:
        print(f"Could not load {checkpoint}. Skipping (Error: {e})")
        continue


    generator = torch.Generator(device="cuda").manual_seed(test_seed)
    
    image = pipe(
        prompt=positive_prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=30, 
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    generated_images.append(image)
