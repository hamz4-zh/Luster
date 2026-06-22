import gradio as gr
import torch
from diffusers import StableDiffusionXLPipeline

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16, 
    variant="fp16",
    use_safetensors=True
).to("cuda")

# "checkpoint-1000"
pipe.load_lora_weights("jewelry_lora_model/checkpoint-1000") 
print("Model loaded")


# Left side is what the user sees  right side is what the model need
shape_map = {
    "Round": "round cut", 
    "Princess (square)": "princess cut", 
    "Cushion": "cushion cut"
}

stone_map = {
    "Diamond": "diamond", 
    "Emerald": "emerald", 
    "Sapphire": "sapphire"
}

color_map = {
    "Yellow Gold": "yellow gold", 
    "White Gold": "white gold", 
    "Rose Gold": "rose gold"
}

style_map = {
    "Plain Band (Solitaire)": "solitaire", 
    "With Side Stones": " " # blank tag (Pave')
}


def generate_jewelry(ui_shape, ui_stone, ui_color, ui_style):
    backend_shape = shape_map[ui_shape]
    backend_stone = stone_map[ui_stone]
    backend_color = color_map[ui_color]
    backend_style = style_map[ui_style]

    raw_tags = f"{backend_shape}_{backend_stone}_{backend_color}_{backend_style}"
    # for the folder names 
    clean_tags = raw_tags.replace(" _", "_").replace("__", "_").strip("_")
    
    positive_prompt = f"A beautiful jewelry ring featuring {clean_tags}, professional jewelry macro photography, pure white studio background, 8k resolution, photorealistic"
    negative_prompt = "cluttered background, bad anatomy, blurry, low quality, sketch, 3d render, plastic, hand"
    
    # plain band, block side stones
    if backend_style == "solitaire":
        negative_prompt += ", side stones, diamonds on band, pave, halo"

    print(f"User selected: {ui_shape}, {ui_stone}, {ui_color}, {ui_style}")
    print(f"Actual prompt sent to AI: {positive_prompt}")
    
    # Generate the image
    image = pipe(
        prompt=positive_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30, 
        guidance_scale=7.5
    ).images[0]
    
    return image


with gr.Blocks(theme=gr.themes.Monochrome()) as interface:
    gr.Markdown("<h1 style='text-align: center;'> AI Jewelry Studio</h1>")
    gr.Markdown("<p style='text-align: center;'>Select your specifications to generate a bespoke piece.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):

            shape_dropdown = gr.Dropdown(choices=list(shape_map.keys()), value="Round", label="1. Select Cut")
            stone_dropdown = gr.Dropdown(choices=list(stone_map.keys()), value="Diamond", label="2. Select Stone")
            color_dropdown = gr.Dropdown(choices=list(color_map.keys()), value="Yellow Gold", label="3. Select Metal")
            style_dropdown = gr.Dropdown(choices=list(style_map.keys()), value="Plain Band (Solitaire)", label="4. Select Setting Style")
            
            generate_btn = gr.Button(" Generate Concept", variant="primary")
            
        with gr.Column(scale=2):
            output_image = gr.Image(label="Generated Concept")

    # Connect the UI 
    generate_btn.click(
        fn=generate_jewelry,
        inputs=[shape_dropdown, stone_dropdown, color_dropdown, style_dropdown],
        outputs=output_image
    )

interface.launch() 