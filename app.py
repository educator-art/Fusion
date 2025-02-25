"""
Fusion
2025.02.01 CREATED
2025.02.14 UPDATED
"""
import gradio as gr
import re
import torch
from diffusers import StableDiffusionXLPipeline
from diffusers import EulerAncestralDiscreteScheduler
import gc
import random
import numpy as np
from datetime import datetime
import os

# 初期のプロンプトとネガティブプロンプトを設定
INITIAL_PROMPT="masterpiece, high score, great score, absurdres"
INITIAL_NEGATIVE_PROMPT="lowres, bad anatomy, bad hands, text, error, missing finger, extra digits, fewer digits, cropped, worst quality, low quality, low score, bad score, average score, signature, watermark, username, blurry"
MODEL_INFO=[["cagliostrolab/animagine-xl-4.0","4adce86"],["cagliostrolab/animagine-xl-4.0","2b7c1b3"],["cagliostrolab/animagine-xl-4.0-zero","22e4c70"]]
# INTの最大値
INT_MAX_VALUE=2147483647
# 画像を保存するディレクトリ
SAVE_IMAGE_DIR="./save_image_dir"

def generate(width, height, prompt, negative_prompt, model, cfg, steps, seed):
    
    # User Input Information
    prompt_order=f"""\
    Generation Image
    width: {width}
    height: {height}
    prompt: {prompt}
    negative prompt: {negative_prompt}
    model: {model}
    cfg: {cfg}
    steps: {steps}
    seed: {seed}\
    """

    message=re.sub(r"^ +(\S)", r"\1", prompt_order, flags=re.MULTILINE)
    print(message)

    # Selected Model
    if model=="Anim4gine":
        load_model = MODEL_INFO[0]
    elif model=="Anim4gine Opt":
        load_model = MODEL_INFO[1]
    else:
        load_model = MODEL_INFO[2]

    # Generate Image
    torch.cuda.empty_cache()  # GPUメモリの解放をする
    gc.collect() # Pythonの明示的なメモリ解放をする

    pipe = StableDiffusionXLPipeline.from_pretrained(
    load_model[0],
    torch_dtype=torch.float16,
    use_safetensors=True,
    custom_pipeline="lpw_stable_diffusion_xl",
    add_watermarker=False,
    revision=load_model[1]
    )
    pipe.to('cuda')

    # Set Euler a scheduler
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    #  Define seed value
    seed = seed
    if seed == -1:
        print("randomize seed")
        seed = random.randint(0,INT_MAX_VALUE)
    else:
        print(f"seed is {seed}")
    
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    generator= torch.Generator()
    generator.manual_seed(seed)

    # Image is PIL Object
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=int(width),
        height=int(height),
        guidance_scale=int(cfg),
        num_inference_steps=int(steps),
        generator=generator
    ).images[0]

    # 画像を保存するディレクトリを作成する
    if not os.path.isdir(SAVE_IMAGE_DIR):
        print("create save_image_dir")
        os.mkdir(SAVE_IMAGE_DIR)

    # ファイルを保存する
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_filename = f"image_{current_time}.png"
    image_filepath=os.path.join(SAVE_IMAGE_DIR, image_filename)
    image.save(image_filepath)

    image_info_filename = f"image_{current_time}.txt"
    image_info_filepath=os.path.join(SAVE_IMAGE_DIR, image_info_filename)
    
    # User Input Information
    prompt_history=f"""\
    Generation Image
    width: {width}
    height: {height}
    prompt: {prompt}
    negative prompt: {negative_prompt}
    model: {model}
    cfg: {cfg}
    steps: {steps}
    seed: {seed}\
    """

    prompt_history_text=re.sub(r"^ +(\S)", r"\1", prompt_history, flags=re.MULTILINE)
    with open(image_info_filepath, "w") as file:
        file.write(prompt_history_text)

    # Return image(PIL Object) to gradio ui image 
    return image
    

demo = gr.Interface(
    fn=generate,
    inputs=[gr.Slider(minimum=512, maximum=1536, value=1024, step=8, label="Width"),gr.Slider(minimum=512, maximum=1536, value=1024, step=8, label="Height"),gr.Textbox(value=INITIAL_PROMPT,label="Prompt"),gr.Textbox(value=INITIAL_NEGATIVE_PROMPT, label="Negative Prompt"),gr.Dropdown(choices=["Anim4gine","Anim4gine Opt","Anim4gine Zero"],value="Anim4gine Opt", label="Model", info="selected model will be loaded"), gr.Slider(minimum=1, maximum=15, value=5, step=1, label="CFG Scale"),gr.Slider(minimum=1, maximum=40, value=28, step=1, label="Sampling Steps"),gr.Slider(minimum=-1,maximum=INT_MAX_VALUE, value=-1, step=1, label="Seed (option)", info="-1 is  randomize seed"),],
    outputs=[gr.Image(format="png", height=768, show_download_button=True)],
    title="Fusion - a personal image generation",
    description="<center>feel free to prompt. 2025.02.14 update</center>",
    flagging_mode="never",
    api_name=False,# 一時的にFalse（APIの利用を制限する）
    submit_btn=gr.Button("Generate",variant="primary"),
    clear_btn=None,
)

demo.launch(share=True)