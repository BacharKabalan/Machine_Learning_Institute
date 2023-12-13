from diffusers import StableDiffusionPipeline
from PIL import Image
import torch

def plotting_grid(model_id):
    torch.cuda.empty_cache()
    seed = 42
    torch.manual_seed(seed)
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe_base = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-2")
    pipe.to("cuda")
    pipe_base.to("cuda")
    prompt = "A photo of brk1 smiling in a space suit"
    # Run inference using ChatGPT prompts to acquire 4 image panels
    num_inference_steps=50
    image1 = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5).images[0]
    image1.save("tuned_lego_panel_1.png")
    image2 = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5).images[0]
    image2.save("tuned_lego_panel_1.png")
    image3 = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5).images[0]
    image3.save("tuned_lego_panel_1.png")
    image4 = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=7.5).images[0]
    image4.save("tuned_lego_panel_1.png")
    
    all_images = [image1,image2,image3,image4]
    grid = image_grid(all_images, rows=1, cols=4)
    return grid





def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid