import torch
from diffusers import StableDiffusionPipeline

print("---Step 1: Loading AI Model---")
print("This might take a while the first time as it downloads the model...")

model_id = "runwayml/stable-diffusion-v1-5"


pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)


pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

print("-- Model Loaded Successfully!---")


prompt = "A majestic lion wearing a crown on a black background, minimalist line art"


print(f"--- Step 3: Generating image for the prompt: '{prompt}'---")


image = pipe(prompt).image[0]

print("--- image generated successfully!---.capatilize")

output_filename = "tshirt_design.png"
image.save(output_filename)

print(f"---Success! Image saved as a {output_filename} in your project folder. ---")
