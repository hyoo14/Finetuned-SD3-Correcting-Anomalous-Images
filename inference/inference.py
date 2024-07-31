from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-3-medium-diffusers",
    torch_dtype=torch.float16
)
lora_output_path = "../training/trained-sd3-lora-miniature"
pipeline.load_lora_weights("trained-sd3-lora-miniature")

pipeline.enable_sequential_cpu_offload()

image = pipeline("lying on the grass/street").images[0]



image.save("output.png")
