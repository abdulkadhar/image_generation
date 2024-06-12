from diffusers import StableDiffusionPipeline

model_id = "runwayml/stable-diffusion-v1-5"  # Pre-trained model identifier
image_prompt = "A retro-futuristic cityscape with neon lights and flying cars."

pipe = StableDiffusionPipeline.from_pretrained(model_id)
print(pipe)
generated_image = pipe(image_prompt)

generated_image.save("retro_cityscape.png")
print("Image generated and saved!")