# import torch
# from diffusers import StableDiffusionDepth2ImgPipeline
# from diffusers.utils import load_image, make_image_grid

# pipeline = StableDiffusionDepth2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-2-depth",
#     torch_dtype=torch.float16,
#     use_safetensors=True,
# ).to("cuda")


# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# init_image = load_image(url)
# prompt = "Remove the background"
# negative_prompt = "bad, deformed, ugly, bad anatomy"
# image = pipeline(prompt=prompt, image=init_image, negative_prompt=negative_prompt, strength=0.7).images[0]
# # make_image_grid([init_image, image], rows=1, cols=2)

# image.save('test.png')

# import torch
# from diffusers import AutoPipelineForImage2Image
# from diffusers.utils import load_image, make_image_grid

# pipeline = AutoPipelineForImage2Image.from_pretrained(
#     "kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
# )
# pipeline.enable_model_cpu_offload()
# # remove following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
# pipeline.enable_xformers_memory_efficient_attention()

# # init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
# url = load_image("http://images.cocodataset.org/val2017/000000039769.jpg")

# prompt = "remove background, movie greenscreen"
# image = pipeline(prompt, image=url).images[0]
# image.save('test.png')


from nanoowl.owl_predictor import OwlPredictor

predictor = OwlPredictor(
    "google/owlvit-base-patch32",
    image_encoder_engine="data/owlvit-base-patch32-image-encoder.engine"
)

image = PIL.Image.open("assets/owl_glove_small.jpg")

output = predictor.predict(image=image, text=["an owl", "a glove"], threshold=0.1)

print(output)