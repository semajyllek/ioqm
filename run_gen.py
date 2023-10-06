
"""
loads a dataset of prompts and generates images from them using a given model
"""


from typing import Any, Optional, Tuple

from diffusers import DiffusionPipeline, StableDiffusionPipeline
from datasets import load_dataset, Dataset

from functools import partial

from pathlib import Path
import torch
import os


def run_gen(
		model_id: str, 
		prompt_range: Optional[Tuple[int, int]] = None, 
		icl_suffix: str = " on a table.", 
		save_folder: str = "",
		prompt_ds: Optional[Dataset] = None
):
	
	prompt_ds = get_selected_prompt_dataset(prompt_ds, prompt_range) # load dataset of prompts
	img_gen_pipe = get_img_gen_pipe(model_id)                        # load image generation pipeline
	model_name = model_id.split('/')[-1]           	                 # get model name from model_id
	img_folder = get_img_folder(save_folder, model_name)
		
	for prompt in prompt_ds:
		icl_prompt = prompt['text'] + icl_suffix                     # add " on a table." to prompt
		image_output = img_gen_pipe(icl_prompt)
		save_path = img_folder / f"{'_'.join(prompt['text'].split())}.png"
		save_image(image_output, save_path) # ex. 1_microwave_and_3_fire_hydrants_and_2_handbags.png


def save_image(image_output: Any, save_path: Path) -> None:
	"""
	desc: save image to file, handle different pipline outputs
	"""
	if isinstance(image_output, torch.Tensor):
		image = image_output[0]
	if isinstance(image_output, dict):
		image = image_output['images'][0]
	image.save(save_path)


def get_img_folder(save_folder, model_name) -> Path:
	"""
	desc: get or create folder to save images in
	"""
	img_folder = Path(f"{save_folder}{model_name}_images")
	if not os.path.isdir(img_folder):
		os.mkdir(img_folder)
	return img_folder

def get_selected_prompt_dataset(prompt_ds: Optional[Dataset] = None, prompt_range: Optional[Tuple[int, int]] = None) -> Dataset:
	if prompt_ds is None:
		prompt_ds = load_dataset("semaj83/ioqm", data_files='mini_prompts.txt')['train']
		
	if prompt_range is not None:
		prompt_ds = prompt_ds.select(range(prompt_range[0], prompt_range[1]))
	
	return prompt_ds


def get_img_gen_pipe(model_id: str) -> Any:
	"""
	returns: an image generation pipeline
	"""
	if model_id == "runwayml/stable-diffusion-v1-5":
			pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
			pipe = pipe.to("cuda")
			return pipe
	elif model_id == "stabilityai/stable-diffusion-xl-refiner-1.0":
			base, refiner = get_sdxl_components()
			return partial(ensemble_pipe, base=base, refiner=refiner)
	else:
		raise ValueError("model not supported")


def get_sdxl_components():
	# load both base & refiner
	base = DiffusionPipeline.from_pretrained(
			"stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
	)
	base.to("cuda")
	refiner = DiffusionPipeline.from_pretrained(
			"stabilityai/stable-diffusion-xl-refiner-1.0",
			text_encoder_2=base.text_encoder_2,
			vae=base.vae,
			torch_dtype=torch.float16,
			use_safetensors=True,
			variant="fp16",
	)
	refiner.to("cuda")
	refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
	return base, refiner


def ensemble_pipe(prompt, base: Any, refiner: Any):
	# Define how many steps and what % of steps to be run on each experts (80/20) here
	n_steps = 40
	high_noise_frac = 0.8

	# run both experts
	image = base(
			prompt=prompt,
			num_inference_steps=n_steps,
			denoising_end=high_noise_frac,
			output_type="latent",
	).images

	image = refiner(
			prompt=prompt,
			num_inference_steps=n_steps,
			denoising_start=high_noise_frac,
			image=image,
	).images[0]

	return image





