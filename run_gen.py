
"""
loads a dataset of prompts and generates images from them using a given model
"""


from typing import Optional, Tuple
from datasets import load_dataset
import os


def run_gen(model_id: str, prompt_range: Optional[Tuple[int, int]] = None, icl_suffix: str = " on a table."):
	prompt_ds = load_dataset("semaj83/ioqm", data_files='mini_prompts.txt')['train']
	if prompt_range is not None:
		prompt_ds = prompt_ds.select(range(prompt_range[0], prompt_range[1]))
		
	img_gen_model = get_gen_model(model_id)
	model_name = model_id.split('/')[-1]
	if not os.path.isdir(f"{model_name}_images"):
		os.mkdir(f"{model_name}_images") 
	
	for prompt in prompt_ds:
		icl_prompt = prompt['text'] + icl_suffix
		image = img_gen_model(icl_prompt).images[0]  
		image.save(f"{model_name}_images/{'_'.join(prompt['text'].split())}.png") # ex. 1_microwave_and_3_fire_hydrants_and_2_handbags.png
		

def get_gen_model(model_id: str):
	if model_id == "runwayml/stable-diffusion-v1-5":
		from diffusers import StableDiffusionPipeline
		import torch

		pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
		pipe = pipe.to("cuda")
		return pipe
	else:
		raise ValueError("model not supported")
	



