## image object quantity metric (IOQM)

This repo contains code for:

1. building a dataset of image prompts with quantities of objects for evaluation
   - the objects are a set of 79/80 objects used in the COCO dataset. https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml

   - a mini version of the dataset containing ~16k images from a random subset of 10 COCO object classes lives here: https://huggingface.co/datasets/semaj83/ioqm
2. running various generative models through huggingface diffusers/transformers libraries
3. evaluating images generated using the original object detection model developed by facebook for the dataset: `facebook/detr-resnet-50`


### paper: TBD

### results: TBD

***code to create prompts***:
```python create_prompts.py --mini=True```

***code to generate images***:
```python run_gen.py --model='stable-diffusion-v1-5'```

***code to evaluate generated images***:
```python run_eval.py```