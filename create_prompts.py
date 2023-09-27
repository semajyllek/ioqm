
from itertools import combinations, product
from pathlib import Path
from typing import List
import random

SAVE_PATH = "prompts.txt"
MINI_SAVE_PATH = "mini_prompts2.txt"

# COCO Classes, from https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml
OBJECT_PATH = Path(__file__).parent / "coco_classes.txt"

MAX_IMG_OBJECTS = 3
MAX_OBJ_QUANTITY = 5



def get_objects() -> List[str]:
    with open(OBJECT_PATH) as f:
        objects = [o.strip('\n') for o in f.readlines()]
    return objects


def get_plural_suffix(object: str) -> str:
    suffix = ""
    if object == "person":
        object = "people"

    elif object.endswith("s") or object.endswith("sh") or object.endswith("ch"):
        suffix = "es"

    elif object.endswith("y"):
        suffix = "ies"
        object = object[:-1]
            
    else:
        suffix = "s"
    
    object += suffix
    return object

    

def gen_prompt_quantifier_helper(object: str, n: int) -> str:
    """
    object:  str, ex: 'person', 'car', 'dog', etc.
    returns: list of object strings with quantifier prefix and pluralization
    """
    if n > 1:
        object = get_plural_suffix(object)
        
    return f"{n} {object}"



def gen_single_object_prompts(objects: List[str]) -> List[str]:
    prompts = []
    for obj in objects:
        prompts.extend(gen_prompt_quantifier_helper(obj))
    return prompts  

def generate_prompts(objects):
    """
    Generates a list of prompts and saves them for model evaluation
    e.g. 
        1 person
        2 people
        3 people
        1 person and 1 car
        2 people and 1 car
        3 people and 1 car
        1 person and 2 cars
        2 people and 2 cars
        3 people and 2 cars
        ...
    """
    prompts = []
    for num_img_objects in range(1, MAX_IMG_OBJECTS + 1):
        for combo in combinations(objects, num_img_objects):
            for count_permute in product(range(1, MAX_OBJ_QUANTITY + 1), repeat=num_img_objects):
                prompt = ""
                for i, obj in enumerate(combo):
                    if i > 0:
                        prompt += " and "
                    prompt += gen_prompt_quantifier_helper(obj, count_permute[i])

                prompts.append(prompt)
                
    return prompts


def save_prompts(prompts: List[str], save_path: str = SAVE_PATH):
    with open(save_path, 'w') as f:
        for prompt in prompts:
            f.write(prompt + '\n')
    


    

if __name__ == "__main__":
    objects = get_objects()
    prompts = generate_prompts(objects)
    save_prompts(prompts)

    # mini_objects = random.sample(objects, 10)
    # prompts = generate_prompts(mini_objects)
    # save_prompts(prompts, MINI_SAVE_PATH)



