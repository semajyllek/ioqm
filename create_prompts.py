
from itertools import combinations, product
from textblob import TextBlob
from pathlib import Path
from typing import List
import random

SAVE_PATH = "prompts.txt"
MINI_SAVE_PATH = "mini_prompts.txt"
MINI_SAVE_PATH_V2 = "mini_prompts_v2.txt"

# COCO Classes, from https://docs.ultralytics.com/datasets/detect/coco/#dataset-yaml
OBJECT_PATH = Path(__file__).parent / "coco_classes.txt"
V2_OBJECT_PATH = Path(__file__).parent / "coco_classes_v2.txt"  # minus 10 random objects from v1

MAX_IMG_OBJECTS = 2
MAX_OBJ_QUANTITY = 3
NUM_OBJECT_CLASSES = 30 # 10 for v1, 30 for v2



def get_objects(object_path: Path = OBJECT_PATH) -> List[str]:
    with open(object_path, 'r') as f:
        objects = [o.strip('\n') for o in f.readlines()]
    return objects
    

def gen_prompt_quantifier_helper(object: str, n: int) -> str:
    """
    object:  str, ex: 'person', 'car', 'dog', etc.
    returns: pluralized object with quantity, ex: '1 person', '2 people', '3 people', etc.
    """
    if n > 1:
        object = TextBlob(object).words.pluralize().pop() # dog -> dogs, person -> people, etc.
        
    return f"{n} {object}"



def gen_single_object_prompts(objects: List[str]) -> List[str]:
    prompts = []
    for obj in objects:
        prompts.extend(gen_prompt_quantifier_helper(obj))
    return prompts  

def generate_prompts(objects: List[str], max_img_objects: int, max_obj_quantity: int, joiner: str):
    """
    desc: Generates list of all possible prompt combinations and saves them for model evaluation
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

    objects: list of str objects to generate prompts for
    max_img_objects: max number of objects in an image
    max_obj_quantity: max quantity of an object in an image
    joiner: str, ex: ",", "and", "with", "holding", etc.

    returns: list of prompt strings
    """
    prompts = []
    for num_img_objects in range(1, max_img_objects + 1):
        for combo in combinations(objects, num_img_objects):
            for count_permute in product(range(1, max_obj_quantity + 1), repeat=num_img_objects):
                prompt = ""
                for i, obj in enumerate(combo):
                    if i > 0:
                        prompt += f" {joiner} " if (joiner != ",") else ", "
                    prompt += gen_prompt_quantifier_helper(obj, count_permute[i])

                prompts.append(prompt)
                
    return prompts

def save_prompts(prompts: List[str], save_path: str = SAVE_PATH):
    with open(save_path, 'w') as f:
        for prompt in prompts:
            f.write(prompt + '\n')
    


    

if __name__ == "__main__":

    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--n_objects", type=int, default=NUM_OBJECT_CLASSES)
    parser.add_argument("--max_objects", type=int, default=MAX_IMG_OBJECTS)
    parser.add_argument("--max_quantity", type=int, default=MAX_OBJ_QUANTITY)
    parser.add_argument("--save_path", type=str, default=MINI_SAVE_PATH_V2)
    parser.add_argument("--joiner", type=str, default="and")
    args = parser.parse_args()

    objects = get_objects(OBJECT_PATH)
    if args.mini:
        objects = random.sample(objects, args.n_objects)

    objects = ['sheep', 'potted plant', 'ski', 'orange', 'kite']

    prompts = generate_prompts(objects, args.max_objects, args.max_quantity, args.joiner)
    save_prompts(prompts, args.save_path)



