"""

script to parse objects and quantities from string
5 wine glasses and 5 cups and 4 umbrellas
"""


from typing import List, Tuple

def parse_objects_and_quantities(prompt: str) -> List[Tuple[str, int]]:
	"""
	prompt: str, e.g. "5 wine glasses and 5 cups and 4 umbrellas"
	returns: list of tuples, e.g. [("wine glass", 5), ("cup", 5), ("umbrella", 4)]
	"""
	objects = []
	quantities = []
	obj_text = ""
	for word in prompt.split():
		if word == "and":
			continue
		try:
			quantities.append(int(word))
			if obj_text != "":
				objects.append(obj_text)
				obj_text = ""
		except ValueError:
			if obj_text == "":
				obj_text = word
			else:
				obj_text += f" {word}"
	objects.append(obj_text)
	return list(zip(objects, quantities))



if __name__ == "__main__":
	prompt = "5 wine glasses and 5 cups and 4 umbrellas"
	# [('wine glasses', 5), ('cups', 5), ('umbrellas', 4)]
	print(parse_objects_and_quantities(prompt))