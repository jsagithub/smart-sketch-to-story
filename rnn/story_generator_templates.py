import random

def simple_template(objects):
    objects_str = ", ".join(objects[:-1]) + " and " + objects[-1] if len(objects) > 1 else objects[0]

    templates = [
        f"A scene with {objects_str}.",
        f"The image shows {objects_str} together.",
        f"{objects_str.capitalize()} appear in this drawing.",
        f"In this sketch, you can see {objects_str}.",
        f"This scene contains {objects_str}.",
        f"{objects_str.capitalize()} are part of the illustration.",
    ]
    return random.choice(templates)

def add_style_variation(story):
    styles = [
        "",
        " It looks peaceful.",
        " The scene feels lively.",
        " The atmosphere is calm.",
        " Sketched with a simple style.",
        " Drawn in a minimalistic way.",
    ]
    return story + random.choice(styles)

def generate_story(objects):
    base = simple_template(objects)
    return add_style_variation(base)