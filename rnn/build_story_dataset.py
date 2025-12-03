import json

from rnn.story_generator_template import get_object_sets
from rnn.story_generator_templates import generate_story

object_sets = get_object_sets(1000)
with open("../data/stories/story_dataset.jsonl", "w") as f:
    for objs in object_sets:
        story = generate_story(objs)
        f.write(json.dumps({"objects": objs, "story": story}) + "\n")

print("Dataset created: rnn/story_dataset.jsonl")