# smart-sketch-to-story
A system that takes a hand-drawn sketch, identifies what's in it, then generates a short description or story, and finally produces a stylized generated image based on the story

# Start Up
- Download parquet files from https://huggingface.co/datasets/sdiaeyu6n/tu-berlin/tree/main/data To: data/datasets
- Train CNN by running python cnn/train_cnn.py
- Check CNN predictions:
  - Add an image in data/sketches_raw
  - Run prediction script: python cnn/predict.py
- Generate stories by running rnn/story_generator_templates.py