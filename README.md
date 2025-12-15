# smart-sketch-to-story
A system that takes a hand-drawn sketch, identifies what's in it, then generates a short description or story.

# Start Up
- Download parquet files from https://huggingface.co/datasets/sdiaeyu6n/tu-berlin/tree/main/data To: data/datasets
- run the script create_class_mapping.py to create class mapping file at cnn folder
- Train CNN by running python cnn/train_cnn.py
- Check CNN predictions:
  - Add an image in data/sketches_raw
  - Run prediction script: python cnn/predict.py
- Generate stories dataset by running rnn/story_generator_templates.py
- Train RNN by running python rnn/train_rnn.py
- Check RNN predictions:
  - Run prediction script: python rnn/inference_rnn.py
-Run pipeline:
  python -m app.pipeline \
  --image data/sketches_raw/car.jpeg