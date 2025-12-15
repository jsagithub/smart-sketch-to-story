import argparse
from cnn.predict import predict_image_bytes
from rnn.inference_rnn import load_resources, generate_story


def run_pipeline_from_image(sketch_path: str):
    # Predict Image in CNN
    with open(sketch_path, "rb") as f:
      cnn_prediction = predict_image_bytes(f.read())
    print(f"CNN Prediction: {cnn_prediction}")
    # Run RNN to generate story
    model, dataset = load_resources()
    story = generate_story(model, dataset, cnn_prediction)
    return story

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="End-to-end: sketch -> objects -> story -> (image)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--image", type=str, help="Path to sketch image (PNG/JPG)")

    args = ap.parse_args()

    if args.image:
        out = run_pipeline_from_image(
            sketch_path=args.image,
        )
    else:
        raise NotImplementedError("Only --image input is currently supported in CLI.")


    print("Pipeline output:", out)

if __name__ == "__main__":
    main()
