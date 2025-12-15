import pandas as pd
import random
from data.datasets.items import items

def get_object_sets(n_samples=1000)->list:
    df = pd.read_parquet("../data/datasets/train-00000-of-00001.parquet")

    # If label column is "category" (string)
    if "category" in df.columns:
        classes = sorted(df["category"].unique())

    # If label column is "label" (numeric id)
    elif "label" in df.columns:
        classes_ids = sorted(df["label"].unique())

        classes = [items[i] for i in classes_ids]

    def sample_object_sets(classes, n_samples, max_objects=3):
        samples = []
        for _ in range(n_samples):
            k = random.randint(1, max_objects)
            objs = random.sample(classes, k)
            samples.append(objs)
        return samples

    return sample_object_sets(classes, n_samples)