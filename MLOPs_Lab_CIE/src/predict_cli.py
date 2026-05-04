import argparse
import joblib
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--read_length_bp", type=float, required=True)
parser.add_argument("--sample_quality_score", type=float, required=True)
parser.add_argument("--coverage_depth", type=float, required=True)
parser.add_argument("--is_whole_genome", type=float, required=True)

args = parser.parse_args()

model = joblib.load("models/best_model.pkl")

features = np.array([[
    args.read_length_bp,
    args.sample_quality_score,
    args.coverage_depth,
    args.is_whole_genome
]])

prediction = model.predict(features)[0]

print(prediction)