import json

output = {
    "image_name": "genomeflow-predictor",
    "image_tag": "v1",
    "base_image": "python:3.10-slim",
    "test_input": {
        "read_length_bp": 5865,
        "sample_quality_score": 6.2,
        "coverage_depth": 47,
        "is_whole_genome": 1
    },
    "prediction": 0.0
}

with open("results/step2_s3.json", "w") as f:
    json.dump(output, f, indent=4)

print("Task 2 completed")