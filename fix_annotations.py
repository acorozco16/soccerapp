import json

# Load the annotations
with open('training_data/processed_dataset/metadata/annotations.json', 'r') as f:
    data = json.load(f)

# Add quality_category to each annotation
for annotation in data['annotations']:
    visibility = annotation['ball_visibility']
    
    # Classify based on ball visibility score
    if visibility >= 0.7:
        annotation['quality_category'] = 'high_quality'
    elif visibility >= 0.4:
        annotation['quality_category'] = 'medium_quality'
    else:
        annotation['quality_category'] = 'low_quality'

# Save the updated annotations
with open('training_data/processed_dataset/metadata/annotations.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Updated {len(data['annotations'])} annotations with quality categories")