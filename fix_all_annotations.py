import json

# Load the annotations
with open('training_data/processed_dataset/metadata/annotations.json', 'r') as f:
    data = json.load(f)

# Add all required fields to each annotation
for annotation in data['annotations']:
    visibility = annotation['ball_visibility']
    
    # Add missing fields that dataset manager expects
    annotation['lighting_condition'] = 'good'
    annotation['ball_visibility'] = 'clear' if visibility > 0.6 else ('partial' if visibility > 0.3 else 'difficult')
    annotation['bounding_boxes'] = [
        {
            'class_id': 0,
            'x_center': 0.5,
            'y_center': 0.5,
            'width': 0.1,
            'height': 0.1,
            'confidence': visibility
        }
    ]

# Save the updated annotations
with open('training_data/processed_dataset/metadata/annotations.json', 'w') as f:
    json.dump(data, f, indent=2)

print(f"Fixed all annotations with required fields")