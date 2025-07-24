#!/bin/bash

# Loop through all .yaml files in conf/ directory that start with "C"
for file in C*.yaml; do
    # Check if file exists (in case no matching files)
    if [[ -f "$file" ]]; then
        # Extract filename without path and extension
        filename=$(basename "$file" .yaml)
        
        # Replace the model_variant line using sed
        sed -i "s/^model_variant:.*$/model_variant: $filename/" "$file"
        
        echo "Updated $file with model_variant: $filename"
    fi
done

echo "All files updated successfully!"
