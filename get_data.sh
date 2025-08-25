#!/bin/bash

# Define the source file and download directory
SOURCE_FILE="data_source.txt"
DOWNLOAD_DIR="raw_data"

# Create the download directory if it doesn't exist
mkdir -p "$DOWNLOAD_DIR"

# Check if the source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file '$SOURCE_FILE' not found!"
    exit 1
fi

# Filter valid URLs and download them to the specified directory
echo "Starting downloads from $SOURCE_FILE..."
grep -v '^#' "$SOURCE_FILE" | while read url; do
    file_name=$(basename "$url")
    target_path="$DOWNLOAD_DIR/$file_name"

    # Skip downloading if the file already exists
    if [ -f "$target_path" ]; then
        echo "File already exists: $target_path. Skipping..."
        continue
    fi

    echo "Downloading: $url -> $target_path"
    wget -O "$target_path" "$url"

    # Check if the downloaded file is a ZIP file
    if [[ "$file_name" == *.zip ]]; then
        # Extract the base name of the ZIP file (without directory and .zip extension)
        base_name=$(basename "$file_name" .zip)
        target_dir="$DOWNLOAD_DIR/$base_name"

        # Create a directory with the ZIP file's name
        mkdir -p "$target_dir"

        # Extract the ZIP file into the newly created directory
        echo "Extracting: $file_name -> $target_dir"
        unzip -o "$target_path" -d "$target_dir"

        # Delete the original ZIP file
        echo "Deleting: $target_path"
        rm -f "$target_path"
    else
        # Extract the base name of the libsvm file (without directory and .scale extension)
        base_name=$(basename "$file_name" .scale)
        target_dir="$DOWNLOAD_DIR/$base_name"

        # Create a directory with the libsvm file's name
        mkdir -p "$target_dir"
        mv "$target_path" "$target_dir"

        echo "File is not a ZIP: $file_name. Skipping extraction."
    fi
done

# Completion message
echo "All downloads completed. ZIP files extracted and non-ZIP files saved!"