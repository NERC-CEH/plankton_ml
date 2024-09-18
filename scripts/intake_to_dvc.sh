#!/bin/bash

#!/bin/bash

# Load environment variables from .env file
source .env

# Download the intake "catalog" (really a filename listing)
URL="${AWS_URL_ENDPOINT}/untagged-images-lana/catalog.csv"
wget -O catalog.csv "$URL"

# Add tracking for each file to DVC
while IFS= read -r filename; do
  if [[ "$filename" == *.tif ]]; then
    dvc import-url --no-download "$filename" data/
  fi
done < catalog.csv

# Add the tracked files to git
git add "data/*.dvc"
