```bash
#!/bin/bash

# Install system-level dependencies
if [ "$(uname)" == "Darwin" ]; then
    # macOS
    brew install pkg-config cairo git-lfs
elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Linux (Generic)
    sudo apt-get update
    sudo apt-get install pkg-config libcairo2-dev git-lfs
fi

# Initialize Git LFS
git lfs install

# Install Python dependencies
pip install -r requirements.txt

# Pull all LFS files
git lfs pull
