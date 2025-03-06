#!/bin/bash

# Activate conda environment (環境名は適宜変更してください)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate your_env_name

# Create requirements file
echo "# Package versions" > package_versions.txt
echo "# Generated on $(date)" >> package_versions.txt
echo "" >> package_versions.txt

# Check versions of required packages
packages=(
    "numpy"
    "torch"
    "torchvision"
    "pandas"
    "scikit-learn"
    "matplotlib"
    "seaborn"
    "tensorboardX"
    "tqdm"
    "pillow"
    "h5py"
    "opencv-python"
    "pynvml"
    "google-auth"
    "google-auth-oauthlib"
    "google-api-python-client"
    "paramiko"
    "chardet"
    "omegaconf"
    "pyyaml"
    "japanize-matplotlib"
)

for package in "${packages[@]}"
do
    version=$(python -c "import pkg_resources; print(f'$package=={pkg_resources.get_distribution(\"$package\").version}')" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "$version" >> package_versions.txt
    else
        echo "# $package not found" >> package_versions.txt
    fi
done

# Add Python version
echo "" >> package_versions.txt
echo "# Python version" >> package_versions.txt
python --version >> package_versions.txt

# Add CUDA version if available
echo "" >> package_versions.txt
echo "# CUDA version" >> package_versions.txt
if command -v nvcc &> /dev/null; then
    nvcc --version >> package_versions.txt
else
    echo "CUDA not found" >> package_versions.txt
fi

echo "Package versions have been written to package_versions.txt" 