#!/bin/bash
# Script to manually download ImageNet pretrained weights for ResNet-18
# Use this if SSL certificate issues prevent automatic download

echo "=========================================="
echo "Manual ResNet-18 Weight Download Script"
echo "=========================================="

# Set cache directory
CACHE_DIR="${HOME}/.cache/torch/hub/checkpoints"
mkdir -p "$CACHE_DIR"

# ResNet-18 ImageNet weights URL
WEIGHT_URL="https://download.pytorch.org/models/resnet18-f37072fd.pth"
WEIGHT_FILE="resnet18-f37072fd.pth"

echo ""
echo "Downloading to: ${CACHE_DIR}/${WEIGHT_FILE}"
echo "URL: ${WEIGHT_URL}"
echo ""

# Try wget first (no certificate check)
if command -v wget &> /dev/null; then
    echo "Using wget..."
    wget --no-check-certificate -O "${CACHE_DIR}/${WEIGHT_FILE}" "${WEIGHT_URL}"
    if [ $? -eq 0 ]; then
        echo "✓ Download successful with wget!"
        ls -lh "${CACHE_DIR}/${WEIGHT_FILE}"
        exit 0
    fi
fi

# Try curl as fallback
if command -v curl &> /dev/null; then
    echo "Using curl..."
    curl -k -L -o "${CACHE_DIR}/${WEIGHT_FILE}" "${WEIGHT_URL}"
    if [ $? -eq 0 ]; then
        echo "✓ Download successful with curl!"
        ls -lh "${CACHE_DIR}/${WEIGHT_FILE}"
        exit 0
    fi
fi

echo "✗ Download failed. Please check your network connection."
echo ""
echo "Alternative: Copy weights from another machine:"
echo "  scp <user>@<machine>:~/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth ${CACHE_DIR}/"
exit 1
