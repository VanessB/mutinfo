#!/bin/bash

# Script to clean up model checkpoints, keeping only last.ckpt or the highest epoch checkpoint

set -e

# Check if directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

TARGET_DIR="$1"

if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' does not exist"
    exit 1
fi

echo "Searching for checkpoints in: $TARGET_DIR"

# Find all directories containing checkpoint files
find "$TARGET_DIR" -type f -name "*.ckpt" -o -name "*.pth" -o -name "*.pt" | \
    xargs -r dirname | sort -u | while read -r dir; do
    
    echo ""
    echo "Processing directory: $dir"
    
    # Check if last.ckpt (or variants) exists
    LAST_CKPT=$(find "$dir" -maxdepth 1 -type f \( \
        -name "last.ckpt" -o \
        -name "last.pth" -o \
        -name "last.pt" \) | head -n 1)
    
    if [ -n "$LAST_CKPT" ]; then
        echo "  Found last checkpoint: $(basename "$LAST_CKPT")"
        echo "  Removing all other checkpoints..."
        
        # Remove all checkpoints except the last one
        find "$dir" -maxdepth 1 -type f \( \
            -name "*.ckpt" -o \
            -name "*.pth" -o \
            -name "*.pt" \) ! -name "$(basename "$LAST_CKPT")" \
            -exec echo "    Deleting: {}" \; -delete
    else
        echo "  No 'last' checkpoint found. Looking for highest epoch..."
        
        # Find checkpoint with highest epoch number
        # Matches patterns like: epoch_epoch=999.ckpt, epoch=999.ckpt, epoch_999.ckpt, etc.
        HIGHEST_EPOCH_FILE=$(find "$dir" -maxdepth 1 -type f \( \
            -name "*.ckpt" -o \
            -name "*.pth" -o \
            -name "*.pt" \) | \
            grep -E "epoch[_=\-]*(epoch=)?[0-9]+" | \
            sed 's/.*epoch[_=\-]*\(epoch=\)\?\([0-9]\+\).*/\2 &/' | \
            sort -rn | head -n 1 | cut -d' ' -f2-)
        
        if [ -n "$HIGHEST_EPOCH_FILE" ]; then
            EPOCH_NUM=$(echo "$HIGHEST_EPOCH_FILE" | grep -oE "epoch[_=\-]*(epoch=)?[0-9]+" | grep -oE "[0-9]+$")
            echo "  Found highest epoch checkpoint: $(basename "$HIGHEST_EPOCH_FILE") (epoch=$EPOCH_NUM)"
            echo "  Removing all other checkpoints..."
            
            # Remove all checkpoints except the highest epoch one
            find "$dir" -maxdepth 1 -type f \( \
                -name "*.ckpt" -o \
                -name "*.pth" -o \
                -name "*.pt" \) ! -name "$(basename "$HIGHEST_EPOCH_FILE")" \
                -exec echo "    Deleting: {}" \; -delete
        else
            echo "  No epoch-numbered checkpoints found. Skipping directory."
        fi
    fi
done

echo ""
echo "Cleanup complete!"