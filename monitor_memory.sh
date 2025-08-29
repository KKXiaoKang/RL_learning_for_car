#!/bin/bash

# Memory monitoring script for Q-chunking training

echo "=== Memory Monitoring Started ==="
while true; do
    echo "$(date): Memory usage:"
    free -h | grep "内存"
    echo "GPU memory:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    echo "Python processes using most memory:"
    ps aux --sort=-%mem | grep python | head -3
    echo "---"
    sleep 10
done
