# M2/M1 Optimization Guide

## Performance Optimizations

### Docker Configuration
- Native ARM64 container support
- Optimized memory and CPU allocation
- Thread management for Apple Silicon

### Python/PyTorch Settings
- OpenBLAS threading configuration
- Worker process optimization
- Memory management tuning

## Benchmarks
- Docker: ~1.7 it/s
- Native Python: ~1.6 it/s
- Original Docker: ~0.05 it/s (19s/it)

## Future GPU Support
Expected performance on NVIDIA GPUs:
- RTX 4090: ~15-20 it/s
- RTX 3090: ~10-15 it/s
- A100: ~25-30 it/s

Required changes for GPU support:
```yaml
services:
  cktgnn:
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Configuration Files
Key changes in:
- Dockerfile: ARM64 base, optimized dependencies
- docker-compose.yml: Resource allocation
- main.py: Worker and batch settings 