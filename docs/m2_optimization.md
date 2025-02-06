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

## Configuration Files
Key changes in:
- Dockerfile: ARM64 base, optimized dependencies
- docker-compose.yml: Resource allocation
- main.py: Worker and batch settings 