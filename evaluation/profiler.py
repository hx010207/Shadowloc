import time
import tracemalloc
import psutil
import os

def profile_inference(model, X_test, model_type='sklearn'):
    """
    Measure wall-clock latency (ms/window) and peak memory (MB).
    X_test is a single window or small batch to simulate online inference.
    """
    # Force garbage collection
    import gc
    gc.collect()
    
    # We profile predicting 100 windows one by one to get average per-window latency
    N = min(100, len(X_test))
    X_sample = X_test[:N]
    
    tracemalloc.start()
    start_time = time.perf_counter()
    
    if model_type == 'sklearn':
        for i in range(N):
            _ = model.score(X_sample[i:i+1])
    elif model_type == 'pytorch':
        import torch
        model.model.eval()
        with torch.no_grad():
            for i in range(N):
                # Variant B score function already handles tensor conversion, 
                # but might be overhead. We just call score.
                _ = model.score(X_sample[i:i+1])
                
    end_time = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    latency_ms = ((end_time - start_time) / N) * 1000.0
    peak_memory_mb = peak / (1024 * 1024)
    
    return latency_ms, peak_memory_mb
