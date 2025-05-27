# Memory-Efficient-Logits
A benchmarking tool for testing memory-efficient logits computation against standard implementations with faithful forward+backward pass measurements.

**Features:**

Memory-Efficient Implementation: Custom autograd function with chunked processing
Faithful Benchmarking: Complete forward+backward pass measurement
GPU Memory Tracking: Accurate total and additional memory usage
Performance Analysis: Detailed timing and speedup analysis
Visualizations: Automated performance charts and graphs
Breaking Point Detection: Identifies when standard methods fail due to OOM

**Key Results:**
MetricFindingMemory Reduction: 2.5-3.7x less memory usage. Speed Impact: 5-15% slower during training. Scale Enablement: 2-3x larger vocabularies possibleCost Impact: 60-80% reduction in GPU costs


**Installation:** 

**bash git clone https://github.com/NeelM0906/memory-efficient-logits-stress-test.git**

**cd memory-efficient-logits-stress-test**

**pip install -r requirements.txt**

**Usage:**

**bash python stress_test_memory_efficient_logits.py**


Performance Tradeoff

Memory-efficient approach uses 60-75% less memory
Training is 5-15% slower due to chunked processing
Enables training with much larger vocabularies on same hardware

**When to Use:**

Use Memory-Efficient Approach:

Large vocabulary models (>100K tokens)
GPU memory constrained environments
Training phases where cost matters more than speed
Multi-language models with massive vocabularies

Use Standard Approach:

Inference serving where latency is critical
Small vocabulary models (<50K tokens)
Abundant GPU memory available

**Configuration**
Modify test parameters in the script:
python# Vocabulary sizes to test
vocab_sizes = [10_000, 25_000, 50_000, 75_000, 100_000]

# Model dimensions
bsz = 4          # Batch size
seq_len = 2048   # Sequence length
hidden_dim = 2048 # Hidden dimension
Implementation Example
pythonfrom stress_test_memory_efficient_logits import memory_efficient_cross_entropy

# Replace standard cross entropy
input_tensor = torch.randn(8192, 4096, device='cuda', requires_grad=True)
weight_matrix = torch.randn(4096, 100000, device='cuda', requires_grad=True)
target_labels = torch.randint(0, 100000, (8192,), device='cuda')


**Troubleshooting:**

CUDA Out of Memory: Reduce batch size or vocabulary size
Slow performance: Normal for large vocabularies due to chunked processing
Visualization errors: Install matplotlib: pip install matplotlib

**License:**
MIT License 
Contributing
Contributions welcome! Please feel free to submit issues and pull requests.
