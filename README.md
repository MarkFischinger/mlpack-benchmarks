<img src="https://camo.githubusercontent.com/97b6c8f302fe31570fe0ea3bc0153f018d277f0a83cf34e4b89eaf84de839789/68747470733a2f2f63646e2e6a7364656c6976722e6e65742f67682f6d6c7061636b2f6d6c7061636b2e6f72674065376433366564382f6d6c7061636b2d626c61636b2e737667" width="200">

# MLPack Algorithm Optimization Benchmarks

## Introduction

This repository is dedicated to the research and testing of optimization techniques for neural networks and k-means clustering algorithms using the MLPack library. The benchmarks aim to improve algorithm efficiency and accuracy through systematic testing.

## System Requirements

### C++ Environment
- C++11 compatible compiler
- MLPack library installed

### Python Environment
- Python version 3.8 or higher
- Install dependencies: `pip install -r requirements.txt`

## Dataset

Download the required dataset from Kaggle (for the mnist benchmark):
- [Digit Recognizer Dataset](https://www.kaggle.com/c/digit-recognizer/data)

## Installation Instructions

1. **Clone the Repository:**

```
git clone https://your-repository-url
cd your-repository-directory
```

2. **Compile C++ Code with:**
`g++ -std=c++11 your_cpp_code.cpp -o output_name -larmadillo -lmlpack` 


3. **Setup Python Environment:**
`pip install -r requirements.txt`


## Repository Structure

- **src/**: Source files, with subfolders corresponding to specific issues or PRs in the MLPack library.
- **default/**: Contains benchmark templates, including OpenMP implementation (tested for RDTSC).


## References

Each subfolder within `src` correlates to specific issues or PRs in the MLPack library, linking back to the exact optimizations discussed.




