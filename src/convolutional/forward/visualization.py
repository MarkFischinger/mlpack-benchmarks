import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('results.csv')
data.set_index('Size', inplace=True)

plt.figure(figsize=(10, 7))

plt.scatter(data.index, data['ForwardPassTime'], color='blue', s=50, label='Forward Pass Time')
plt.plot(data.index, data['ForwardPassTime'], linestyle='-', color='blue', linewidth=2)

plt.scatter(data.index, data['TraditionalConvolutionTime'], color='red', s=50, label='Traditional Convolution Time')
plt.plot(data.index, data['TraditionalConvolutionTime'], linestyle='-', color='red', linewidth=2)

plt.title('Performance Comparison', fontsize=16)
plt.xlabel('Size', fontsize=14)
plt.ylabel('Time', fontsize=14)
plt.xscale('log')  
plt.yscale('log')  
plt.grid(True, linestyle='--', linewidth=0.5)
plt.legend(fontsize=12)
plt.savefig('results.png', dpi=300)
