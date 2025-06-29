import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the data
data_file = Path("results/curves/si2_ccsd_data.csv")
df = pd.read_csv(data_file)

# Zero the data at the last point
last_energy = df['Energy'].iloc[-1]
df['Energy_zeroed'] = df['Energy'] - last_energy

# Find the minimum
min_idx = df['Energy_zeroed'].idxmin()
min_distance = df.loc[min_idx, 'Distance']
min_energy = df.loc[min_idx, 'Energy_zeroed']

print(f"Minimum found at:")
print(f"  Distance: {min_distance:.3f} Å")
print(f"  Energy: {min_energy:.6f} Hartree")

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(df['Distance'], df['Energy_zeroed'], 'b-', linewidth=2, label='Si2 CCSD')
plt.plot(min_distance, min_energy, 'ro', markersize=8, label=f'Minimum ({min_distance:.3f} Å)')

plt.xlabel('Distance (Å)')
plt.ylabel('Energy (Hartree)')
plt.title('Si2 Dissociation Curve (Zeroed at Last Point)')
plt.grid(True, alpha=0.3)
plt.legend()

# Add minimum point annotation
plt.annotate(f'Min: ({min_distance:.3f} Å, {min_energy:.6f})', 
             xy=(min_distance, min_energy), 
             xytext=(min_distance + 0.5, min_energy + 0.01),
             arrowprops=dict(arrowstyle='->', color='red', alpha=0.7))

plt.tight_layout()
plt.show()
