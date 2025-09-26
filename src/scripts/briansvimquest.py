import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# F= 0.14255735E+03 E0= 0.14257531E+03  d E =-.359240E-01
#  F= 0.58630809E+01 E0= 0.58632470E+01  d E =-.332325E-03
# F= -.10257445E+02 E0= -.10257445E+02  d E =-.132104E-10
# F= -.11369977E+02 E0= -.11369977E+02  d E =-.101749E-12
# F= -.11370035E+02 E0= -.11370035E+02  d E =-.278124E-12
# F= -.11373166E+02 E0= -.11373166E+02  d E =-.282026E-12
# F= -.11376268E+02 E0= -.11376268E+02  d E =-.260560E-12
# F= -.11381181E+02 E0= -.11381181E+02  d E =-.255835E-12

# Populate the results array with the commented F values above, in order.
# The F values are:
# 0.14255735E+03, 0.58630809E+01, -0.10257445E+02, -0.11369977E+02, -0.11370035E+02,
# -0.11373166E+02, -0.11376268E+02, -0.11381181E+02

results = np.array([
    0.14255735E+03,
    0.58630809E+01,
    -0.10257445E+02,
    -0.11369977E+02,
    -0.11370035E+02,
    -0.11373166E+02,
    -0.11376268E+02,
    -0.11381181E+02
])

# The encutoff array should match the number of results for plotting.
encutoff = np.array([100, 150, 200, 250, 300, 350, 400, 450])

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(encutoff, results, marker='o')
plt.xlabel('ENCUT (eV)')
plt.ylabel('F')
plt.title('F vs ENCUT')
plt.grid(True)
plt.tight_layout()
plt.savefig('f_vs_encut.png')
