import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Tworzenie danych do wykresu
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
x, y = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

# Tworzenie wykresu 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Rysowanie powierzchni
surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# Dodanie tytułów i etykiet
ax.set_title('Złożony wykres 3D: Sinusoidalna powierzchnia')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Dodanie paska kolorów
fig.colorbar(surf)

# Pokazanie wykresu
plt.show()
