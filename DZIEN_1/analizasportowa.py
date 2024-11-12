import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Generowanie danych finansowych
np.random.seed(1)
dni = np.arange(1, 101)
ceny = 100 + np.cumsum(np.random.normal(0, 1, 100))

# Tworzenie DataFrame
df = pd.DataFrame({
    'Dzień': dni,
    'Cena': ceny
})

# Obliczanie średnich kroczących
df['Średnia 5-dniowa'] = df['Cena'].rolling(window=5).mean()
df['Średnia 20-dniowa'] = df['Cena'].rolling(window=20).mean()

# Wizualizacja
plt.figure(figsize=(10, 5))
plt.plot(df['Dzień'], df['Cena'], label='Cena akcji', color='blue')
plt.plot(df['Dzień'], df['Średnia 5-dniowa'], label='Średnia 5-dniowa', color='orange')
plt.plot(df['Dzień'], df['Średnia 20-dniowa'], label='Średnia 20-dniowa', color='green')
plt.xlabel('Dzień')
plt.ylabel('Cena akcji (PLN)')
plt.title('Analiza cen akcji')
plt.legend()
plt.show()
