import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def demonstrate_defuzzification(universe, aggregated_mf):

    
    cog = fuzz.defuzz(universe, aggregated_mf, 'centroid')
    
    boa = fuzz.defuzz(universe, aggregated_mf, 'bisector')
    
    mom = fuzz.defuzz(universe, aggregated_mf, 'mom')
    
    som = fuzz.defuzz(universe, aggregated_mf, 'som')
    
    lom = fuzz.defuzz(universe, aggregated_mf, 'lom')
    
    print("--- Defuzzification Results ---")
    print(f"Centroid (CoG):        {cog:.4f}")
    print(f"Bisector (BoA):        {boa:.4f}")
    print(f"Mean of Maximum (MoM): {mom:.4f}")
    print(f"Smallest of Max (SoM): {som:.4f}")
    print(f"Largest of Max (LoM):  {lom:.4f}")

    plt.figure(figsize=(10, 6))
    plt.plot(universe, aggregated_mf, 'b', linewidth=2.5, label='Aggregated Fuzzy Set')
    
    plt.axvline(cog, color='r', linestyle='--', label=f'Centroid ({cog:.2f})')
    plt.axvline(boa, color='g', linestyle='-.', label=f'Bisector ({boa:.2f})')
    plt.plot([mom, mom], [0, 1.0], 'k:', label=f'MoM ({mom:.2f})') # Using a vertical line for MoM
    plt.plot([som, som], [0, 1.0], 'c:', label=f'SoM ({som:.2f})')
    plt.plot([lom, lom], [0, 1.0], 'm:', label=f'LoM ({lom:.2f})')
    
    plt.title('Comparison of Defuzzification Techniques')
    plt.ylabel('Membership Degree')
    plt.xlabel('Output Universe')
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True)
    plt.show()
    

X = np.arange(0, 26, 0.1)

mf_1 = fuzz.trapmf(X, [0, 5, 8, 11])
mf_2 = fuzz.trimf(X, [9, 15, 25])

aggregated_mf = np.fmax(mf_1 * 0.7, mf_2 * 1.0) # Assume mf_1 was scaled down by rule strength 0.7

demonstrate_defuzzification(X, aggregated_mf)