import numpy as np
import matplotlib.pyplot as plt

N = 10000  # Total population size
beta = 0.3  # Infection rate
gamma = 0.05  # Recovery rate

# Initialize population matrix: 0=Susceptible, 1=Infected, 2=Recovered
population = np.zeros((100, 100))

# Set initial outbreak
outbreak = np.random.choice(range(100), 2)
population[outbreak[0], outbreak[1]] = 1  # Mark the outbreak point as infected

# Simulate the spread
for t in range(101):
    new_population = population.copy()  # Copy the current state
    
    # Find all infected points
    infectedIndex = np.where(population == 1)
    
    # Infect neighbors
    for i in range(len(infectedIndex[0])):
        x = infectedIndex[0][i]
        y = infectedIndex[1][i]
        
        # Infect all 8 neighbors(this is a bit finicky, is there a better way?)
        for xNeighbour in range(x-1, x+2):
            for yNeighbour in range(y-1, y+2):
                # Ensure not infecting itself and within bounds(Is this strictly necessary?)
                if (xNeighbour, yNeighbour) != (x, y) and 0 <= xNeighbour < 100 and 0 <= yNeighbour < 100:
                    # Only infect susceptible neighbors
                    if population[xNeighbour, yNeighbour] == 0:
                        if np.random.rand() < beta:
                            new_population[xNeighbour, yNeighbour] = 1  # Infect susceptible neighbor
    
    # Recover infected individuals
    for i in range(len(infectedIndex[0])):
        x = infectedIndex[0][i]
        y = infectedIndex[1][i]
        if np.random.rand() < gamma:
            new_population[x, y] = 2  # Mark the infected individual as recovered
    
    # Update the population matrix
    population = new_population.copy()
    
    # Plot the population state at specific time steps
    if t in [0, 10, 50, 100]:
        plt.figure(figsize=(6, 4), dpi=150)
        plt.imshow(population, cmap='viridis', interpolation='nearest')
        plt.title(f'Time step: {t}')
        plt.colorbar(label='State (0=Susceptible, 1=Infected, 2=Recovered)')
        plt.show()
