import matplotlib.pyplot as plt
import numpy as np


################################################################################
# Fonctions                                                                    #
################################################################################

def initialisation_pop(N:int, p_init:float) -> list[int]:
    """Initialise une population de taille N avec une fréquence initiale de l'allèle A
        input :  N = taille de la population
                 p_init = fréquence initiale de l'allèle A
        output : liste remplie de 0 (allèle B) et de 1 (A)"""
    
    num_ones = round(N*p_init)
    num_zeros = N - num_ones
    pop = [1] * num_ones + [0] * num_zeros

    return pop


def new_generation(pop:list[int], N:int, Fitness:dict[int:float]) -> list[int]:
    """Génère une nouvelle génération (liste) à partir de la précédente
        input :  pop = population actuelle (liste d'entiers, chaque entier représente un allèle)
                 N = taille de la population
                 Fitness = dictionnaire contenant les fitness de chaque allèle
        output : new_pop = nouvelle génération"""
    
    fitness_values = np.array([Fitness[allele] for allele in pop]) #fitness pour CHAQUE allèle dans la population (longueur du vecteur : N)
    selection_prob = fitness_values / np.sum(fitness_values) #probabilités de sélection pour chaque allèle 
    new_pop = np.random.choice(pop, N, p=selection_prob) #chaque allèle est sélectionné avec une proba donnée par selection_prob

    return list(new_pop)



################################################################################
# Simulations                                                                  #
################################################################################

p_init = 0.75 #Fréquence initiale de l'allèle A
N = 200
Gen = 100


fig = plt.figure()
fig.set_facecolor('#d147ad')
ax = fig.add_subplot()

s_values =[0.05, 0.1, 0.3, 0.5, 0.7] #contrainte de sélection 
for s in s_values:
    Fitness = {1:1, 0:1+s} #Fitness de l'allèle A = 1 et de l'allèle B = 1+s
    pop = initialisation_pop(N, p_init)
    FreqsB = [pop.count(0)/N]
    for g in range(Gen):
        pop = new_generation(pop, N, Fitness)
        FreqsB.append(pop.count(0)/N)
    ax.plot(np.arange(Gen+1), FreqsB, label=f"Fitness = {1+s}")
ax.set_title(f"Evolution de la fréquence de l'allèle B au cours de {Gen} générations,\npour une population de {N} individus avec " + rf"$p^B_0={1-p_init}$")
ax.set_ylabel("Fréquence de l'allèle B")
ax.set_xlabel("Générations")
ax.set_ylim(-0.01, 1.03)
plt.grid(True, linestyle="--")
plt.legend()
plt.show()

#Plus s est grand, plus la contrainte de sélection en faveur de l'allèle B est forte, et donc, plus la fréquence de l'allèle B augmente rapiedement