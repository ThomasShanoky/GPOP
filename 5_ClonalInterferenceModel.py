import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


################################################################################
# Fonctions                                                                    #
################################################################################

def initialisation_pop(N:int, p_init:float) -> list[int]:
    """Initialise une population de taille N avec une fréquence initiale de l'allèle A
        input :  N = taille de la population
                 p_init = fréquence initiale de l'allèle A et de l'allèle B
        output : liste remplie de 0 (allèle A) de 1 (B) et de 2 (C)"""
    num_zeros = round(N*p_init[0])
    num_ones = round(N*p_init[1])
    num_twos = N - num_zeros - num_ones
    pop = [0]*num_zeros + [1]*num_ones + [2]*num_twos #0 = allèle A, 1 = allèle B, 2 = allèle C

    return pop


def new_generation(pop:list[int], N:int, Fitness:dict[int:float]) -> list[int]:
    """Génère une nouvelle génération (liste) à partir de la précédente
        input :  pop = population actuelle (liste d'entiers, chaque entier représente un allèle)
                 N = taille de la population
                 Fitness = dictionnaire contenant les fitness de chaque allèle
        output : new_pop = nouvelle génération"""
    
    fitness_values = np.array([Fitness[allele] for allele in pop])
    selection_prob = fitness_values / np.sum(fitness_values)
    new_pop = np.random.choice(pop, N, p=selection_prob)

    return list(new_pop)


def simulate_freqs_pop(N:int, p_init:float, Gen:int, Fitness:dict[int:float], plot:bool=False) -> tuple[list[float], list[float], list[float]]:
    """Génère 3 listes de fréquences des allèles A, B et C pour une population de taille N sur Gen générations
        input :  N = taille de la population
                 p_init = fréquence initiale de l'allèle A et de l'allèle B
                 Gen = nombre de générations à simuler
                 Fitness = dictionnaire contenant les fitness de chaque allèle
                 plot = tracer les graphiques des fréquences des allèles
        output : freqsA = liste des fréquences de l'allèle A
                 freqsB = liste des fréquences de l'allèle B
                 freqsC = liste des fréquences de l'allèle C"""

    pop = initialisation_pop(N, p_init)

    freqsA = [pop.count(0)/N]
    freqsB = [pop.count(1)/N]
    freqsC = [pop.count(2)/N]

    for _ in range(Gen):
        pop = new_generation(pop, N, Fitness)
        freqsA.append(pop.count(0)/N)
        freqsB.append(pop.count(1)/N)
        freqsC.append(pop.count(2)/N)

    if plot:
        fig = plt.figure()
        fig.set_facecolor('#f16284')
        ax = fig.add_subplot()
        ax.plot(np.arange((Gen+1)), freqsA, 'g', label=rf"Fréquence de A $(Fitness=1,00)$")
        ax.plot(np.arange((Gen+1)), freqsB, 'b', label=rf"Fréquence de B $(Fitness=1,05)$")
        ax.plot(np.arange((Gen+1)), freqsC, 'r', label=rf"Fréquence de C $(Fitness=1,10)$")
        ax.set_title(f"Simulation d'une population de taille {N} sur {Gen} générations")
        ax.set_ylim(-0.01, 1.01)
        ax.set_ylabel("Fréquences")
        ax.set_xlabel("Générations")
        plt.grid(True, linestyle="--")
        plt.legend()
        plt.show()

    return freqsA, freqsB, freqsC


################################################################################
# Simulations                                                                  #
################################################################################

Gen = 500
P = 100
p_initA, p_initB = 0.79, 0.20
p_init = [p_initA, p_initB]
Fitness = {0:1, 1:1.05, 2:1.1}


# Simulation d'une population pour N = 100, N = 500 et N = 1000, on regarde la fréquence des allèles A, B et C
simulate_freqs_pop(100, p_init, Gen, Fitness, True)
simulate_freqs_pop(500, p_init, Gen, Fitness, True)
simulate_freqs_pop(1000, p_init, Gen, Fitness, True)


# Simulation de P populations afin d'avoir une estimation précise des fréquences par valeur de N

Nlist = np.arange(100, 1001, 100)  # tailles de population à simuler

fig, axs = plt.subplots(1, 3, figsize=(18, 6))  #créer une figure avec 3 sous-graphiques
fig.set_facecolor('#f16284')

for N in tqdm(Nlist):
    AllfreqsA, AllfreqsB, AllfreqsC = [], [], [] #contiendra les fréquences de chaque allèle pour chaque population simulée
    for p in range(P):
        Fa, Fb, Fc = simulate_freqs_pop(N, p_init, Gen, Fitness)
        AllfreqsA.append(Fa)
        AllfreqsB.append(Fb)
        AllfreqsC.append(Fc)
    AllfreqsA, AllfreqsB, AllfreqsC = np.array(AllfreqsA), np.array(AllfreqsB), np.array(AllfreqsC)
    
    FreqsA_moy = [np.mean(AllfreqsA[:, g]) for g in range(Gen + 1)] #fréquence moyenne par génération pour chaque allèle
    axs[0].plot(np.arange(Gen + 1), FreqsA_moy, label=f"f(A) (N={N})")
    FreqsB_moy = [np.mean(AllfreqsB[:, g]) for g in range(Gen + 1)]
    axs[1].plot(np.arange(Gen + 1), FreqsB_moy, label=f"f(B) (N={N})")
    FreqsC_moy = [np.mean(AllfreqsC[:, g]) for g in range(Gen + 1)]
    axs[2].plot(np.arange(Gen + 1), FreqsC_moy, label=f"f(C) (N={N})")
    
#sous-graphes
axs[0].set_title("Fréquence de A")
axs[0].set_ylabel("Fréquences")
axs[0].set_ylim(0, 1.01)
axs[0].grid(True, linestyle="--")
axs[0].legend()

axs[1].set_title("Fréquence de B")
axs[1].set_xlabel("Générations")
axs[1].set_ylim(0, 1.01)
axs[1].grid(True, linestyle="--")
axs[1].legend()

axs[2].set_title("Fréquence de C")
axs[2].set_ylim(0, 1.01)
axs[2].grid(True, linestyle="--")
axs[2].legend()

plt.suptitle("Evolution des fréquences alléliques moyennes pour différentes tailles de population")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()