import matplotlib.pyplot as plt
import numpy as np


################################################################################
# Fonctions                                                                    #
################################################################################

def initialisation_pops(N:int, p_init:float) -> list[list[int]]:
    """Initialise une population de taille N, avec une fréquence allélique p_init de l'allèle A
        input :  N = taille de la population
                 p_init = fréquence initiale de l'allèle A
        output : ensemble des 10 sous-populations"""
    
    num_ones = round(N*p_init)
    num_zeros = N - num_ones
    pop = [1] * num_ones + [0] * num_zeros #1 = allèle A et 0 = allèle B
    np.random.shuffle(pop)

    sub_pops = []
    for k in range(10): #génération de 10 sous-populations
        sub_pops.append(pop[k*int(N/10):(k+1)*int(N/10)]) #10 sous populations

    return sub_pops


def new_generation(sous_pop:list[int], N_sub:int) -> list[int]:
    """A partir d'une sous-population, génère une nouvelle génération
        input :  sous_pop = sous-population actuelle
                 N_sub = taille de la sous-population
        output : new_gen = nouvelle génération"""
    
    p_current = np.sum(sous_pop) / N_sub # fréq allèle A
    new_num_ones = np.random.binomial(N_sub, p_current)
    new_num_zeros = N_sub - new_num_ones
    new_gen = [1] * new_num_ones + [0] * new_num_zeros
    # np.random.shuffle(new_gen)

    return new_gen


def simulate_freqs_pop(N:int, p_init:float, Gen:int, plot:bool=False) -> list[list[int]]:
    """Simule l'évolution des fréquences alléliques dans une population qui se divise en 10 sous-populations
        input :  N = taille de la population initiale
                 p_init = fréquence initiale de l'allèle A
                 Gen = nombre de générations
                 plot = booléen pour afficher le graphique
        output : freqs = liste des fréquences de l'allèle A dans les sous-populations à chaque génération (Gen x 10)"""
    
    sub_pops = initialisation_pops(N, p_init) #liste des sous pop
    freqs = [[sub_pop.count(1)/int(N/10) for sub_pop in sub_pops]] #fréquences de A pour chaque sous population
    for g in range(Gen):
        for s in range(10):
            sub_pops[s] = new_generation(sub_pops[s], int(N/10))
        freqs.append([sub_pop.count(1)/int(N/10) for sub_pop in sub_pops])

    if plot:
        fig = plt.figure()
        fig.set_facecolor('#b69264')
        ax = fig.add_subplot()
        freqs = np.array(freqs)
        for s in range(10):
            FreqsSubPop = freqs[:, s]
            ax.plot(np.arange(Gen+1), FreqsSubPop, label=f"pop n°{s+1}")
        ax.set_title("Simulation d'une population s'étant divisée en 10, on suit les fréquences\nde l'allèle A dans les différentes sous-populations avec " + rf"$p_0={p_init}$")
        ax.set_ylabel("Fréquences")
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel("Générations")
        plt.grid(True, linestyle="--")
        plt.legend()
        plt.show()

    return freqs


################################################################################
# Simulations                                                                  #
################################################################################

p_init = 0.5 #fréquence initiale de l'allèle A (dans la population de base)
N = 2_000 #taille de la population initiale (avant division)
Gen = 1_000


FreqsA = simulate_freqs_pop(N, p_init, Gen, plot=True)
print(f"Nombres de sous-populations où l'allèle A a été fixé : {sum(FreqsA[-1])}")

# Les evolutions des frequences alléliques dans les sous-populations sont indépendantes les unes des autres



