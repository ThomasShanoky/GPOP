import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


################################################################################
# Fonctions                                                                    #
################################################################################

def initialisation_pop(N:int) -> list:
    """Initialise une population de N lignées
        input :  N = taille de la population
        output : liste de N lignées"""
    return np.arange(N) #individus uniques = lignées


def new_generation(pop:list, N:int) -> list[int]:
    """Génère une nouvelle génération à partir de la génération actuelle
        input :  pop = liste des individus (lignées) de la génération actuelle
                 N = taille de la population
        output : new_pop = nouvelle génération (nouvelle liste des lignées)"""
    
    new_pop = []
    for _ in range(N):
        ind = np.random.randint(N) #on choisit un parent aléatoire parmi les N individus
        new_pop.append(pop[ind]) 

    return new_pop


def number_ancestor(pop:list[int]) -> int:
    """Compte le nombre d'ancêtre commun unique de la population
        input :  pop = liste des individus (lignées) de la génération actuelle
        output : nombre d'ancêtres uniques dans la population"""
    return len(set(pop))


def simulate_coalescence(N:int) -> int:
    """Simule une population jusqu'à la coalescence de toutes les lignées en une
        input :  N = taille de la population
        output : G = génération à laquelle on n'a plus qu'une seule lignée"""
    
    pop = initialisation_pop(N)
    G = 0
    while number_ancestor(pop) > 1: #tant qu'on a pas qu'une seule lignée, on a pas tous les individus identiques par descendance
        pop = new_generation(pop, N)
        G += 1
    return G


def get_coalescent_time(N:int, n:int) -> int:
    """Retourne le temps (en génération) pour avoir un événement de coalescence dans un échantillon de taille n dans une population de taille N
        input :  N = taille de la population
                 n = tailel de l'échantillon
        output : G = Temps en génération qu'il faut pour avoir un événement de coalescence"""
    
    G = 0
    n_sample = np.arange(n)
    while len(np.unique(n_sample)) == n:
        G += 1
        for ind in range(n):
            n_sample[ind] = np.random.randint(N) #on choisi un parent pour chaque individu de l'échantillon parmi N individus

    return G


################################################################################
# Simulations                                                                  #
################################################################################

N = 100
P = 1000
P = 100
nList = [2, 3, 4, 5]


CoalescentComplete = 0
for _ in tqdm(range(P)):
    CoalescentComplete += simulate_coalescence(N)
print(f"Temps théorique pour avoir tous les allèles identiques par la descendance (= temps pour remonter jusqu'au MRCA): {2*N} générations")
print(f"Temps empirique pour avoir tous les allèles identiques par la descendance: {CoalescentComplete/P} générations")
print("\n")


print(f"Temps théorique pour avoir un événement de coalescence avec n = 2: {N}")
print(f"Temps théorique pour avoir un événement de coalescence avec n = 3: {N/3}")
print(f"Temps théorique pour avoir un événement de coalescence avec n = 4: {N/6}")
print(f"Temps théorique pour avoir un événement de coalescence avec n = 5: {N/10}")

for n in nList:
    CoalescentTime = 0
    for _ in range(P):
        CoalescentTime += get_coalescent_time(N, n)
    print(f"Temps empirique pour avoir un événement de coalescence avec n = {n} : {CoalescentTime/P}")

#Temps d'execution : 3min (P = 1000) ; 20s (P = 100)