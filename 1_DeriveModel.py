import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


################################################################################
# Fonctions                                                                    #
################################################################################

def initialisation_pop(N:int, p_init:float) -> list[int]:
    """ Initialisation d'une population de taille N avec une fréquence initiale de l'allèle A
        input :  N = taille de la population
                 p_init = fréquence initiale de l'allèle A
        output : pop = population initiale, l'allèle A est représenté par 1 et l'allèle B par 0"""
    
    num_ones = round(N*p_init) 
    num_zeros = N - num_ones
    pop = [1] * num_ones + [0] * num_zeros 
    return pop


def new_generation(pop:list[int], N:int) -> list[int]:
    """ Génère une nouvelle génération à partir de la génération actuelle
        input :  pop = génération actuelle
                 N = taille de la population
        output : new_gen = nouvelle génération"""
    
    p_current = np.sum(pop) / N # fréq allèle A dans la population précédente
    new_num_ones = np.random.binomial(N, p_current) #génère un nombre aléatoire de succès (1) selon une loi binomiale 
    new_num_zeros = N - new_num_ones
    new_gen = [1] * new_num_ones + [0] * new_num_zeros
    return new_gen


def simulate_freqs_pop(N:int, p_init:float, Gen:int) -> list[float]:
    """Simule une population et retourne une liste des fréquences de l'allèle A par génération
        input :  N = taille de la population
                 p_init = fréquence initiale de l'allèle A dans la population
                 Gen = Nombre de génération à simuler
                 plot = Tracer le graphe de suivi de la fréquence de l'allèle A par génération
        output : freqs_p = liste des fréquences de l'allèle A au cours des générations"""
    
    pop = initialisation_pop(N, p_init)
    freqs_p = [sum(pop)/N] #première fréquence
    for _ in range(Gen):
        pop = new_generation(pop, N)
        p = sum(pop)/N
        freqs_p.append(p)
    return freqs_p


def simulate_genetic_drift(N:int, p_init:float, Gen:int, P:int, plot:bool=False) -> tuple[float, float]:
    """Simule une population et retourne une liste des fréquences de l'allèle A par génération
        input :  N = taille de la population
                 p_init = fréquence initiale de l'allèle A dans la population
                 Gen = Nombre de génération à simuler
                 P = Nombres de populations à simuler
                 plot = Tracer le graphe de suivi de la fréquence de l'allèle A par génération pour toutes les populations simulées
        output : Estimations de la probabilité de fixation et du temps de fixation"""
    
    Allfreqs = [] #liste de liste des fréquences de l'allèle A pour chaque population simulée
    fixed = 0
    T_fixed = []
    for _ in range(P): #population
        freqs = simulate_freqs_pop(N, p_init, Gen)
        Allfreqs.append(freqs)
        if freqs[-1] == 1:  #moment où l'allèle A est fixé
            fixed += 1
            T_fixed.append(freqs.index(1))
            #index prend l'indice de la première occurence de 1 ou 0 dans la liste freqs     

    if plot:
        T = np.arange(Gen+1)
        fig = plt.figure()
        fig.set_facecolor('#ffebab')
        ax = fig.add_subplot()
        for p in range(P):
            ax.plot(T, Allfreqs[p], label=f"Population {p+1}")
        ax.set_title(f"Simulation de la dérive génétique pour {P} populations, {Gen} générations,\nN = {N}, avec une fréquence allélique initiale de " + rf"$p_0={p_init}$")
        ax.set_xlabel("Générations")
        ax.set_ylabel("Fréquence de l'allèle p")
        plt.grid(True, linestyle="--")
        plt.legend()
        plt.show()
    if len(T_fixed) == 0: #si aucun allèle n'a été fixé
        return 0, 0
    return fixed/P, np.mean(T_fixed) 


def analytical_time_fixation(N:int, p_init:float) -> float:
    """Retourne le temps théorique de fixation de l'allèle A
        input :  N = taille de la population
                 p_init = fréquence initiale de l'allèle A
        output : T = temps de fixation théorique"""
    
    return -(2*N/p_init)*(1-p_init)*np.log(1-p_init)



################################################################################
# Simulations                                                                  #
################################################################################

p_init = 0.5 #fréquence initiale de l'allèle A
N = 200 #Nombre d'individus de 200 
Gen = 4000 #Nombre de génération à simuler par population
P = 100 #nombre de populations à simuler (afin d'estimer la probabilité/temps de fixation)
# P = 10 #test



# a) Tracer la fréquence de l'allèle A au cours des générations sur 10 populations
simulate_genetic_drift(N, p_init, Gen, 10, plot=True)
#Temps d'execution : quelques secondes


# b) Tracer la probabilité de fixation en fonction de la fréquence initiale p_init
p_inits = np.linspace(0, 1, 101)


fix_prob = []
for p_init in tqdm(p_inits):
    fix_prob.append(simulate_genetic_drift(N, p_init, Gen, P)[0])
fig = plt.figure()
fig.set_facecolor('#ffebab')
ax = fig.add_subplot()
ax.plot(p_inits, p_inits, color="k", label="Probabilité de fixation théorique")
ax.plot(p_inits, fix_prob, color="r", label="Probabilité de fixation empirique")
plt.legend()
plt.title(f"Probabilité de fixation en fonction de la fréquence allélique initiale,\npour N = {N}, {Gen} générations, {P} simulations par valeur de " + rf"$p_0$")
plt.xlabel("Fréquence allélique initiale")
plt.ylabel("Probabilité de fixation")
plt.grid(True, linestyle="--")
plt.show()
#Temps d'execution : 9min (P=100) ; 1min (P=10)


# c) Tracer le temps de fixation en fonction de la taille de la population
p_init = 0.5 #fréquence initiale de l'allèle A
Nlist = np.arange(50, 2_001, 50)
print(Nlist)

TimeFixTheor = analytical_time_fixation(Nlist, p_init)

TimeFixation = []
for N in tqdm(Nlist):
    TimeFixation.append(simulate_genetic_drift(N, p_init, Gen, P)[1])
fig = plt.figure()
fig.set_facecolor('#ffebab')
ax = fig.add_subplot()
ax.plot(Nlist, TimeFixation, color='r', label="Temps empirique de fixation")
ax.plot(Nlist, TimeFixTheor, color='k', linestyle='--', label="Temps théorique de fixation")
ax.set_title(f"Temps de fixation de l'allèle A en fonction de la taille de la population,\n{Gen} générations, {P} simulations par valeur de N, pour " + rf"$p_0={p_init}$")
ax.set_xlabel("Taille de la population")
ax.set_ylabel("Temps de fixation (en générations)")
plt.grid(True, linestyle="--")
plt.legend()
plt.show()
#Temps d'execution : 50min (P=100) ; 1min (P=10)



