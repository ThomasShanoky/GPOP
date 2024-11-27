import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
    np.random.shuffle(new_gen)

    return new_gen


def simulate_freqs_pop(N:int, p_init:float, Gen:int, m:float, plot:bool=False) -> list[list[int]]:
    """Simule l'évolution des fréquences alléliques dans une population qui se divise en 10 sous-populations
        input :  N = taille de la population initiale
                 p_init = fréquence initiale de l'allèle A
                 Gen = nombre de générations
                 m = taux de migration
                 plot = booléen pour afficher le graphique
        output : sub_pops = liste des sous-population de la dernière génération
                 freqs = liste des fréquences de l'allèle A dans les sous-populations à chaque génération (Gen x 10)"""
    
    sub_pops = initialisation_pops(N, p_init) #liste des sous pop
    freqs = [[sum(sub_pop)/int(N/10) for sub_pop in sub_pops]] #fréquences de A pour chaque sous population
    num_migrants = int(m*N)
    for g in tqdm(range(Gen)):
        for s in range(10):
            sub_pops[s] = new_generation(sub_pops[s], int(N/10))

        for _ in range(num_migrants):
            s1, s2 = np.random.choice(10, 2, replace=False)
            i1, i2 = np.random.randint(10), np.random.randint(10)
            sub_pops[s1][i1], sub_pops[s2][i2] = sub_pops[s2][i2], sub_pops[s1][i1]

        freqs.append([sum(sub_pop)/int(N/10) for sub_pop in sub_pops])

    if plot:
        fig = plt.figure()
        fig.set_facecolor('#e37043')
        ax = fig.add_subplot()
        freqs = np.array(freqs)
        for s in range(10):
            FreqsSubPop = freqs[:, s]
            ax.plot(np.arange(Gen+1), FreqsSubPop, label=f"pop n°{s+1}")
        ax.set_title("Simulation d'une population s'étant divisée en 10, on suit les fréquences\nde l'allèle A dans les différentes sous-populations avec " + rf"$p_0={p_init}$ et $m={m}$")
        ax.set_ylabel("Fréquences")
        ax.set_ylim(-0.01, 1.01)
        ax.set_xlabel("Générations")
        plt.grid(True, linestyle="--")
        plt.legend()
        plt.show()

    return sub_pops, freqs


def get_coalescent_time(N:int, p_init:float, Gen:int, m:float) -> int:
    """Simule l'évolution d'une population qui se divise en 10 sous-populations et retourne le temps de coalescence
        input :  N = taille de la population initiale
                 p_init = fréquence initiale de l'allèle A
                 Gen = nombre de générations
                 m = taux de migration
        output : Temps de fixation de l'allèle A (s'il n'y a pas eu fixation, retourne Gen)"""
    
    sub_pops = initialisation_pops(N, p_init) 
    num_migrants = int(m*N)
    for g in range(Gen):
        for s in range(10):
            sub_pops[s] = new_generation(sub_pops[s], int(N/10))

        for m in range(num_migrants):
            s1, s2 = np.random.choice(10, 2, replace=False)
            i1, i2 = np.random.randint(10), np.random.randint(10)
            sub_pops[s1][i1], sub_pops[s2][i2] = sub_pops[s2][i2], sub_pops[s1][i1]

        total_sum = np.sum(sub_pops)
        if total_sum == N:
            return g
        
        if total_sum == 0:
            return Gen #il n'y a pas eu fixation
        
    return Gen


def analytical_time_fixation(N:int, p_init:float) -> float:
    """Retourne le temps théorique de fixation de l'allèle A
        input :  N = taille de la population
                 p_init = fréquence initiale de l'allèle A
        output : T = temps de fixation théorique"""
    
    return -(2*N/p_init)*(1-p_init)*np.log(1-p_init)



################################################################################
# Simulations                                                                  #
################################################################################

p_init = 0.5 #fréquence initiale de l'allèle A (dans la population de base)
N = 2_000 #taille de la population initiale
Gen = 4_000
m = 0.1
P = 100
# P = 1 #test


print(f"A chaque génération, {m*(N/10)} individus par sous-population sont échangés entre les sous-populations")
print(f"Au total, {m*N} individus sont échangés à chaque génération")
simulate_freqs_pop(N, p_init, Gen, m, True)
print("\n") #Temps d'exécution : 30s


Nlist = np.arange(100, 2_001, 100)
print(Nlist)

TimeFixation = []
for N in tqdm(Nlist):
    coalTimeN = []
    for _ in tqdm(range(P)):
        Time = get_coalescent_time(N, p_init, Gen, m)
        if Time != Gen: #s'il y a eu fixation
            coalTimeN.append(Time)

    if len(coalTimeN) == 0: #s'il n'y a pas eu fixation
        TimeFixation.append(0)
    else: #s'il y a eu fixation, on fait la moyenne des temps de fixation
        TimeFixation.append(np.mean(coalTimeN))

TimeFixTheor = analytical_time_fixation(Nlist, p_init)

fig = plt.figure()
fig.set_facecolor('#e37043')
ax = fig.add_subplot()
ax.plot(Nlist, TimeFixTheor, color='k', linestyle='--', label="Temps théorique de fixation (comparaison)")
ax.plot(Nlist, TimeFixation, color="r", label="Temps de fixation empirique")
ax.set_title("Temps de fixation de l'allèle A en fonction de la taille (avant division)\nde la population avec " + rf"$p_0={p_init}$ et $m={m}$")
ax.set_xlabel("Taille de la population")
ax.set_ylabel("Temps de fixation (en générations)")
plt.grid(True, linestyle="--")
plt.show()

#Temps d'execution : 2h50 (P = 50) ; 3min (P = 1)