import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


################################################################################
# Fonctions                                                                    #
################################################################################

def initialisation_pop(N:int) -> list[int]:
    """Initialise une population de taille N d'un seul allèle
        input :  N = taille de la population
        output : liste remplie de 0"""
    return [0] * N #un seul allèle au départ


def new_generation(pop:list[int], N:int, mu:float, nb_allele:int) ->tuple[list[int], int]:
    """Génère une nouvelle génération (liste) à partir de la précédente
        input :  pop = population actuelle (liste d'entiers, chaque entier représente un allèle)
                 N = taille de la population
                 mu = taux de mutation / génération / individu
                 nb_allele = nombre d'allèles accumulés par la population
        output : new_pop = nouvelle génération (liste d'entiers)
                 nb_allele = nombre d'allèles accumulés par la population"""
    
    new_pop = []
    for i in range(N):
        ind = np.random.randint(N) #on choisit un parent
        new_pop.append(pop[ind])
        if np.random.rand() < mu: #avec une probabilité mu
            nb_allele += 1  #on a un nouvel allèle unique 
            new_pop[i] = nb_allele 
    return new_pop, nb_allele


def simulate_mutations_with_drift_and_mutation(N:int, Gen:int, mu:float, plot:bool=False) -> tuple[list[list[int]], int]:
    """Simule une population sous l'effet de la dérive génétique avec un taux de mutation
        input :  N = taille de la population
                 Gen = nombre de génération à simuler
                 mu = taux de mutation
                 plot = Tracer le graphe de suivi des fréquences des différents allèles
        output : AllPops = liste contenant toutes les générations simulées
                 nb_allele = nombre total d'allèle apparu au sein de la population"""
    
    pop = initialisation_pop(N)
    AllPops = [pop]
    nb_allele = 0
    for g in range(Gen):
        pop, nb_allele = new_generation(pop, N, mu, nb_allele)
        AllPops.append(pop)

    if plot:
        FreqAlleles = [[] for _ in range(nb_allele+1)]
        for g in range(Gen):
            for allele in range(nb_allele+1):
                FreqAlleles[allele].append(np.sum(np.array(AllPops[g])==allele)/N)

        fig = plt.figure()
        fig.set_facecolor('#b392e8')
        ax = fig.add_subplot()
        for allele, freq in enumerate(FreqAlleles):
            ax.plot(np.arange(Gen), freq, label=f"Allèle n°{allele}")
        ax.set_title(f"Simulation de l'évolution des fréquences alléliques d'une population\nN = {N}, {Gen} générations, avec " + rf"$\mu={mu}$")
        ax.set_xlabel("Générations")
        ax.set_ylabel("Fréquences alléliques")
        plt.grid(True, linestyle="--")
        # plt.legend()
        plt.show()
    return AllPops, nb_allele


def getFixationIndex(Pops:list[list[int]]) -> list[float]:
    """Calcule l'indice de fixation empirique d'une population (= probabilité d'avoir 2 allèle identique)
        input :  Pops = liste contenant toutes les générations simulées
        output : liste des indices de fixation par génération"""
    
    FixIndex = []
    for gen in Pops: #pour chaque génération
        ind = 0
        n = 0
        for i in range(len(gen)):
            for j in range(i):
                n += 1
                if gen[i] == gen[j]: #si les deux allèles sont identitiques
                    ind += 1
        FixIndex.append(ind/n)
    return FixIndex
        


################################################################################
# Simulations                                                                  #
################################################################################

N = 300
Gen = 800
mu = 0.001
P = 100
P = 10



Pops, nb_mutations = simulate_mutations_with_drift_and_mutation(N, Gen, mu, plot=True) #Temps d'execution : quelques secondes


AllFixIndex = []
total_mut = 0
for _ in tqdm(range(P)):
    Pops, mut = simulate_mutations_with_drift_and_mutation(N, Gen, mu)
    total_mut += mut-1 #on enlève l'allèle initial
    FixIndex = getFixationIndex(Pops)
    AllFixIndex.append(FixIndex)

AllFixIndex = np.mean(AllFixIndex, axis=0)

print(f"Nombre moyen de mutations: {total_mut/P}")
print(f"Nombre théorique de mutations: {N*Gen*mu}")

fig = plt.figure()
fig.set_facecolor('#b392e8')
ax = fig.add_subplot()
ax.plot(np.arange(Gen+1), AllFixIndex)
ax.set_title(f"Moyenne des indices de fixation au cours des générations, N = {N},\n{Gen} générations" +rf"et $\mu={mu}$ pour {P} populations")
ax.set_xlabel("Générations")
ax.axhline(y= 1 / (2*N*mu + 1), color='k', linestyle='--', label=rf"Equilibre : $\frac{{1}}{{2*N*\mu+1}}$")
ax.set_ylabel("Indice de fixation")
ax.set_ylim(0, 1.01)
plt.grid(True, linestyle="--")
plt.legend()
plt.show()

#Temps d'execution : 16min (P = 100) ; 2min (P = 10)
