#!/usr/bin/env python
# coding: utf-8

# Koristimo biblioteku networkx za rad sa grafovima. Radimo za sad za graf koji je dat kao primer u pdf-u. Prilagoditi kasnije za testiranje sa drugačijim grafovima.

import networkx as nx
import matplotlib.pyplot as plt
import random

# Funkcija označavanja f : {v1,v2,...,vn} -> {1,2,...,n}

# Definišemo dve pomoćne funkcije. labeling_from_nodes je pomoćna funkcija koja nam od liste čvorova v_j daje listu indeksa j (tj. označavanje tih čvorova).
# nodes_from_labeling je pomoćna funkcija koja nam od liste indeksa j daje listu čvorova v_j.

def labeling_from_nodes(nodes):
    n = len(nodes)
    return [int(nodes[i].strip('v')) for i in range(n)]

def nodes_from_labeling(labeling):
    n = len(labeling)
    return ['v'+str(labeling[i]) for i in range(n)]

# Definišemo funkciju label koja za dati graf G, čvor node i funkciju označavanja f vraća oznaku (labelu) koju tom čvoru dodeljuje funkcija f

def label(G, node, f):
    
    index_of_node = int(node.strip('v'))-1
    return f[index_of_node]


# Definišemo funkciju bandwidth_of_a_node koja za dati graf G, čvor node i funkciju označavanja f izračunava bandwidth tog čvora 

def bandwidth_of_a_node(G, node, f):
    neighbors = [u for u in G.neighbors(node)]
    label_node = label(G,node,f)
    maximum = -1
    for u in neighbors:
        label_u = label(G,u,f)
        diff = abs(label_u - label_node)
        if diff > maximum:
            maximum = diff
    return maximum


# Definišemo funkciju bandwidth koja za dati graf G i funkciju označavanja f izračunava bandwidth grafa 

def bandwidth(G, f):
    nodes = [v for v in G.nodes]
    bandwidths = [bandwidth_of_a_node(G,node,f) for node in nodes]
    return max(bandwidths)


# Definišemo funkciju initial_solution koja za dati graf G generiše početno rešenje - početno označavanje f čvorova grafa G, koristeći BFS sa nasumičnim izborom čvora od kojeg se započinje pretraga.
    
def initial_solution(G):
    nodes = [u for u in G.nodes]
    start = random.choice(nodes)
    edges_iterator = nx.bfs_edges(G,start)
    edges = [edge for edge in edges_iterator]
    solution = [start] + [edge[1] for edge in edges]
    return labeling_from_nodes(solution)


# Definišemo funkciju distance(f,f_p) koja računa rastojanje između rešenja f i f_p na osnovu formule (4) iz pdf-a. Potrebna da bi definisali pojam okoline nekog rešenja.

def distance(f,f_p):
    n = len(f)
    dist = 0
    for i in range(n):
        if f[i] != f_p[i]:
            dist += 1
    dist -= 1
    return dist


# Definišemo funkcije max_labeled_neighbor i min_labeled_neighbor koje za dati graf G, čvor node i labeliranje f računaju, redom, najveću i najmanju labelu među čvorovima susednim čvoru node

def max_labeled_neighbor(G, node, f):
    neighbors = [u for u in G.neighbors(node)]
    max_label = float('-inf')
    for u in neighbors:
        label_u = label(G,u,f)
        if label_u > max_label:
            max_label = label_u
    return max_label

def min_labeled_neighbor(G, node, f):
    neighbors = [u for u in G.neighbors(node)]
    min_label = float('inf')
    for u in neighbors:
        label_u = label(G,u,f)
        if label_u < min_label:
            min_label = label_u
    return min_label


# Definišemo funkciju set_K koja bira podskup čvorova K na osnovu Initialization koraka u Algorithm 1 pdf-a. 

def set_K(G, f, k):
    nodes = [u for u in G.nodes]
    B = bandwidth(G,f)
    B_p = random.randint(1,B)
    
    K = []
    while True:
        for node in nodes:
            bandwidth_node = bandwidth_of_a_node(G, node, f)
            if bandwidth_node >= B_p:
                K.append(node)
                
        if len(K) >= k:
            break
        else:
            K = []
            B_p = random.randint(1,B)
                
    return K


#  Definišemo funkciju critical_node koja za dati graf G, čvor node i labeliranje f nalazi čvor v takav da je  |f(node)-f(v)| = bandwidth_of_a_node(G, node, f)

def critical_node(G, node, f):
    
    nodes = [u for u in G.nodes]
    
    for v in nodes:
        if abs(label(G, node, f) - label(G, v, f)) == bandwidth_of_a_node(G, node, f):
            return v


# Definišemo pomoćnu funkciju swap_labels koja ažurira označavanje f tako što datim čvorovima u i v zameni oznake.

def swap_labels(G, u, v, f):
    
    label_u = label(G, u, f)
    label_v = label(G, v, f)
    
    index_u = f.index(label_u)
    index_v = f.index(label_v)
    
    f[index_u] = label_v
    f[index_v] = label_u
    
    return f


# Definišemo funkciju shaking kojom se realizuje faza razmrdavanja (prema Algorithm 1 sa pdf-a)

def shaking(G, f, k):
    
    K = set_K(G, f, k)
    
    edges = [edge for edge in G.edges]
    nodes = [node for node in G.nodes]
    
    for i in range(k):
        u = random.choice(K)
        v = critical_node(G, u, f) 
        
        if (u,v) in edges:
            f_min_u = min_labeled_neighbor(G, u, f)
            f_max_u = max_labeled_neighbor(G, u, f)

            label_v = label(G, v, f)
            
            min_value = float('inf')
            min_w = None
            
            for w in nodes:
                label_w = label(G, w, f)
                if label_w <= f_max_u and label_w >= f_min_u:
                    current_value = max(max_labeled_neighbor(G, w, f) - label_v,label_v-min_labeled_neighbor(G, w, f))
                    
                    if current_value < min_value:
                        min_value = current_value
                        min_w = w
                        
            f = swap_labels(G,v,min_w,f)
    return f


# Definišemo funkciju best_labeling koja za dati čvor v pronalazi najbolju oznaku.

def best_labeling(G, v, f):
    max_v = max_labeled_neighbor(G, v, f)
    min_v = min_labeled_neighbor(G, v, f)
    
    mid_v = (max_v + min_v) // 2
    
    return mid_v


# Definišemo funkciju suitable_swapping_nodes koja za dati čvor v pronalazi skup čvorova pogodnih za razmenu oznake.

def suitable_swapping_nodes(G, v, f):
    nodes = [u for u in G.nodes]
    mid_v = best_labeling(G, v, f)
    label_v = label(G, v, f)
    N_p = []
    
    for u in nodes:
        label_u = label(G, u, f)
        if abs(mid_v-label_u) < abs(mid_v-label_v):
            N_p.append(u)
            
    return N_p


# Definišemo funkciju number_of_critical_edges koja za dati graf G i labeliranje f izračunava broj kritičnih grana. Kritična grana je ona grana (u,v) takva da je bandwidth_of_a_node(G, u, f) = bandwidth(G, f) i bandwidth_of_a_node(G, v, f) = bandwidth(G, f)


def number_of_critical_edges(G, f):
    
    number = 0
    B = bandwidth(G, f)
    edges = [edge for edge in G.edges]
    
    for edge in edges:
        v_from = edge[0]
        v_to = edge[1]
        
        if bandwidth_of_a_node(G, v_from, f) == B and bandwidth_of_a_node(G, v_to, f) == B:
            number += 1
            
    return number
    
# Definišemo funkciju local_search kojom se realizuje faza lokalne pretrage (prema Algorithm 2 sa pdf-a)

def local_search(G, f):
    
    canImprove = True
    nodes = [u for u in G.nodes]
    
    while canImprove:
        canImprove = False
        
        for v in nodes:
            
            number_of_critical = number_of_critical_edges(G, f)

            if bandwidth_of_a_node(G, v, f) == bandwidth(G, f):
                
                N_p = suitable_swapping_nodes(G,v,f)
                
                for u in N_p:
                    f = swap_labels(G,v,u,f)
                    
                    if number_of_critical_edges(G, f) < number_of_critical:
                        canImprove = True
                        break
                        
                    f = swap_labels(G,v,u,f)
                    
    return f


# Definišemo funkciju number_of_critical_nodes$ koja za dati graf G i označavanje f izračunava broj kritičnih čvorova. Kritični čvor je čvor čiji je bandwidth jednak bandwidth-u grafa

def number_of_critical_nodes(G, f):
    
    Vc = set([])
    B = bandwidth(G, f)
    nodes = [u for u in G.nodes]
    
    for u in nodes:
        if bandwidth_of_a_node(G, u, f) == B:
            Vc.add(u)
            
    return len(Vc)
  

# Definišemo funkciju move kojom se realizuje mehanizam 'move or not' (prema Algorithm 3 sa pdf-a)

def move(G, f, f_p, alpha):
    
    Move = False
    B_f = bandwidth(G, f)
    B_f_p = bandwidth(G, f_p)
    
    if B_f_p < B_f:
        Move = True
        
    else:
        if B_f == B_f_p:
            if number_of_critical_nodes(G,f_p) < number_of_critical_nodes(G,f) or distance(f, f_p) > alpha:
                Move = True
                
    return Move


# Definišemo funkciju VNS kojom se realizuje metod promenljivih okolina (prema Algorithm 4 sa pdf-a)

def VNS(G, k_min, k_max, k_step, alpha):
    B_star = float('inf')
    t = 0
    
    i_max = int((k_max - k_min)/k_step)
    
    f = initial_solution(G)
    f = local_search(G, f)
    i = 0
    k = k_min
    
    while i <= i_max:
        f_p = shaking(G, f, k)
        f_p = local_search(G, f)
        if move(G, f, f_p, alpha):
            f = f_p
            k = k_min
            i = 0
        else:
            k = k + k_step
            i = i + 1
    
    return f


# Pomoćna funkcija generate_graph koja na osnovu broja čvorova n i broja grana e formira random graf.

def generate_graph(n, e):
    
    l = [i for i in range(1,n+1)]
    V = nodes_from_labeling(l)
    G = nx.Graph()

    for i in range(e):
        u = random.choice(V)
        v = random.choice(V)
    
        while u == v:
            v = random.choice(V)
        
        G.add_edge(u,v)
    
    return G


# Pomoćna funkcija visualize_graph koja vizualizuje prosleđeni graf.

def visualize_graph(G):
    
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8,8))
    nx.draw_networkx_nodes(G,pos,node_color='pink')
    nx.draw_networkx_labels(G,pos,font_color='magenta')
    nx.draw_networkx_edges(G,pos,edgelist=list(G.edges),color='purple')
    plt.axis('off')
    plt.title('I am a pretty little pink graph :)', color = 'purple', size = 20)
    plt.show()


## Testiranje ##




