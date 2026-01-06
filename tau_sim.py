# Authors: Toffoli Marte and Leonardo Miculan, University of Trento, Quantitative and Computational Biology
# Exam: Mathematical Modeling and Simulation, A.Y. 2025-2026
# Date: January 2026
# This code implements the tau-leaping method for simulating epidemic spread on networks. Main part of the script is from the original 
# SSA-based epidemic simulation code provided by https://github.com/emmerichmtm/EfficientStochasticSimulationOfEpidemics
# Simulate_tau_step method is the core of the implementation and our main focus.


from cmath import tau
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import time
start_time = time.time()



class EpidemicGraph:
    def __init__(self, infection_rate=0.1, recovery_rate=0, model=2):  
        #model: 0: SI, 1: SIS, 2: SIR
        self.G = nx.Graph()
        self.model = model 
        self.infection_rate, self.recovery_rate = infection_rate, recovery_rate
        self.infected_nodes = []
        self.total_infection_rate, self.total_recovery_rate = 0, 0

    def add_node(self, node_id):
        self.G.add_node(node_id, infected=False,
                        recovered=False, sum_of_weights_i=0.0)

    def add_edge(self, node1, node2, weight):
        self.G.add_edge(node1, node2, weight=weight)

    def simulate_tau_step(self, tau=0.008):
        # tau substitutes wait_time (in original script)
        if not self.infected_nodes:
            return tau

        # Compute "propensities"
        a_inf = self.total_infection_rate # total infection rate
        if (self.model != 0):
            a_rec = self.total_recovery_rate # total recovery rate
        else:
            a_rec = 0.0

        if (a_inf + a_rec < 0.0001): 
            return 0

        # Sample number of events (kj in pseudocode)
        num_inf = np.random.poisson(a_inf * tau)
        num_rec = np.random.poisson(a_rec * tau)

        # Infection events
        for _ in range(num_inf):

            # choose infected source proportional to sum_of_weights_i
            target = random.uniform(0, self.total_infection_rate)
            cumulative = 0.0

            for node in self.infected_nodes:
                cumulative += self.G.nodes[node]['sum_of_weights_i']
                if (cumulative > target):
                    self.infect_neighbor(node)
                    break
        # Recovery events
        if (self.model != 0):
            for _ in range(num_rec):
                # if no infected nodes, skip
                if (not self.infected_nodes):
                    break

                # choose infected node uniformly (same as SSA)
                node = random.choice(self.infected_nodes)
                self.recover_node(node)

        return tau 


    def recover_node(self, node):
        self.infected_nodes.remove(node)
        self.total_recovery_rate -= self.recovery_rate
        if self.model == 1:  # SIS
            for neighbor in self.G.neighbors(node):
                self.G.nodes[neighbor]['sum_of_weights_i'] += self.G[node][neighbor]['weight']
            self.total_infection_rate += self.G.nodes[node]['sum_of_weights_i']
            self.G.nodes[node]['infected'] = False
        elif self.model == 2:  # SIR
            self.G.nodes[node]['recovered'], self.G.nodes[node]['infected'] = True, False

    def infect_neighbor(self, node):
        neighbors = [n for n in self.G.neighbors(node) if
                     
                     not self.G.nodes[n]['infected'] and not self.G.nodes[n]['recovered']]
        if neighbors:
            weights = np.array([self.G[node][n]['weight'] for n in neighbors])
            cumulative, target = 0, random.uniform(0, np.sum(weights))
            for i, weight in enumerate(weights):
                cumulative += weight
                if cumulative > target:
                    self.infect_node(neighbors[i])
                    break

    def infect_node(self, node):
        self.G.nodes[node]['infected'] = True
        self.infected_nodes.append(node)
        self.total_recovery_rate += self.recovery_rate
        for neighbor in self.G.neighbors(node):   # the infection rate becomes now relevant
            if not self.G.nodes[neighbor]['infected']:
                self.G.nodes[node]['sum_of_weights_i'] += self.G[node][neighbor]['weight']
                self.total_infection_rate += self.G[node][neighbor]['weight']
            elif self.G.nodes[neighbor]['infected'] and (neighbor !=node):
                # reduce the rate for the infected neighbor, as it has one less infected neighbor    
                w = self.G[node][neighbor]['weight']
                self.G.nodes[neighbor]['sum_of_weights_i'] = self.G.nodes[neighbor]['sum_of_weights_i']-w
                self.total_infection_rate -= w

    def plot_graph(self, title="Graph", scale=300):
        # layout & colors as before
        pos    = nx.spring_layout(self.G)
        colors = ['red' if self.G.nodes[n]['infected'] else 'green'
                  for n in self.G.nodes()]

        # compute sizes proportional to (sum_of_weights + 1)
        sizes = [
            (self.G.nodes[n].get('sum_of_weights_i', 0) + 1) * scale
            for n in self.G.nodes()
            ]

        # draw with variable node_size
        nx.draw(self.G, pos,
                node_color=colors,
                with_labels=True,
                node_size=sizes)

        nx.draw_networkx_edges(self.G, pos, width=1.0, alpha=0.5)
        
        # (optional) show total infection rate in title
        plt.title(f"{title} — Total infection rate: {self.total_infection_rate:.2f}")
        plt.show()


# Test large networks
def test_large_network(model="barabasi_albert"):
    num_nodes = 10000
    if model == "barabasi_albert":
        graph_instance = nx.barabasi_albert_graph(num_nodes, 2)
    elif model == "erdos_renyi":
        graph_instance = nx.erdos_renyi_graph(num_nodes, 0.01)
    elif model == "watts_strogatz":
        graph_instance = nx.watts_strogatz_graph(num_nodes, 4, 0.1)
    else:
        raise ValueError("Invalid model.")

    graph = EpidemicGraph(0.2, 0.8, model=2)
    for node in graph_instance.nodes:
        graph.add_node(node)
    for edge in graph_instance.edges:
        graph.add_edge(edge[0], edge[1], 1.0)
    graph.infect_node(0)

    infections_over_time, simulated_time, total_time = [], [], 0.0
    for _ in range(1000): # number of events/steps we want to observe (it define the simulation time)
        infections_over_time.append(
            len([n for n in graph.G.nodes if graph.G.nodes[n]['infected']]))
        wait_time = graph.simulate_tau_step()
        total_time += wait_time
        simulated_time.append(total_time)

    plt.scatter(simulated_time, infections_over_time)
    plt.xlabel("Simulated Time"), plt.ylabel("Infected Nodes")
    plt.title(
        f"Infection Spread in {model.replace('_', ' ').title()} Network"), 
    plt.savefig("small_network_step.png")
    plt.show(block=False)
    plt.pause(5)
    plt.close()

# Run tests
if __name__ == "__main__":
    start_time1 = time.time()
    test_large_network("barabasi_albert")
    print("--- %s seconds ---" % (time.time() - start_time1))

    start_time1 = time.time()
    test_large_network("erdos_renyi")
    print("--- %s seconds ---" % (time.time() - start_time1))

    start_time1 = time.time()
    test_large_network("watts_strogatz")
    print("--- %s seconds ---" % (time.time() - start_time1))

    print("--- %s seconds ---" % (time.time() - start_time)) # total time from the start of the program

