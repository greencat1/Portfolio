from tqdm import tqdm
import random
class VolatileThreshold:
    def __init__(self, graph):
        self.G = graph
        self.config = None
        self.status = {n: 0 for n in graph.nodes}
        self.threshold = {n: 0 for n in graph.nodes}
        self.num_sample = 0  # since the graph is connected
        self.N = len(graph.nodes)
        
    def set_initial_status(self, config):
        self.config = config
        # set threshold
        thred = config.__dict__['config']['nodes']['threshold']
        for n in self.G.nodes:
            self.threshold[n] = thred[n]
        # set number of samples
        self.num_sample = config.__dict__['config']['model']['num_sample']
        # set seed nodes
        if 'fraction_infected' in config.__dict__['config']['model']:
                seed_nodes = random.sample(
                list(self.G.nodes()), int(config.__dict__['config']['model']['fraction_infected'] * len(self.G.nodes())))
        else:
            seed_nodes = config.__dict__['config']['status']['Infected']
        for n in seed_nodes:
            self.status[n] = 1
    
    def iteration(self):
        one = 0  # change parameter name
        tmp = {n: self.status[n] for n in self.G.nodes}
        for n in self.G.nodes:
            if self.status[n] == 0:
                # initiate a counter for its infected neighbors
                ### BEGIN SOLUTION 
                n_infected_neighbors = 0
                ### END SOLUTION
                
                # get a sample from its heighbors
                ### BEGIN SOLUTION 
                neighbors = list(self.G.neighbors(n))
                if len(neighbors) > self.num_sample:
                    neighbors = random.sample(neighbors, self.num_sample)
                ### END SOLUTION
                
                # count the number of infected neighbors in the sampled neighbors
                ### BEGIN SOLUTION 
                for v in neighbors:
                    n_infected_neighbors += tmp[v]
                ### END SOLUTION
                
                # compute the ratio of infectious in its sampled neighbors
                ### BEGIN SOLUTION 
                sample_rate = n_infected_neighbors/len(neighbors)
                ### END SOLUTION
                
                # if the ratio is heigher than its threshold, the node becomes infected
                ### BEGIN SOLUTION 
                if sample_rate >= self.threshold[n]:
                    self.status[n] = 1
                    one += self.status[n]
                ### END SOLUTION
            else:
                one += 1
        return one
    
    def iteration_bunch(self, bunch_size):
        results = []
        one = 0
        for n in self.G.nodes:
            one += self.status[n]
        results.append({0: self.N - one, 1: one})
        for i in tqdm(range(bunch_size-1)):
            one = self.iteration()
            results.append({0: self.N - one, 1: one})
        return results
