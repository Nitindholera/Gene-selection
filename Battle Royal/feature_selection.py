import random
import pandas as pd
from sklearn import preprocessing
import numpy as np
from numpy.linalg import norm

class BaseBRO:

    def debug_karde_bhai(self):
        print(self.sigmoid(self.pop))

    def __init__(self, X, y, epoach):
        self.ID_POS = 0
        self.ID_LAB = 1
        self.epoach = epoach
        self.pop_size = X.shape[0]
        self.pop = self.create_pop(X, y)

    def create_pop(self,X, y):
        pop = []
        for i in range(self.pop_size):
            position = X[i]
            label = y[i]
            pop.append([position, label])
        return pop

    def get_global_best_solution(self, pop: list):
        random.shuffle(pop)
        return pop, pop[0]
    
    def cosine_similarity(self, v1, v2):
        cos_sim = np.dot(v1, v2)/(norm(v1)*norm(v2))
        return cos_sim

    def get_lab_position(self, child_pos, p1, p2):
        p1_sim = self.cosine_similarity(child_pos, p1[self.ID_POS])
        p2_sim = self.cosine_similarity(child_pos, p2[self.ID_POS])
        if p1_sim<p2_sim:
            return p2[self.ID_LAB]
        else:
            return p1[self.ID_LAB]

    def after_evolve(self):
        sorted_pop, g1 = self.get_global_best_solution(self.pop) #returns sorted population and the best solution
        g2 = sorted_pop[1] #2nd best player
        
        g1_children = [] # will contain children from g1
        g2_children = [] # will contain children from g2
        eta = 3          # distribution index used during SBC

        for idx, parent in enumerate(sorted_pop):
            if idx != 0:
                u = np.random.random() # generates a random number between [0,1)
                
                if u<=0.5:
                    beta = pow(2*u, 1/(eta+1))
                else:
                    beta = pow(1/(2*(1-u)), 1/(eta+1))

                child1, child2 = self.generate_children_SBC(parent, g1, beta)
                g1_children.append(child1)
                g1_children.append(child2) #generating and inserting the children in the list for g1
            
            if idx!=1:
                u = np.random.random() # generates a random number between [0,1)
                
                if u<=0.5:
                    beta = pow(2*u, 1/(eta+1))
                else:
                    beta = pow(1/(2*(1-u)), 1/(eta+1))

                child1, child2 = self.generate_children_SBC(parent, g2, beta)
                g2_children.append(child1)
                g2_children.append(child2) #generating and inserting the children in the list for g2

        after_pop = self.pop + g1_children + g2_children #concanated population containing initial population and the children
        random.shuffle(after_pop)

        self.pop = after_pop[:self.pop_size] #chose best n players in population for next
        return self.pop

    def generate_children_SBC(self, parent1, parent2, beta):
        temp1 = [i*(1+beta) for i in parent1[self.ID_POS]]
        temp2 = [j*(1-beta) for j in parent2[self.ID_POS]]
        child1_pos = np.array([(temp1[i]+temp2[i])*0.5 for i in range(len(temp1))]) #generate child1 position according to SBC formula
        child1_lab = self.get_lab_position(child1_pos, parent1, parent2) #finding fitness from position
        

        temp3 = [i*(1-beta) for i in parent1[self.ID_POS]]
        temp4 = [j*(1+beta) for j in parent2[self.ID_POS]]
        child2_pos = np.array([0.5*(temp3[i]+temp4[i]) for i in range(len(temp3))]) #generate child2 position according to SBC formula
        child2_lab = self.get_lab_position(child2_pos, parent1, parent2) ##finding fitness from position
        

        child1 = [child1_pos, child1_lab]
        child2 = [child2_pos, child2_lab]

        return child1, child2

    def sigmoid(self, pop: list):
        pop_size = len(pop)
        sig_pop = []
        for i in range(pop_size):
            pos = 1/(1+np.exp(pop[i][0]))
            lab = pop[i][1]
            sig_pop.append([pos, lab])
        
        return sig_pop

    def feature_score(self, pop):
        num = np.random.uniform(0.5,1/(1+np.exp(-1))) # generates a random number between [0,1)
        Is1 = []
        score = np.zeros((len(pop[0][0])))
        for i in range(len(pop)):
            lab = pop[i][self.ID_LAB]
            pos = []
            for j in pop[i][self.ID_POS]:
                if j<num:
                    pos.append(0)
                else:
                    pos.append(1)
            Is1.append([pos, lab])
            score = np.add(score, pos)
        return score

    def feature_ranking(self):
        for i in range(self.epoach):
            I = self.after_evolve()
            I_sig = self.sigmoid(I)
            score = self.feature_score(I_sig)
            print(*score, sep = ', ')


df = pd.read_csv("Datasets/Diabetic/messidor_features.csv", sep=",")

min_max_scaler = preprocessing.MinMaxScaler()
d = min_max_scaler.fit_transform(df.iloc[:,:-1].transpose())
names = df.columns[:-1]

scaled_df = pd.DataFrame(d.transpose(), columns=names) #row wise min-maxscaled
X = np.array(scaled_df.iloc[:,:])
y = np.array(df.iloc[:,-1])

optimizer = BaseBRO(X, y, 5)
optimizer.feature_ranking()