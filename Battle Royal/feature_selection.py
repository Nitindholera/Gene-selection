import random
import pandas as pd
from sklearn import preprocessing
import numpy as np
from numpy.linalg import norm
from sklearn import metrics, preprocessing, svm
from sklearn.model_selection import KFold, train_test_split
from copy import deepcopy
from mealpy.optimizer import Optimizer

class BaseBRO(Optimizer):

    def debug(self):
        pass

    def __init__(self, X, y, epoach):
        self.w1 = 0.5
        self.w2 = 0.5
        self.ID_POS = 0
        self.ID_LAB = 1
        self.ID_FIT = 2
        self.ID_DAM = 3
        self.epoach = epoach
        self.pop_size = X.shape[0]
        self.pop = self.create_pop(X, y)

    def create_pop(self,X, y):
        pop = []
        for i in range(self.pop_size):
            position = X[i]
            label = y[i]
            fitness = self.get_fitness_position(position)
            damage = 0
            pop.append([position, label, fitness, damage])
        return pop

    def get_fitness_position(self, position):
        df = pd.read_csv("Datasets/Diabetic/messidor_features.csv", sep=",")

        min_max_scaler = preprocessing.MinMaxScaler()
        d = min_max_scaler.fit_transform(df.iloc[:,:-1].transpose())
        names = df.columns[:-1]

        scaled_df = pd.DataFrame(d.transpose(), columns=names) #row wise min-maxscaled
        X = np.array(scaled_df.iloc[:,:])
        y = np.array(df.iloc[:,-1])

        num = np.random.uniform(0,1)
        new_pos = []
        for i in range(len(position)):
            if position[i]<num:
                new_pos.append(0)
            else:
                new_pos.append(1)

        selected_feature_indices = []
        for i in range(len(new_pos)):
            if new_pos[i]==1:
                selected_feature_indices.append(i)
        

        accuracy = np.zeros(5)

        clf = svm.SVC(kernel='linear')

        kf = KFold(n_splits=5, shuffle=True)

        idx = 0
        #print(X.shape)
        


        top_features_indices = sorted(selected_feature_indices, reverse=True)

        X2 = [[] for i in range(X.shape[0])]

        while len(top_features_indices)>0:
            z=top_features_indices.pop()
            for i in range(X.shape[0]):
                X2[i].append(X[i][int(z)])

        X2 = np.array(X2)
        for train_index, test_index in kf.split(X2):
            X_train, X_test = X2[train_index], X2[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train,y_train)
            y_pred = clf.predict(X_test)
            #print("Accuracy",metrics.accuracy_score(y_test, y_pred))
            accuracy[idx]=(metrics.accuracy_score(y_test, y_pred))
            idx+=1
            
            
        #print("min accuracy", accuracy.min())
        #print("max accuracy", accuracy.max())
        #print("avg accuracy", accuracy.mean())
        fitness = self.w1*accuracy.mean() + self.w2*len(selected_feature_indices)/len(position)
        print(fitness)
        return fitness

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

    def evolve(self, epoch):
        """
        Args:
            epoch (int): The current iteration
        """
        nfe_epoch = 0
        for i in range(self.pop_size):
            # Compare ith soldier with nearest one (jth)
            j = self.find_argmin_distance(self.pop[i][self.ID_POS], self.pop)
            if self.compare_agent(self.pop[i], self.pop[j]):
                ## Update Winner based on global best solution
                pos_new = self.pop[i][self.ID_POS] + np.random.uniform() * \
                          np.mean(np.array([self.pop[i][self.ID_POS], self.g_best[self.ID_POS]]), axis=0)
                fit_new = self.get_fitness_position(pos_new)
                dam_new = self.pop[i][self.ID_DAM] - 1  ## Substract damaged hurt -1 to go next battle
                self.pop[i] = [pos_new, fit_new, dam_new]
                ## Update Loser
                if self.pop[j][self.ID_DAM] < self.threshold:  ## If loser not dead yet, move it based on general
                    pos_new = np.random.uniform() * (np.maximum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS]) -
                                                       np.minimum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS])) + \
                                          np.maximum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS])
                    dam_new = self.pop[j][self.ID_DAM] + 1

                    self.pop[j][self.ID_FIT] = self.get_fitness_position(self.pop[j][self.ID_POS])
                else:  ## Loser dead and respawn again
                    pos_new = np.random.uniform(self.problem.lb, self.problem.ub)
                    dam_new = 0
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                self.pop[j] = [pos_new, fit_new, dam_new]
                nfe_epoch += 2
            else:
                ## Update Loser by following position of Winner
                self.pop[i] = deepcopy(self.pop[j])
                ## Update Winner by following position of General to protect the King and General
                pos_new = self.pop[j][self.ID_POS] + np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[j][self.ID_POS])
                pos_new = self.amend_position_faster(pos_new)
                fit_new = self.get_fitness_position(pos_new)
                dam_new = 0
                self.pop[j] = [pos_new, fit_new, dam_new]
                nfe_epoch += 1
        self.nfe_per_epoch = nfe_epoch
        if epoch >= self.dyn_delta:  # max_epoch = 1000 -> delta = 300, 450, >500,....
            pos_list = np.array([self.pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
            pos_std = np.std(pos_list, axis=0)
            lb = self.g_best[self.ID_POS] - pos_std
            ub = self.g_best[self.ID_POS] + pos_std
            self.problem.lb = np.clip(lb, self.problem.lb, self.problem.ub)
            self.problem.ub = np.clip(ub, self.problem.lb, self.problem.ub)
            self.dyn_delta += np.round(self.dyn_delta / 2)


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
            pos = 1/(1+np.exp(pop[i][self.ID_POS]))
            lab = pop[i][self.ID_LAB]
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
        #new_arr = [old arr[i] fpr i in indees]
        overall_score = np.zeros((len(self.pop[0][0])))
        for i in range(self.epoach):
            I = self.after_evolve()
            I_sig = self.sigmoid(I)
            score = self.feature_score(I_sig)
            overall_score = np.add(score, overall_score)
        
        indices = sorted([(x,i) for i,x in enumerate(overall_score)])
        top_fetures = indices[(len(self.pop[0][0]))//2:]
        return top_fetures



df = pd.read_csv("Datasets/Diabetic/messidor_features.csv", sep=",", header=None)

min_max_scaler = preprocessing.MinMaxScaler()
d = min_max_scaler.fit_transform(df.iloc[:,:-1].transpose())
names = df.columns[:-1]

scaled_df = pd.DataFrame(d.transpose(), columns=names) #row wise min-maxscaled
X = np.array(scaled_df.iloc[:,:])
y = np.array(df.iloc[:,-1])
    
optimizer = BaseBRO(X, y, 20)
#top_features= optimizer.feature_ranking() #prints a array of tuples of feature indices and feature score
#optimizer.debug()



# accuracy = np.zeros(5)

# clf = svm.SVC(kernel='linear')

# kf = KFold(n_splits=5, shuffle=True)

# idx = 0
# #print(X.shape)
# top_features_indices = np.zeros(len(top_features))
# for i in range(len(top_features)):
#     top_features_indices[i] = top_features[i][1]


# top_features_indices = sorted(top_features_indices, reverse=True)

# X2 = [[] for i in range(X.shape[0])]

# while len(top_features_indices)>0:
#     z=top_features_indices.pop()
#     for i in range(X.shape[0]):
#         X2[i].append(X[i][int(z)])

# X2 = np.array(X2)
# for train_index, test_index in kf.split(X2):
#     X_train, X_test = X2[train_index], X2[test_index]
#     y_train, y_test = y[train_index], y[test_index]

#     clf.fit(X_train,y_train)
#     y_pred = clf.predict(X_test)
#     #print("Accuracy",metrics.accuracy_score(y_test, y_pred))
#     accuracy[idx]=(metrics.accuracy_score(y_test, y_pred))
#     idx+=1
    
    
# print("min accuracy", accuracy.min())
# print("max accuracy", accuracy.max())
# print("avg accuracy", accuracy.mean())
