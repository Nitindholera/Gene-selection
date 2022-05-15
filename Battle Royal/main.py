from pickle import NONE
import pandas as pd
from sklearn import preprocessing, svm, metrics
import numpy as np
from mealpy.optimizer import Optimizer
from sklearn.model_selection import KFold
from copy import deepcopy
from scipy.spatial.distance import cdist
from mealpy.utils.history import History
import time
from numpy.linalg import norm

attr = {'n_iteration':[], 'min_fitness':[], 'max_accur':[], 'rft':[]}

class BRO(Optimizer):

    def __init__(self, X, y, epoch):
            self.X = X
            self.y = y
            self.w1 = 0.5
            self.w2 = 0.5
            self.pop_size = X.shape[0]
            self.epoch = epoch
            self.ID_POS = 0
            self.ID_LAB = 1
            self.ID_FIT = 2
            self.ID_DAM = 3
            self.ID_RNUM = 4
            self.ID_ACC = 5
            self.ID_RFT = 6
            self.threshold = 3
            self.g_best = None
            self.problem = {"lb": np.array([0 for _ in range(self.X.shape[1])]), "ub": np.array([1 for _ in range(self.X.shape[1])])}
            
            self.nfe_per_epoch = self.pop_size
            self.history = History()
            ## Dynamic variable
            shrink = np.ceil(np.log10(self.epoch))
            self.dyn_delta = round(self.epoch / shrink)

    def gridsearch(self, w1, w2):
        sum = 0
        for i in range(self.pop_size):
            sum += (w1*(1-self.pop[i][self.ID_ACC]) + w2*self.pop[i][self.ID_RFT])**2
        return sum

    def getWeights(self):
        w1 = 0.01 
        w2 = 1-w1
        delta = 0.01
        min_sum = self.gridsearch(w1, w2)
        min_w1 = w1

        while(w1<1):
            if self.gridsearch(w1,w2) < min_sum:
                min_sum = self.gridsearch(w1,w2)
                min_w1 = w1
            w1 = w1 + delta
            w2 = 1-w1

        return min_w1, 1-min_w1

    def createPop(self):
        pop = []
        r_num = np.random.uniform(0.5,1/(1+np.exp(-1)))
        for i in range(self.pop_size):
            position = self.X[i]
            label = self.y[i]
            fitness, acc, rft = self.get_fitness(position, r_num)
            damage = 0
            pop.append([position, label, fitness, damage, r_num, acc, rft])
        return pop
    
    def get_fitness(self, position, r_num):
        sig_pos = self.sigmoid(position)
        binary_pos = []
        for i in sig_pos:
            if i<r_num:
                binary_pos.append(0)
            else:
                binary_pos.append(1)

        selected_feature_indices = []
        for i in range(len(binary_pos)):
            if binary_pos[i]==1:
                selected_feature_indices.append(i)
        
        #if selected features indices list is empty then we will randomly select one feature
        if(len(selected_feature_indices) == 0):
            selected_feature_indices.append(np.random.randint(0,self.X.shape[1]))

        accuracy = np.zeros(5)
        clf = svm.SVC(kernel='linear')
        kf = KFold(n_splits=5, shuffle=True)
        idx = 0        

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
            # print("Accuracy",metrics.accuracy_score(y_test, y_pred))
            accuracy[idx]=(metrics.accuracy_score(y_test, y_pred))
            idx+=1
        
        fitness = self.w1*(1-accuracy.mean()) + self.w2*len(selected_feature_indices)/len(position)
        return fitness, accuracy.mean(), len(selected_feature_indices)/len(position)
    
    def find_argmin_distance(self, target_pos=None, pop=None):
        list_pos = np.array([pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
        target_pos = np.reshape(target_pos, (1, -1))
        dist_list = cdist(list_pos, target_pos, 'euclidean')
        dist_list = np.reshape(dist_list, (-1))
        idx = np.argmin(dist_list[np.nonzero(dist_list)])
        return idx
    
    def compare_agent(self, agent_a: list, agent_b: list):
        if agent_a[self.ID_FIT]<agent_b[self.ID_FIT]:
            return True
        return False

    def evolve(self, epoch, r_num):
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
                fit_new, acc_new, rft_new = self.get_fitness(pos_new, r_num)
                dam_new = self.pop[i][self.ID_DAM] - 1  ## Substract damaged hurt -1 to go next battle
                lab_new = self.pop[i][self.ID_LAB]
                self.pop[i] = [pos_new, lab_new, fit_new, dam_new, r_num, acc_new, fit_new]
                ## Update Loser
                if self.pop[j][self.ID_DAM] < self.threshold:  ## If loser not dead yet, move it based on general
                    pos_new = np.random.uniform() * (np.maximum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS]) -
                                                       np.minimum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS])) + \
                                          np.maximum(self.pop[j][self.ID_POS], self.g_best[self.ID_POS])
                    dam_new = self.pop[j][self.ID_DAM] + 1

                    self.pop[j][self.ID_FIT], self.pop[j][self.ID_ACC], self.pop[j][self.ID_RFT] = self.get_fitness(self.pop[j][self.ID_POS], r_num)
                else:  ## Loser dead and respawn again
                    pos_new = np.random.uniform(self.problem["lb"], self.problem["ub"])
                    dam_new = 0
                pos_new = np.clip(pos_new, self.problem["lb"], self.problem["ub"])
                fit_new, acc_new, rft_new = self.get_fitness(pos_new, r_num)
                lab_new = self.pop[j][self.ID_LAB]
                self.pop[j] = [pos_new, lab_new, fit_new, dam_new, r_num, acc_new, fit_new]
                nfe_epoch += 2
            else:
                ## Update Loser by following position of Winner
                self.pop[i] = deepcopy(self.pop[j])
                ## Update Winner by following position of General to protect the King and General
                pos_new = self.pop[j][self.ID_POS] + np.random.uniform() * (self.g_best[self.ID_POS] - self.pop[j][self.ID_POS])
                pos_new = np.clip(pos_new, self.problem["lb"], self.problem["ub"])
                fit_new, acc_new, rft_new = self.get_fitness(pos_new, r_num)
                dam_new = 0
                lab_new = self.pop[j][self.ID_LAB]
                self.pop[j] = [pos_new, lab_new, fit_new, dam_new, r_num, acc_new, rft_new]
                nfe_epoch += 1
        self.nfe_per_epoch = nfe_epoch
        if epoch >= self.dyn_delta:  # max_epoch = 1000 -> delta = 300, 450, >500,....
            pos_list = np.array([self.pop[idx][self.ID_POS] for idx in range(0, self.pop_size)])
            pos_std = np.std(pos_list, axis=0)
            lb = self.g_best[self.ID_POS] - pos_std
            ub = self.g_best[self.ID_POS] + pos_std
            self.problem["lb"] = np.clip(lb, self.problem["lb"], self.problem["ub"])
            self.problem["ub"] = np.clip(ub, self.problem["lb"], self.problem["ub"])
            self.dyn_delta += np.round(self.dyn_delta / 2)

    def after_evolve(self, epoch, prev_best : list):
        eta = 3          # distribution index used during SBC
        
        for idx, parent in enumerate(self.pop):
            u = np.random.random() # generates a random number between [0,1)

            if u<=0.5:
                beta = pow(2*u, 1/(eta+1))
            else:
                beta = pow(1/(2*(1-u)), 1/(eta+1))
            
            child1, child2 = self.generate_children_SBC(prev_best, parent, beta)
            #chosing agent having minimum fitness amaong child1 child2 and parent
            if parent[self.ID_FIT] < child1[self.ID_FIT] and parent[self.ID_FIT] < child2[self.ID_FIT]:
                self.pop[idx] = parent
            
            elif child1[self.ID_FIT] < child2[self.ID_FIT]:
                self.pop[idx] = child1
            else:
                self.pop[idx] = child2
            
    def generate_children_SBC(self, parent1, parent2, beta):
        temp1 = [i*(1+beta) for i in parent1[self.ID_POS]]
        temp2 = [j*(1-beta) for j in parent2[self.ID_POS]]
        child1_pos = np.array([(temp1[i]+temp2[i])*0.5 for i in range(len(temp1))]) #generate child1 position according to SBC formula
        child1_lab = self.get_lab_position(child1_pos, parent1, parent2) #finding fitness from position
        child1_fit, child1_acc, child1_rft = self.get_fitness(child1_pos, parent2[self.ID_RNUM])
        child1_dam = parent2[self.ID_DAM]
        

        temp3 = [i*(1-beta) for i in parent1[self.ID_POS]]
        temp4 = [j*(1+beta) for j in parent2[self.ID_POS]]
        child2_pos = np.array([0.5*(temp3[i]+temp4[i]) for i in range(len(temp3))]) #generate child2 position according to SBC formula
        child2_lab = self.get_lab_position(child2_pos, parent1, parent2) ##finding fitness from position
        child2_fit, child2_acc, child2_rft = self.get_fitness(child2_pos, parent2[self.ID_RNUM])
        child2_dam = parent2[self.ID_DAM]

        r_num = parent2[self.ID_RNUM]

        child1 = [child1_pos, child1_lab, child1_fit, child1_dam, r_num, child1_acc, child1_rft]
        child2 = [child2_pos, child2_lab, child2_fit, child2_dam, r_num, child2_acc, child2_rft]

        return child1, child2
    
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
    
    def sigmoid(self, pop : list):
        sig_pop = [] 
        for i in pop:
            sig_pop.append(1/(1+np.exp(-i)))
        sig_pop = np.array(sig_pop)
        return sig_pop

    def get_global_best_solution(self, pop: list):
        sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_FIT])  # Already returned a new sorted list
        return sorted_pop, deepcopy(sorted_pop[0])

    def initialization(self):
        self.pop = self.createPop()
        self.w1, self.w2 = self.getWeights()
        print(self.w1, self.w2)
        _, self.g_best = self.get_global_best_solution(self.pop)    

    def print_epoch(self, epoch, runtime):
        """
        Print out the detailed information of training process
        Args:
            epoch (int): current iteration
            runtime (float): the runtime for current iteration
        """
        print(f"> Epoch: {epoch}, Current best: {self.history.list_current_best[-1][self.ID_FIT]}, "
                f"Global best: {self.history.list_global_best[-1][self.ID_FIT]}, Runtime: {runtime:.5f} seconds")
    
    def get_better_solution(self, agent1: list, agent2: list):
        if agent1[self.ID_FIT] < agent2[self.ID_FIT]:
            return deepcopy(agent1)
        return deepcopy(agent2)

    def update_global_best_solution(self, pop=None, save=True):
        """
        Update the global best solution saved in variable named: self.history_list_g_best
        Args:
            pop (list): The population of pop_size individuals
            save (bool): True if you want to add new current global best and False if you just want update the current one.

        Returns:
            Sorted population and the global best solution
        """

        sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_FIT])
        
        current_best = sorted_pop[0]
        # self.history_list_c_best.append(current_best)
        # better = self.get_better_solution(current_best, self.history_list_g_best[-1])
        # self.history_list_g_best.append(better)
        if save:
            self.history.list_current_best.append(current_best)
            better = self.get_better_solution(current_best, self.history.list_global_best[-1])
            self.history.list_global_best.append(better)
            return deepcopy(sorted_pop), deepcopy(better)
        else:
            local_better = self.get_better_solution(current_best, self.history.list_current_best[-1])
            self.history.list_current_best[-1] = local_better
            global_better = self.get_better_solution(current_best, self.history.list_global_best[-1])
            self.history.list_global_best[-1] = global_better
            return deepcopy(sorted_pop), deepcopy(global_better)

    def solve(self):
        self.initialization()
        self.history.save_initial_best(self.g_best)

        for epoch in range(self.epoch):
            time_epoch = time.time()

            r_num = np.random.uniform(0.5,1/(1+np.exp(-1)))
            self.evolve(epoch, r_num)
            self.after_evolve(epoch, self.history.list_current_best[-1])

            self.pop, self.g_best = self.update_global_best_solution(self.pop)
            # print(self.g_best)
            time_epoch = time.time() - time_epoch
            self.history.list_epoch_time.append(time_epoch)
            self.history.list_population.append(deepcopy(self.pop))
            self.print_epoch(epoch + 1, time_epoch) 
            attr['n_iteration'].append(epoch)
            attr["min_fitness"].append(self.g_best[self.ID_FIT])
            attr["max_accur"].append(self.g_best[self.ID_ACC])
            attr["rft"].append(self.g_best[self.ID_RFT])
            # attr = {'n_iteration':[], 'min_fitness':[], 'max_accur':[], 'rft':[]}

        # self.save_optimization_process() 
        return self.g_best
            
    def get_predicted_label(self, best_player):
        sig_pos = self.sigmoid(best_player[self.ID_POS])
        binary_pos = []
        for i in sig_pos:
            if i<best_player[self.ID_RNUM]:
                binary_pos.append(0)
            else:
                binary_pos.append(1)
        
        selected_feature_indices = []
        for i in range(len(binary_pos)):
            if binary_pos[i]==1:
                selected_feature_indices.append(i)
        
        #if selected features indices list is empty then we will randomly select one feature
        if(len(selected_feature_indices) == 0):
            selected_feature_indices.append(np.random.randint(0,self.X.shape[1]))

        clf = svm.SVC(kernel='linear')
        kf = KFold(n_splits=5, shuffle=True)
        idx = 0        

        top_features_indices = sorted(selected_feature_indices, reverse=True)

        X2 = [[] for i in range(X.shape[0])]

        while len(top_features_indices)>0:
            z=top_features_indices.pop()
            for i in range(X.shape[0]):
                X2[i].append(X[i][int(z)])

        X2 = np.array(X2)
        clf.fit(X2,y)
        ypred = clf.predict(X2)

        return ypred

df = pd.read_csv("Datasets/Australian/australian.csv", sep=" ")
min_max_scaler = preprocessing.MinMaxScaler()
d = min_max_scaler.fit_transform(df.iloc[:,:-1])
names = df.columns[:-1]

scaled_df = pd.DataFrame(d, columns=names) #column wise min-maxscaled
X = np.array(scaled_df.iloc[:,:])
y = np.array(df.iloc[:,-1])

optimizer = BRO(X, y, 100)
best_player = optimizer.solve()
y_pred = optimizer.get_predicted_label(best_player)
print(metrics.confusion_matrix(y,y_pred))

D = pd.DataFrame(attr)
D.to_csv('attribute.csv', index=None)