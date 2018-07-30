import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import f, norm
import math

class PCA_Model():
	def __init__(self, X_train, X_test, no_of_eigenvals):
		self.X_train = X_train
		self.X_test = X_test
		self.no_of_eigenvals = no_of_eigenvals
		
		#Scale the training dataset, return model and scaled training dataset
		self.X_train_norm, self.X_train_model = self.__scale_dataset()
		
		#Calculate eigenvalues and eigenvectors
		self.eigen_vec, self.eigen_vals, self.eigen_vals_res = self.__singular_value_decomposition()
		
		#Construct projection matrix and diagonal eigenvalues matrix
		self.P = np.hstack(self.eigen_vec)
		self.A = np.diag(self.eigen_vals)
		
	def __scale_dataset(self):
		sc = StandardScaler()
		X_train_model = sc.fit(self.X_train)
		X_train_norm = X_train_model.transform(self.X_train)
		return X_train_norm, X_train_model
		
	def __singular_value_decomposition(self):
		#Calculate covariance matrix and get eigen values and eigen vectors
		cov_mat = np.cov(self.X_train_norm.T)
		eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
		
		#Sort eigenvectors and eigenvalues by descending
		eigen_pairs_all =[(np.abs(eigen_vals[i]),eigen_vecs[:,i]) for i in range(len(eigen_vals))]
		eigen_pairs_all.sort(reverse=True)

		#Select top K eigenvectors and eigenvalues
		eigen_pairs = eigen_pairs_all[:self.no_of_eigenvals]
		eigen_vec = [(x[1][:, np.newaxis]) for x in eigen_pairs]
		eigen_vals_list = [x[0] for x in eigen_pairs]

		#Get residual eigenvalues
		eigen_pairs = eigen_pairs_all[self.no_of_eigenvals:]
		eigen_vals_list_res = [x[0] for x in eigen_pairs]
		
		return eigen_vec, eigen_vals_list, eigen_vals_list_res
		
	def calculate_statistics(self, type):
		X_test_norm = self.X_train_model.transform(self.X_test)
		if type == 'T_Squared':
			self.T_Squared = np.dot(np.dot(np.dot(X_test_norm,self.P),np.linalg.inv(self.A)),np.dot(self.P.T,X_test_norm.T)).diagonal()	
			return self.T_Squared
		if type == 'SPE':
			self.SPE = np.dot(X_test_norm, np.dot(np.diag([1]*np.dot(self.P,self.P.T).shape[1]) - np.dot(self.P,self.P.T), X_test_norm.T)).diagonal()
			return self.SPE
			
	def calculate_threshold(self, type, alpha):
		p, N, alpha = self.no_of_eigenvals, self.X_train.shape[0], 1 - alpha
		if type == 'T_Squared':
			self.T_Squared_TH = f.ppf(alpha, p, N-p)*(p*(N-1)*(N+1)/(N**2-N*p))
			return self.T_Squared_TH
			
		if type == 'SPE':
			c = norm.ppf(alpha)
			theta_1 = sum([x**1 for x in self.eigen_vals_res])
			theta_2 = sum([x**2 for x in self.eigen_vals_res])
			theta_3 = sum([x**3 for x in self.eigen_vals_res])
			h_0 = 1 - (2*theta_1*theta_3)/(3*theta_2**2)
			self.SPE_TH = theta_1*(c*math.sqrt(2*theta_2*h_0**2)/theta_1 + 1 + theta_2*h_0*(h_0 - 1)/theta_1**2)**(1/h_0)
			return self.SPE_TH
			
	def validate_model(self, type, method):
		if (type == 'T_Squared') & (method == 'fdr'):	
			fault_forecast_vec = [1 if x == True else 0 for x in self.T_Squared > self.T_Squared_TH]
			fault_forecast_vec_FDR = fault_forecast_vec[160:]
			FDR = sum(fault_forecast_vec_FDR)/len(fault_forecast_vec_FDR)
			return FDR
		if (type == 'T_Squared') & (method == 'far'):
			fault_forecast_vec = [1 if x == True else 0 for x in self.T_Squared > self.T_Squared_TH]
			fault_forecast_vec_FAR = fault_forecast_vec[:160]
			FAR = sum(fault_forecast_vec_FAR)/len(fault_forecast_vec_FAR)
			return FAR
		if (type == 'SPE') & (method == 'fdr'):
			fault_forecast_vec = [1 if x == True else 0 for x in self.SPE > self.SPE_TH]
			fault_forecast_vec_FDR = fault_forecast_vec[160:]
			FDR = sum(fault_forecast_vec_FDR)/len(fault_forecast_vec_FDR)
			return FDR
		if (type == 'SPE') & (method == 'far'):
			fault_forecast_vec = [1 if x == True else 0 for x in self.SPE > self.SPE_TH]
			fault_forecast_vec_FAR = fault_forecast_vec[:160]
			FAR = sum(fault_forecast_vec_FAR)/len(fault_forecast_vec_FAR)
			return FAR
		return None