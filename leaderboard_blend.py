#leaderboard_blend

import numpy as np

# include 1s
num_predictors = 21
qual_length = 10000
qual_sq = 0.52380



print 'loading qual preds'

# Bias
p0 = np.ones((qual_length, 1))
mse0 = (1-0.52380)
# Logistic regression
p1 = np.loadtxt('logistic_linear_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p1 = np.maximum(np.minimum(p1, 1), 0) # note rounding first improves pred by .006 for 1 model fit
mse1 = (1-.85420)
# NN
p2 = np.loadtxt('nn_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p2 = np.maximum(np.minimum(p2, 1), 0)
mse2 = (1-.85180)
# GBDT
p3 = np.loadtxt('gbdt_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p3 = np.maximum(np.minimum(p3, 1), 0)
mse3 = (1-.84800)
# Naive bayes
p4 = np.loadtxt('naive_bayes_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p4 = np.maximum(np.minimum(p4, 1),0)
mse4 = (1-.82400)
# Random forest
p5 = np.loadtxt('rf_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p5 = np.maximum(np.minimum(p5, 1),0)
mse5 = (1-.82780)
# Adaboost
p6 = np.loadtxt('adaboost_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p6 = np.maximum(np.minimum(p6, 1),0)
mse6 = (1-0.83320)
# SGDClassifier
p7 = np.loadtxt('naive_linear_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p7 = np.maximum(np.minimum(p7, 1),0)
mse7 = (1-0.84680)

# Just Ridge(), impressively
p8 = np.loadtxt('pure_linear_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p8 = np.maximum(np.minimum(p8, 1),0)
mse8 = (1-0.84700)

# manual nonlinear transform with ridge
p9 = np.loadtxt('nonlinear_ridge_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p9 = np.maximum(np.minimum(p9, 1),0)
mse9 = (1-0.85320)
# Pipelined RBF SVC on whitened raw PCA
p10 = np.loadtxt('RBF_SVC_whitenedRawPCA__pred_labels.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p10 = np.maximum(np.minimum(p10, 1), 0)
mse10 = (1-0.85140)
# Pipelined Linear SVC on truncated PCA
p11 = np.loadtxt('Linear_SVC_truncPCA_pred_labels.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p11 = np.maximum(np.minimum(p11, 1), 0)
mse11 = (1-0.84420)

# Pipelined RBF SVC on whitened truncated PCA
p12 = np.loadtxt('RBF_SVC_whitenedTruncPCA__pred_labels.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p12 = np.maximum(np.minimum(p12, 1), 0)
mse12 = (1-0.84020)

# Extra trees
p13 = np.loadtxt('et_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p13 = np.maximum(np.minimum(p13, 1), 0)
mse13 = (1-0.84640)

# Kernel ridge regression with a polynomial kernel of degree 3
p14 = np.loadtxt('kernel_ridge_polynomial_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p14 = np.maximum(np.minimum(p14, 1), 0)
mse14 = (1-0.85380)

# Kernel ridge regression with a laplacian kernel
p15 = np.loadtxt('kernel_ridge_laplacian_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p15 = np.maximum(np.minimum(p15, 1), 0)
mse15 = (1-0.84780)

# Kernel ridge regression with a RBF kernel
p16 = np.loadtxt('kernel_ridge_rbf_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p16 = np.maximum(np.minimum(p16, 1), 0)
mse16 = (1-0.80940)

# autoencoder output passed into neural network
p17 = np.loadtxt('autoencoded_nn_result.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p17 = np.maximum(np.minimum(p17, 1), 0)
mse17 = (1-0.76860)

# Improved_Pipeline__pred_labels
p18 = np.loadtxt('Improved_Pipeline__pred_labels.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p18 = np.maximum(np.minimum(p18, 1), 0)
mse18 = (1-0.84820)

# XGBoost
p19 = np.loadtxt('XGBoost_pred_labels.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p19 = np.maximum(np.minimum(p19, 1), 0)
mse19 = (1-0.84720)

# XGBoost 2
p20 = np.loadtxt('XGBoost_refined_pred_labels.txt', skiprows=1, delimiter=",")[:,1].reshape((qual_length, 1))
p20 = np.maximum(np.minimum(p20, 1), 0)
mse20 = (1-0.85020)


Xty = np.zeros((num_predictors))

all_preds = np.hstack((p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20))

Xty[0] = .5*(qual_sq + np.mean(np.power(p0,2))-mse0)*qual_length
Xty[1] = .5*(qual_sq + np.mean(np.power(p1,2))-mse1)*qual_length
Xty[2] = .5*(qual_sq + np.mean(np.power(p2,2))-mse2)*qual_length
Xty[3] = .5*(qual_sq + np.mean(np.power(p3,2))-mse3)*qual_length
Xty[4] = .5*(qual_sq + np.mean(np.power(p4,2))-mse4)*qual_length
Xty[5] = .5*(qual_sq + np.mean(np.power(p5,2))-mse5)*qual_length
Xty[6] = .5*(qual_sq + np.mean(np.power(p6,2))-mse6)*qual_length
Xty[7] = .5*(qual_sq + np.mean(np.power(p7,2))-mse7)*qual_length
Xty[8] = .5*(qual_sq + np.mean(np.power(p8,2))-mse8)*qual_length
Xty[9] = .5*(qual_sq + np.mean(np.power(p9,2))-mse9)*qual_length
Xty[10] = .5*(qual_sq + np.mean(np.power(p10,2))-mse10)*qual_length
Xty[11] = .5*(qual_sq + np.mean(np.power(p11,2))-mse11)*qual_length
Xty[12] = .5*(qual_sq + np.mean(np.power(p12,2))-mse12)*qual_length
Xty[13] = .5*(qual_sq + np.mean(np.power(p13,2))-mse13)*qual_length
Xty[14] = .5*(qual_sq + np.mean(np.power(p14,2))-mse14)*qual_length
Xty[15] = .5*(qual_sq + np.mean(np.power(p15,2))-mse15)*qual_length
Xty[16] = .5*(qual_sq + np.mean(np.power(p16,2))-mse16)*qual_length
Xty[17] = .5*(qual_sq + np.mean(np.power(p17,2))-mse17)*qual_length
Xty[18] = .5*(qual_sq + np.mean(np.power(p18,2))-mse18)*qual_length
Xty[19] = .5*(qual_sq + np.mean(np.power(p19,2))-mse19)*qual_length
Xty[20] = .5*(qual_sq + np.mean(np.power(p20,2))-mse20)*qual_length

print 'learning'
l = 0.01#0.001
beta = np.dot(np.linalg.inv(np.dot(all_preds.T, all_preds)+ l*qual_length*np.eye(num_predictors)), Xty)
print beta, np.sum(beta)
pred = np.dot(all_preds, beta.reshape((num_predictors, 1)))

pred = np.round(np.maximum(np.minimum(pred, 1), 0))
result_col_1 = (np.array(range(len(pred)))+1).reshape((len(pred),1))
results = np.concatenate((result_col_1,pred.reshape((len(pred),1))), axis = 1)
np.savetxt('quiz_blended'+str(l)+'.txt', results, fmt="%d", header='Id,Prediction', delimiter=',', comments="")
print 'predicted number of ones:', np.sum(results[:,1])
