#leaderboard_blend

import numpy as np

# include 1s
num_predictors = 9
qual_length = 10000
qual_sq = 0.52380



print 'loading qual preds'

# Bias TODO
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
'''
# vae_u-autorec (w/ probe) 0.05757793
p9 = h5py.File('../mu/vae_uautorec_reduced_ratings_probe.h5')['qual_predictions'][:].reshape((qual_length, 1))
p9 = np.maximum(np.minimum(p9, 5), 1)
mse9 = 0.91448**2
# vae_autorec 0.00449524
p10 = h5py.File('../mu/vae_autorec_reduced_ratings.h5')['qual_predictions'][:].reshape((qual_length, 1))
p10 = np.maximum(np.minimum(p10, 5), 1)
mse10 = .92079**2
# vae_u-autorec 0.02401581
p11 = h5py.File('../mu/vae_uautorec_reduced_ratings.h5')['qual_predictions'][:].reshape((qual_length, 1))
p11 = np.maximum(np.minimum(p11, 5), 1)
mse11 = .91869**2
# timeSVD++ 0.07255479
p12 = np.loadtxt('../timeSVDplusplus-master/timeSVD++/preds893.txt').reshape((qual_length, 1))
p12 = np.maximum(np.minimum(p12, 5), 1) # note rounding first improves pred by .006 for 1 model fit
mse12 = .89412**2
# SVD++ 0.04098122
p13 = np.loadtxt('../timeSVDplusplus-master/SVD++/preds898.txt').reshape((qual_length, 1))
p13 = np.maximum(np.minimum(p13, 5), 1) # note rounding first improves pred by .006 for 1 model fit
mse13 = 0.89998**2
# SVD 0.05908947
p14 = np.loadtxt('../timeSVDplusplus-master/SVD/svd_preds903.txt').reshape((qual_length, 1))
p14 = np.maximum(np.minimum(p14, 5), 1) # note rounding first improves pred by .006 for 1 model fit
mse14 = 0.9046**2
# vae u-autorec no baseline (without probe) 0.02618476
p15 = h5py.File('../mu/vae_uautorec_reduced_ratings_no_baseline.h5')['qual_predictions'][:].reshape((qual_length, 1))
p15 = np.maximum(np.minimum(p15, 5), 1)
mse15 = .93619**2
# TimeSVD++ (w/ probe) (300 factors) 0.1036974
p16 = np.loadtxt('../timeSVDplusplus-master/timeSVD++/preds885_300.txt').reshape((qual_length, 1))
p16 = np.maximum(np.minimum(p16, 5), 1) # note rounding first improves pred by .006 for 1 model fit
mse16 = .8842**2
# vae u-autorec no baseline (with probe) 0.02846272
p17 = h5py.File('../mu/vae_uautorec_reduced_ratings_no_baseline_probe.h5')['qual_predictions'][:].reshape((qual_length, 1))
p17 = np.maximum(np.minimum(p17, 5), 1)
mse17 = .93019**2
# u autorec (with probe) 0.02930931
p18 = h5py.File('../mu/uautorec_reduced_ratings_probe.h5')['qual_predictions'][:].reshape((qual_length, 1))
p18 = np.maximum(np.minimum(p18, 5), 1)
mse18 = .92192**2
# movie frequencies 0.00547659
p19 = num_users[qual_items].reshape((qual_length, 1))
mse19 = 2.1492**2
# user frequencies 0.00337973
p20 = num_movies[qual_users].reshape((qual_length, 1))
mse20 = 1.73015**2
# time of rating 0.006446
p21 = (np.log(qual_times+1)/np.log(6.76)).reshape((qual_length, 1))
mse21 = 1.17486**2
# timesvd++ baselinemode 0.04267296
p22 = h5py.File('../mu/timesvd++_baselined.h5')['qual_predictions'][:].reshape((qual_length, 1))
p22 = np.maximum(np.minimum(p22, 5), 1)
mse22 = .93502**2
# RBM (DK) (with probe) 0.03197938
p23 = h5py.File('../mu/rbm_qual.h5')['qual_predictions'][:].reshape((qual_length, 1))
p23 = np.maximum(np.minimum(p23, 5), 1)
mse23 = .90586**2
# RBM  (DK) 0.01946677
p24 = h5py.File('../mu/rbm_qual_no_probe.h5')['qual_predictions'][:].reshape((qual_length, 1))
p24 = np.maximum(np.minimum(p24, 5), 1)
mse24 = .91239**2
# autorec no base (with probe) 0.01179067
p25 = h5py.File('../mu/autorec_no_base.h5')['qual_predictions'][:].reshape((qual_length, 1))
p25 = np.maximum(np.minimum(p25, 5), 1)
mse25 = .93142**2
# timesvd++ (without valid) 0.0739124
p26 = np.loadtxt('../timeSVDplusplus-master/timeSVD++/preds885_no_valid.txt').reshape((qual_length, 1))
p26 = np.maximum(np.minimum(p26, 5), 1)
mse26 = .88488**2
# other 10 0.12418904
p27 = np.zeros((qual_length, 1))#np.loadtxt('other_mu_10.txt').reshape((qual_length, 1))
#p27 = np.maximum(np.minimum(p27, 5), 1)
mse27 = qual_sq#0.87392**2
# bellkor baseline 0.02363115
p28 = h5py.File('../mu/qual_baselined.h5')['qual_predictions'][:].reshape((qual_length, 1))
p28 = np.maximum(np.minimum(p28, 5), 1)
mse28 = 0.96028**2
# other 10_2 0.13750554
p29 = np.zeros((qual_length, 1))#np.loadtxt('other_mu_10_2.txt').reshape((qual_length, 1))
#p29 = np.maximum(np.minimum(p29, 5), 1)
mse29 = qual_sq#0.87375**2
# other 10_3 0.12526214
p30 = np.zeros((qual_length, 1))#np.loadtxt('other_mu_10_3.txt').reshape((qual_length, 1))
#p30 = np.maximum(np.minimum(p30, 5), 1)
mse30 = qual_sq#0.87391**2
# other 5 0.37920861
p31 = np.loadtxt('other_mu_38.txt').reshape((qual_length, 1))
p31 = np.maximum(np.minimum(p31, 5), 1)
mse31 = 0.8689**2'''

Xty = np.zeros((num_predictors))

all_preds = np.hstack((p0, p1, p2, p3, p4, p5, p6, p7, p8))#, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28, p29, p30, p31))

Xty[0] = .5*(qual_sq + np.mean(np.power(p0,2))-mse0)*qual_length
Xty[1] = .5*(qual_sq + np.mean(np.power(p1,2))-mse1)*qual_length
Xty[2] = .5*(qual_sq + np.mean(np.power(p2,2))-mse2)*qual_length
Xty[3] = .5*(qual_sq + np.mean(np.power(p3,2))-mse3)*qual_length
Xty[4] = .5*(qual_sq + np.mean(np.power(p4,2))-mse4)*qual_length
Xty[5] = .5*(qual_sq + np.mean(np.power(p5,2))-mse5)*qual_length
Xty[6] = .5*(qual_sq + np.mean(np.power(p6,2))-mse6)*qual_length
Xty[7] = .5*(qual_sq + np.mean(np.power(p7,2))-mse7)*qual_length
Xty[8] = .5*(qual_sq + np.mean(np.power(p8,2))-mse8)*qual_length
'''Xty[9] = .5*(qual_sq + np.mean(np.power(p9,2))-mse9)*qual_length
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
Xty[21] = .5*(qual_sq + np.mean(np.power(p21,2))-mse21)*qual_length
Xty[22] = .5*(qual_sq + np.mean(np.power(p22,2))-mse22)*qual_length
Xty[23] = .5*(qual_sq + np.mean(np.power(p23,2))-mse23)*qual_length
Xty[24] = .5*(qual_sq + np.mean(np.power(p24,2))-mse24)*qual_length
Xty[25] = .5*(qual_sq + np.mean(np.power(p25,2))-mse25)*qual_length
Xty[26] = .5*(qual_sq + np.mean(np.power(p26,2))-mse26)*qual_length
Xty[27] = .5*(qual_sq + np.mean(np.power(p27,2))-mse27)*qual_length
Xty[28] = .5*(qual_sq + np.mean(np.power(p28,2))-mse28)*qual_length
Xty[29] = .5*(qual_sq + np.mean(np.power(p29,2))-mse29)*qual_length
Xty[30] = .5*(qual_sq + np.mean(np.power(p30,2))-mse30)*qual_length
Xty[31] = .5*(qual_sq + np.mean(np.power(p31,2))-mse31)*qual_length'''
print 'learning'
l = 0.001
beta = np.dot(np.linalg.inv(np.dot(all_preds.T, all_preds)+ l*qual_length*np.eye(num_predictors)), Xty)
print beta
pred = np.dot(all_preds, beta.reshape((num_predictors, 1)))

pred = np.round(np.maximum(np.minimum(pred, 1), 0))
result_col_1 = (np.array(range(len(pred)))+1).reshape((len(pred),1))
results = np.concatenate((result_col_1,pred.reshape((len(pred),1))), axis = 1)
np.savetxt('quiz_blended.txt', results, fmt="%d", header='Id,Prediction', delimiter=',', comments="")
print(np.sum(results[:,1]))
'''
# get training data for large-scale postprocessing
print 'preparing training data predictions'
ratings = datafile['train_rating_list'][:]
probe_ratings = datafile['probe_rating_list'][:]
train_probe_users = np.concatenate((users, probe_users))
train_probe_items = np.concatenate((items, probe_items))

qual_length = len(train_probe_items)
# get in mu order
ordering = np.lexsort((train_probe_users, train_probe_items))
train_probe_ratings = np.concatenate((ratings, probe_ratings))[ordering]
train_probe_times = np.concatenate((times, probe_times))[ordering]
print 'aggregating predictors'
batch_size = 10000000


def get_train_ratings_ordered(filename, i, order=True):
    f = h5py.File(filename)
    t = min(i + batch_size, qual_length)
    try:
        trains = f['train_rating_list'][:]
        if len(trains) <= 98291669:
            res = (train_probe_ratings[i:t] - np.concatenate((trains, f['probe_rating_list']))[ordering][i:t]).reshape((t-i, 1))

        elif order:
            res = (train_probe_ratings[i:t] - trains[ordering][i:t]).reshape((t-i, 1))
        else:
            res =  (train_probe_ratings[i:t]-trains[i:t]).reshape((t-i, 1))
    except KeyError:
        trains = f['train_predictions'][:]
        if len(trains) <= 98291669:
            res = (np.concatenate((trains, f['probe_predictions']))[ordering][i:t]).reshape((t-i, 1))

        elif order:
            res = (trains[ordering][i:t]).reshape((t-i, 1))
        else:
            res =  (trains[i:t]).reshape((t-i, 1))
    f.close()
    return res

final = np.zeros(qual_length)
import progressbar
bar = progressbar.ProgressBar(maxval=qual_length, widgets=["Testing: ",
                                                         progressbar.Bar(
                                                             '=', '[', ']'),
                                                         ' ', progressbar.Percentage(),

                                                         ' ', progressbar.ETA()]).start()
for i in range(0, qual_length, batch_size):
    bar.update(i)
    #predictors
    # Bias
    t = min(i + batch_size, qual_length)
    p0 = np.ones((t-i, 1))

    # TimeSVD++ (w/ probe)
    p1 = get_train_ratings_ordered('../timeSVDplusplus-master/timeSVD++/timesvd++_resids885.h5',i)
    p1 = np.maximum(np.minimum(p1, 5), 1) # note rounding first improves pred by .006 for 1 model fit

    # SVD++ (w/ probe)
    p2 = get_train_ratings_ordered('../timeSVDplusplus-master/SVD++/svd++_resids892.h5',i)
    p2 = np.maximum(np.minimum(p2, 5), 1)

    # SVD (w/ probe)
    p3 = get_train_ratings_ordered('../timeSVDplusplus-master/SVD/svd_resids898.h5',i)
    p3 = np.maximum(np.minimum(p3, 5), 1)

    # Autorec (w/ probe)
    p4 = get_train_ratings_ordered('../mu/autorec_reduced_ratings_probe.h5',i, True)
    p4 = np.maximum(np.minimum(p4, 5), 1)

    # true svd (w/ probe)
    p5 = get_train_ratings_ordered('../mu/true_svd_keras_probe.h5',i, True)
    p5 = np.maximum(np.minimum(p5, 5), 1)

    # deep svd (w/ probe)
    p6 = get_train_ratings_ordered('../mu/svd_keras_probe.h5',i)
    p6 = np.maximum(np.minimum(p6, 5), 1)

    # deep svd (w/ 300 factors)
    p7 = get_train_ratings_ordered('../mu/svd_reduced_ratings.h5',i)
    p7 = np.maximum(np.minimum(p7, 5), 1)
    # deep svd
    p8 = get_train_ratings_ordered('../mu/svd_reduced_ratings910.h5',i)
    p8 = np.maximum(np.minimum(p8, 5), 1)

    # vae_u-autorec (w/ probe)
    p9 = get_train_ratings_ordered('../mu/vae_uautorec_reduced_ratings_probe.h5',i, False)
    p9 = np.maximum(np.minimum(p9, 5), 1)

    # vae_autorec
    p10 = get_train_ratings_ordered('../mu/vae_autorec_reduced_ratings.h5',i)
    p10 = np.maximum(np.minimum(p10, 5), 1)

    # vae_u-autorec
    p11 = get_train_ratings_ordered('../mu/vae_uautorec_reduced_ratings.h5',i)
    p11 = np.maximum(np.minimum(p11, 5), 1)

    # timeSVD++
    p12 = get_train_ratings_ordered('../timeSVDplusplus-master/timeSVD++/timesvd++_resids893.h5',i)
    p12 = np.maximum(np.minimum(p12, 5), 1) # note rounding first improves pred by .006 for 1 model fit

    # SVD++
    p13 = get_train_ratings_ordered('../timeSVDplusplus-master/SVD++/svd++_resids898.h5',i)
    p13 = np.maximum(np.minimum(p13, 5), 1) # note rounding first improves pred by .006 for 1 model fit

    # SVD
    p14 = get_train_ratings_ordered('../timeSVDplusplus-master/SVD/svd_resids903.h5',i)
    p14 = np.maximum(np.minimum(p14, 5), 1) # note rounding first improves pred by .006 for 1 model fit

    # vae u-autorec no baseline (without probe)
    p15 = get_train_ratings_ordered('../mu/vae_uautorec_reduced_ratings_no_baseline.h5',i)
    p15 = np.maximum(np.minimum(p15, 5), 1)

    # TimeSVD++ (w/ probe) (300 factors)
    p16 = get_train_ratings_ordered('../timeSVDplusplus-master/timeSVD++/timesvd++_resids885_300.h5',i)
    p16 = np.maximum(np.minimum(p16, 5), 1) # note rounding first improves pred by .006 for 1 model fit

    # vae u-autorec no baseline (with probe)
    p17 = get_train_ratings_ordered('../mu/vae_uautorec_reduced_ratings_no_baseline_probe.h5',i, False)
    p17 = np.maximum(np.minimum(p17, 5), 1)

    # u autorec (with probe)
    p18 = get_train_ratings_ordered('../mu/uautorec_reduced_ratings_probe.h5',i, False)
    p18 = np.maximum(np.minimum(p18, 5), 1)


    # movie frequencies
    p19 = num_users[train_probe_items][i:t].reshape((t-i, 1))

    # user frequencies
    p20 = num_movies[train_probe_users][i:t].reshape((t-i, 1))

    # time of rating
    p21 = (np.log(train_probe_times[i:t]+1)/np.log(6.76)).reshape((t-i, 1))

    # timesvd++ baselinemode
    p22 = get_train_ratings_ordered('../mu/timesvd++_baselined.h5',i)
    p22 = np.maximum(np.minimum(p22, 5), 1)

    # RBM (DK) (with probe)
    p23 = np.zeros((t-i, 1))#get_train_ratings_ordered('../mu/rbm_qual.h5',i)
    #p23 = np.maximum(np.minimum(p23, 5), 1)

    # RBM  (DK)
    p24 = np.zeros((t-i, 1))#get_train_ratings_ordered('../mu/rbm_qual_no_probe.h5',i)
    #p24 = np.maximum(np.minimum(p24, 5), 1)

    # autorec no base (with probe)
    p25 = get_train_ratings_ordered('../mu/autorec_no_base.h5',i, False)
    p25 = np.maximum(np.minimum(p25, 5), 1)

    # timesvd++ (without valid)
    p26 = get_train_ratings_ordered('../timeSVDplusplus-master/timeSVD++/timesvd++_resids885_no_valid.h5',i)
    p26 = np.maximum(np.minimum(p26, 5), 1)
    #print p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26
    all_preds = np.hstack((p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26))

    #print 'generating training preds'
    pred = np.dot(all_preds, beta.reshape((num_predictors, 1)))

    pred = np.maximum(np.minimum(pred, 5), 1)
    #print pred
    final[i:t] = pred.reshape(t-i)
bar.finish()
print 'writing'
result = h5py.File('quiz_blended.h5')
result.create_dataset('train_predictions', data=final)

result.close()
'''
