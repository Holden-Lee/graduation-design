# -*- coding: utf-8 -*-
"""
Created on Sun Feb 27 10:28:27 2022

@author: LHH
"""
import pandas as pd
import numpy as np

#%% load data

enterprise_123Info = pd.read_excel('./论文数据/附件1：123家有信贷记录企业的相关数据.xlsx',
                                   sheet_name=0, index_col=('企业代号'))
input_123Info = pd.read_excel('./论文数据/附件1：123家有信贷记录企业的相关数据.xlsx',
                              sheet_name=1, index_col=('企业代号'))
output_123Info = pd.read_excel('./论文数据/附件1：123家有信贷记录企业的相关数据.xlsx',
                               sheet_name=2, index_col=('企业代号'))

enterprise_302Info = pd.read_excel('./论文数据/附件2：302家无信贷记录企业的相关数据.xlsx',
                                   sheet_name=0, index_col=('企业代号'))
input_302Info = pd.read_excel('./论文数据/附件2：302家无信贷记录企业的相关数据.xlsx',
                              sheet_name=2, index_col=('企业代号'))
output_302Info = pd.read_excel('./论文数据/附件2：302家无信贷记录企业的相关数据.xlsx',
                               sheet_name=1, index_col=('企业代号'))

interest_loss_corr = pd.read_excel('./论文数据/附件3：银行贷款年利率与客户流失率关系的统计数据.xlsx',
                                   sheet_name=0,names=(['贷款年利率', '信誉评级A', '信誉评级B', '信誉评级C']),index_col=('贷款年利率'),skiprows=([1]))


#%% 预处理1——查找缺失值  无缺失值
input_123Info.info()
output_123Info.info()
enterprise_123Info.info()
input_302Info.info()
output_302Info.info()
enterprise_302Info.info()

#%% 预处理2——改变数据类型
    
input_123Info = input_123Info.replace({'有效发票':True, '作废发票':False})
input_123Info['发票状态'] = input_123Info['发票状态'].astype(bool)

output_123Info = output_123Info.replace({'有效发票':True, '作废发票':False})
output_123Info['发票状态'] = output_123Info['发票状态'].astype(bool)

input_302Info = input_302Info.replace({'有效发票':True, '作废发票':False})
input_302Info['发票状态'] = input_302Info['发票状态'].astype(bool)

output_302Info = output_302Info.replace({'有效发票':True, '作废发票':False})
output_302Info['发票状态'] = output_302Info['发票状态'].astype(bool)

enterprise_123Info = enterprise_123Info.replace({'是':True, '否':False})
enterprise_123Info['是否违约'] = enterprise_123Info['是否违约'].astype(bool)

# input_123Info['开票日期'].format = '%Y/%M/%D'
# output_123Info['开票日期'].format = '%Y/%M/%D'

#%% 数据分析
#1 123家企业分析

index = []
for num in range(1,124):
    enterprise = 'E' + str(num)
    index.append(enterprise)

data123 = pd.DataFrame(index=index, columns=['进项作废发票比例', '销项作废发票比例', '进项发票总数', '销项发票总数', 
                                             '利润', '增值税', '运营时间', '利润绝对数', '利润相对数', 
                                             '进项4年企业数', '进项3年企业数', '进项2年企业数', '进项1年企业数', 
                                             '销项4年企业数', '销项3年企业数', '销项2年企业数', '销项1年企业数', 
                                             '是否为分公司', '是否为公司', '是否为下属部门', '是否为个体户', 
                                             '是否扭亏为盈', '是否转为亏损', '信誉评级','是否违约'
                                             ])
    # 利润未减所得税

'''
time = input_123Info['开票日期'].values
set(pd.to_datetime(time, format='%Y/%M/%D'))
print(time.min(), time.max())
    # 2016/10/4 到 2020/2/21

time = output_123Info['开票日期'].values
set(pd.to_datetime(time).year)
print(time.min(), time.max())
    # 2016/10/7 到 2020/2/21
'''
for num in range(1,124):
    enterprise = 'E' + str(num)
    
    # 企业类型
    name = enterprise_123Info.loc[enterprise, '企业名称']
    
    data123.loc[enterprise, '是否为分公司'] = 0
    data123.loc[enterprise, '是否为公司'] = 0
    data123.loc[enterprise, '是否为下属部门'] = 0
    data123.loc[enterprise, '是否为个体户'] = 0
    
    if name.find('分公司') != -1:
        data123.loc[enterprise, '是否为分公司'] = 1
    elif name.find('个体') != -1:
        data123.loc[enterprise, '是否为个体户'] = 1
    elif name.find('部') != -1:
        data123.loc[enterprise, '是否为下属部门'] = 1
    else: data123.loc[enterprise, '是否为公司'] = 1
    
    # 信誉评级
    data123.loc[enterprise, '信誉评级'] = enterprise_123Info.loc[enterprise, '信誉评级']
    
    # 是否违约
    data123.loc[enterprise, '是否违约'] = enterprise_123Info.loc[enterprise, '是否违约']

# 量化信誉评级
data123['信誉评级'] = data123['信誉评级'].replace({'A':1, 'B':2, 'C':3, 'D':4})
data123['是否违约'] = data123['是否违约'].replace({'True':1, 'False':0})


for num in range(1,124):
    enterprise = 'E' + str(num)
    temp_input = input_123Info.loc[[enterprise]]
    temp_output = output_123Info.loc[[enterprise]]
    
    # 运营时间（天）
    if temp_input['开票日期'][0] > temp_output['开票日期'][0]:
        operate_min = temp_output['开票日期'][0]
    else:
        operate_min = temp_input['开票日期'][0]
    
    if temp_input['开票日期'][len(temp_input)-1] > temp_output['开票日期'][len(temp_output)-1]:
        operate_max = temp_output['开票日期'][len(temp_output)-1]
    else:
        operate_max = temp_input['开票日期'][len(temp_input)-1]

    operate = operate_max - operate_min
    data123.loc[enterprise, '运营时间'] = operate.days
    
    # 作废发票比例（进项）
    invalid = len(temp_input[temp_input['发票状态']==False])
    total = len(temp_input)
    data123.loc[enterprise, '进项作废发票比例'] = invalid / total
    data123.loc[enterprise, '进项发票总数'] = total
    
    # 作废发票比例（销项）
    invalid = len(temp_output[temp_output['发票状态']==False])
    total = len(temp_output)
    data123.loc[enterprise, '销项作废发票比例'] = invalid / total
    data123.loc[enterprise, '销项发票总数'] = total
    
    
    # 删除作废发票
    temp_input = temp_input[temp_input['发票状态']==True]
    temp_output = temp_output[temp_output['发票状态']==True]
    

    # data123.loc[enterprise, '进项5年企业数'] = 0 无交易5年的企业
    ## 进项交易企业数
    data123.loc[enterprise, '进项4年企业数'] = 0
    data123.loc[enterprise, '进项3年企业数'] = 0
    data123.loc[enterprise, '进项2年企业数'] = 0
    data123.loc[enterprise, '进项1年企业数'] = 0
    data123.loc[enterprise, '销项4年企业数'] = 0
    data123.loc[enterprise, '销项3年企业数'] = 0
    data123.loc[enterprise, '销项2年企业数'] = 0
    data123.loc[enterprise, '销项1年企业数'] = 0

    t_in16 = temp_input[(temp_input['开票日期'] >= '2016') & (temp_input['开票日期'] < '2017')]
    t_in17 = temp_input[(temp_input['开票日期'] >= '2017') & (temp_input['开票日期'] < '2018')]
    t_in18 = temp_input[(temp_input['开票日期'] >= '2018') & (temp_input['开票日期'] < '2019')]
    t_in19 = temp_input[(temp_input['开票日期'] >= '2019') & (temp_input['开票日期'] < '2020')]
    t_in20 = temp_input[(temp_input['开票日期'] >= '2020') & (temp_input['开票日期'] < '2021')]

    t_in = set(temp_input.loc[enterprise, '销方单位代号'])    # 此处不能用unique，存在只有一个满足条件的数据——会输出成字符串
    in_enterprise = {x: 0 for x in t_in}
    for x in in_enterprise.keys():
        if not(t_in16[t_in16['销方单位代号'].isin([x])].empty):
            in_enterprise[x] += 1;
        if not(t_in17[t_in17['销方单位代号'].isin([x])].empty):
            in_enterprise[x] += 1;
        if not(t_in18[t_in18['销方单位代号'].isin([x])].empty):
            in_enterprise[x] += 1;
        if not(t_in19[t_in19['销方单位代号'].isin([x])].empty):
            in_enterprise[x] += 1;
        if not(t_in20[t_in20['销方单位代号'].isin([x])].empty):
            in_enterprise[x] += 1;
    
    for i in in_enterprise.values():
        if i == 1:
            data123.loc[enterprise, '进项1年企业数'] += 1
        elif i == 2:
            data123.loc[enterprise, '进项2年企业数'] += 1
        elif i == 3:
            data123.loc[enterprise, '进项3年企业数'] += 1
        elif i == 4:
            data123.loc[enterprise, '进项4年企业数'] += 1
    
    ## 销项交易企业数
    t_out16 = temp_output[(temp_output['开票日期'] >= '2016') & (temp_output['开票日期'] < '2017')]
    t_out17 = temp_output[(temp_output['开票日期'] >= '2017') & (temp_output['开票日期'] < '2018')]
    t_out18 = temp_output[(temp_output['开票日期'] >= '2018') & (temp_output['开票日期'] < '2019')]
    t_out19 = temp_output[(temp_output['开票日期'] >= '2019') & (temp_output['开票日期'] < '2020')]
    t_out20 = temp_output[(temp_output['开票日期'] >= '2020') & (temp_output['开票日期'] < '2021')]

    t_out = set(temp_output.loc[enterprise, '购方单位代号'])
    in_enterprise = {x: 0 for x in t_out}
    for x in in_enterprise.keys():
        if not(t_out16[t_out16['购方单位代号'].isin([x])].empty):
            in_enterprise[x] += 1;
        if not(t_out17[t_out17['购方单位代号'].isin([x])].empty):
            in_enterprise[x] += 1;
        if not(t_out18[t_out18['购方单位代号'].isin([x])].empty):
            in_enterprise[x] += 1;
        if not(t_out19[t_out19['购方单位代号'].isin([x])].empty):
            in_enterprise[x] += 1;
        if not(t_out20[t_out20['购方单位代号'].isin([x])].empty):
            in_enterprise[x] += 1;
    
    for i in in_enterprise.values():
        if i == 1:
            data123.loc[enterprise, '销项1年企业数'] += 1
        elif i == 2:
            data123.loc[enterprise, '销项2年企业数'] += 1
        elif i == 3:
            data123.loc[enterprise, '销项3年企业数'] += 1
        elif i == 4:
            data123.loc[enterprise, '销项4年企业数'] += 1

    
    # 利润=销项金额-进项金额
    ## 总利润
    data123.loc[enterprise, '利润'] = np.sum(temp_output['金额']) - np.sum(temp_input['金额'])

    ## 利润绝对数、相对数 （最后一年 - 第一年）
    
    temp_input = temp_input[(temp_input['开票日期']<'2020')]    # 取2020年之前的进项数据
    temp_output = temp_output[(temp_output['开票日期']<'2020')]
    
    y_in1 = temp_input['开票日期'][0].year
    y_in2 = temp_input['开票日期'][len(temp_input)-1].year
    
    begin_in = temp_input[(temp_input['开票日期']>str(y_in1)) & (temp_input['开票日期']<str(y_in1 + 1))]['金额'].sum()
    end_in = temp_input[(temp_input['开票日期']>str(y_in2)) & (temp_input['开票日期']<str(y_in2 + 1))]['金额'].sum()
    
    y_out1 = temp_output['开票日期'][0].year
    y_out2 = temp_output['开票日期'][len(temp_output)-1].year
    
    begin_out = temp_output[(temp_output['开票日期']>str(y_out1)) & (temp_output['开票日期']<str(y_out1 + 1))]['金额'].sum()
    end_out = temp_output[(temp_output['开票日期']>str(y_out2)) & (temp_output['开票日期']<str(y_out2 + 1))]['金额'].sum()

    absolute = (end_out - end_in) - (begin_out - begin_in)
    relative = (end_out - end_in) / (begin_out - begin_in)
    
    data123.loc[enterprise, '利润绝对数'] = absolute
    data123.loc[enterprise, '利润相对数'] = relative
    
    # 扭亏为盈，转为亏损
    data123.loc[enterprise, '是否扭亏为盈'] = 0
    data123.loc[enterprise, '是否转为亏损'] = 0
    if (relative < 0) & (absolute > 0):
        data123.loc[enterprise, '是否扭亏为盈'] = 1
    elif (relative < 0) & (absolute < 0):
        data123.loc[enterprise, '是否转为亏损'] = 1
    
    # 增值税=应纳税额=销项税额-进项税额  （销项税额抵扣进项税额的余额）    增值税为负置0？
    data123.loc[enterprise, '增值税'] = np.sum(temp_output['税额']) - np.sum(temp_input['税额'])

writer = pd.ExcelWriter('123家企业特征.xlsx')
data123.to_excel(writer, sheet_name='特征')
writer.save()
writer.close()



#%% 特征选择
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2

data = pd.read_excel('123家企业特征.xlsx', index_col='企业代号')
data['是否违约'] = data['是否违约'].astype(int)
dataset = MinMaxScaler().fit_transform(data)

x = dataset[:, 0:-1]
y = dataset[:,-1]

# from sklearn.feature_selection import SelectKBest

chi2 = chi2(x,y)
chi2 = chi2[0]

# model_sk = SelectKBest(score_func=chi2, k='all')
# model_sk.fit(x, y)
# scores_ = model_sk.scores_
# pvalues_ = model_sk.pvalues_

data.drop(data.columns[[4,5,7,8,18,20]], axis=1, inplace=True)

#%% 模型训练

# 改变特征比较最后的结果
import numpy as np
import pandas as pd

x = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

# 随机划分训练集与测试集  2-8分 改进
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
# 训练集
## x_train: 训练集的x(特征)  y_train: 训练集的y(是否违约)
# 测试集
## x_test: 测试集的特征   y_test: 测试集的结果
# 预测集
## estimator.predict(x_train) estimator.predict(x_test)

# 数据集标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
x = scaler.transform(x)
# 对分类器进行参数挑选：GridSearchCV(estimator, param_grid={}, ...)

#%% 初始化
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve


# 模型评估函数
## 二分类混淆矩阵绘制
def plot_confusion_matrix(y_true, y_pred, est_name):

    cm = metrics.confusion_matrix(y_true, y_pred)
    class_name = np.array(["0","1"])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),    # x,y轴刻度位置、标签
           xticklabels=class_name, yticklabels=class_name,
           title='Confusion matrix (' + est_name + ')', xlabel='Predicted label', ylabel='True label')

    # 文本注释
    thresh = cm.sum() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    # plot_confusion_matrix(y, y_predict, 'LR')


def plot_roc_curve(y, y_prob, est_name):
    fpr, tpr, threshold = metrics.roc_curve(y, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots()
    plt.stackplot(fpr, tpr, color = 'steelblue', alpha = 0.5)
    plt.plot(fpr, tpr, color = 'black', lw = 1)
    plt.plot([0,1], [0,1], color = 'red', linestyle = '--')
    plt.text(0.5, 0.3, 'ROC curve (area = %0.3f)' %roc_auc)
    ax.set(title='ROC curve (' + est_name + ')', xlabel='FP rate', ylabel='TP rate')
    plt.show()
    # plot_roc_curve(y, proba1[:,1], 'LR')


## 预测概率校准
def plot_calibration(y, proba, estimator0):   # 校准曲线图
    # y: 实际违约情况   proba: 预测违约情况
    fig, ax = plt.subplots()
    predict_p, actual_p = calibration_curve(y, proba)
    ax.plot([0, 1], [0, 1], 'k:', label='Perfectly calibrated')
    ax.plot(predict_p, actual_p, 'r', label=str(estimator0))
    ax.set(title='Calibration curve', xlabel='Prediction probability', ylabel='Actual probability')
    ax.legend()
    plt.show()
    # plot_calibration(y, proba[:,1])


# platt scaling
def plot_calibration_curve(estimator, name, fig_index):
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(estimator, method='isotonic') # 校准要求用于拟合分类器的数据不能与用于拟合回归器的数据相交，此处没有新的数据，不能用cv='prefit'

    # Calibrated with sigmoid calibration
    sigmoid = CalibratedClassifierCV(estimator, method='sigmoid')

    # LR = LogisticRegression(C=0.09, penalty='l1', solver='liblinear')

    plt.figure(1, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)


    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(estimator, name),
                      (isotonic, name + ' + Isotonic'),
                      (sigmoid, name + ' + Sigmoid')]:
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        prob_pos = clf.predict_proba(x_test)[:, 1]

        clf_score = metrics.brier_score_loss(y_test, prob_pos, pos_label=y.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % metrics.precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % metrics.recall_score(y_test, y_pred))
        print("\tF1: %1.3f" % metrics.f1_score(y_test, y_pred))
        print("\tAUC: %1.3f\n" % metrics.roc_auc_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = calibration_curve(y_test, prob_pos)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="%s (%1.3f)" % (name, clf_score))
        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name, histtype="step", lw=2)

    ax1.set_ylim([-0.05, 1.05])
    ax1.set_title('Calibration curve (reliability curve)', fontsize=20)
    ax1.set_ylabel('Fraction of positives', fontsize=18)
    ax1.legend(loc='upper left', fontsize=15)

    ax2.set_xlabel('Mean predicted value', fontsize=18)
    ax2.set_ylabel('Count', fontsize=20)
    ax2.legend(loc="upper center", ncol=2, fontsize=15)

    plt.tight_layout()
    plt.show()
    # plot_calibration_curve(Softvote, "SoftVoting", 1)
    

## 输出分类器挑选、评估结果
def estimate_print(estimator, ytest_predict, ytrain_predict):
    # result = estimator.cv_results_
    # metrics.brier_score_loss(y,proba[:,1])
    print('最优的参数组合：\n', estimator.best_params_)
    print('最优模型的得分：\n', estimator.best_score_)
    print('最佳估计器：\n', estimator.best_estimator_)
    print('分类模型评估报告：\n', metrics.classification_report(y_test, ytest_predict))
    print('训练集准确率为：\n', metrics.accuracy_score(y_train, ytrain_predict))
    print('测试集准确率为：\n', metrics.accuracy_score(y_test, ytest_predict))
    print('ROC面积：\n', metrics.roc_auc_score(y, estimator.predict(x)))
    

#%% logistic regression
estimator_name = 'Logistic Regression'
param = {'penalty':['l2', 'l1', 'elasticnet'], 
         'C':[i * 0.01 for i in range(1,101)], 
         'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
         'max_iter':[100, 200, 300], 
        }
estimator = LogisticRegression()
estimator = GridSearchCV(estimator, param_grid=param, scoring='accuracy') # 以scoring为目标挑选参数
estimator.fit(x_train, y_train)     # 训练模型 Fit the model according to the given training data.

# LogisticRegression(C=0.09, penalty='l1', solver='liblinear')
# estimator.fit(x_train, y_train)
ytrain_predict = estimator.predict(x_train)
ytest_predict = estimator.predict(x_test)
y_predict = estimator.predict(x)

estimate_print(estimator, ytest_predict, ytrain_predict)
## best_params_: {'C': 0.09, 'max_iter': 100, 'penalty': 'l1', 'solver': 'liblinear'} scoring = accuracy

proba1 = estimator.predict_proba(x) # 各企业出现违约的可能性

# 单一模型评估
plot_confusion_matrix(y, y_predict)
plot_calibration(y, proba1[:,1], estimator_name)

#%% SVC
estimator_name = 'SVC'
param = {'kernel':['poly','linear','rbf','sigmoid'],
         'C':[i*0.1 for i in range(1,11)],
         'degree':[3,4,5],
         'gamma':['scale','auto']
        }

estimator = SVC(random_state=42, probability=True)
estimator = GridSearchCV(estimator, param_grid=param, scoring='accuracy')
estimator.fit(x_train,y_train)

# SVC(C=0.2, kernel='linear', probability=True, random_state=42)
# estimator.fit(x_train, y_train)
ytrain_predict = estimator.predict(x_train)
ytest_predict = estimator.predict(x_test)
y_predict = estimator.predict(x)

estimate_print(estimator, ytest_predict, ytrain_predict)
## best_params_: {'C': 0.2, 'degree': 3, 'gamma': 'scale', 'kernel': 'linear'} scoring = accuracy
proba2 = estimator.predict_proba(x)

# 单一模型评估
plot_confusion_matrix(y, y_predict)
plot_calibration(y, proba2[:,1], estimator_name)

#%% AdaBoost
estimator_name = 'AdaBoost'
param = {'n_estimators':[ i * 100 for i in range(1,6)],
         'learning_rate':[ i * 0.1 for i in range(1,11)],
         'algorithm':['SAMME','SAMME.R'], 
         }

estimator = AdaBoostClassifier(random_state=42)
estimator = GridSearchCV(estimator, param_grid=param, scoring='accuracy')

estimator.fit(x_train,y_train)

# AdaBoostClassifier(algorithm='SAMME', learning_rate=0.1, n_estimators=100, random_state=42)
# estimator.fit(x_train, y_train)
ytrain_predict = estimator.predict(x_train)
ytest_predict = estimator.predict(x_test)
y_predict = estimator.predict(x)

estimate_print(estimator, ytest_predict, ytrain_predict)
## best_params_: {'algorithm': 'SAMME', 'learning_rate': 0.1, 'n_estimators': 100} scoring = accuracy
proba3 = estimator.predict_proba(x)

# 单一模型评估
plot_confusion_matrix(y, y_predict)
plot_calibration(y, proba3[:,1], estimator_name)

#%% GBDT
estimator_name = 'GBDT'
param = {'loss':['exponential'],
         'learning_rate':[ i * 0.1 for i in range(1,11)],
         'n_estimators':[ i * 100 for i in range(1,11)],
         'max_features':['auto'],
         'max_depth':[ i * 1 for i in range(1,11)],
        }
# loss='deviance' == logistic regression

estimator = GradientBoostingClassifier(random_state=42)
estimator = GridSearchCV(estimator, param_grid=param, scoring='accuracy')
estimator.fit(x_train,y_train)

# GradientBoostingClassifier(loss='exponential', max_depth=1, max_features='auto', random_state=42)
# estimator.fit(x_train, y_train)
ytrain_predict = estimator.predict(x_train)
ytest_predict = estimator.predict(x_test)
y_predict = estimator.predict(x)

estimate_print(estimator, ytest_predict, ytrain_predict)
# best_params_: {'learning_rate': 0.1, 'loss': 'exponential', 'max_depth': 1, 'max_features': 'auto', 'n_estimators': 100} scoring = accuracy

proba4 = estimator.predict_proba(x)

# 单一模型评估
plot_confusion_matrix(y, y_predict)
plot_calibration(y, proba4[:,1], estimator_name)


#%% RandomForest
estimator_name = 'Random Forest'
param = {'n_estimators':[100, 200, 300],
        'criterion':['gini', 'entropy'],
         'min_samples_split':[2,4,6,8],
         'max_features':['auto', 'sqrt', 'log2']
        }

estimator = RandomForestClassifier(random_state=42)
estimator = GridSearchCV(estimator, param_grid=param, scoring='accuracy')
estimator.fit(x_train,y_train)

# RandomForestClassifier(min_samples_split=6, random_state=42)
# estimator.fit(x_train, y_train)
ytrain_predict = estimator.predict(x_train)
ytest_predict = estimator.predict(x_test)
y_predict = estimator.predict(x)

estimate_print(estimator, ytest_predict, ytrain_predict)
# {'criterion': 'gini', 'max_features': 'auto', 'min_samples_split': 6, 'n_estimators': 100}
proba5 = estimator.predict_proba(x)

# 单一模型评估
plot_confusion_matrix(y, y_predict)
plot_calibration(y, proba5[:,1], estimator_name)


#%% Soft Voting
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
weight = []

LR = LogisticRegression(C=0.09, penalty='l1', solver='liblinear')
SVC = SVC(C=0.2, kernel='linear', probability=True, random_state=42)
AB = AdaBoostClassifier(algorithm='SAMME', learning_rate=0.1, n_estimators=100, random_state=42)
GBDT = GradientBoostingClassifier(learning_rate=0.5, loss='exponential', max_features='auto', random_state=42)
RF = RandomForestClassifier(min_samples_split=6, random_state=42)


for method, label in zip([LR, SVC, AB, GBDT, RF], ['Logistic Regression', 'SVC', 'AdaBoost', 'GBDT', 'Random Forest']):
    method.fit(x_train, y_train)
    weight.append(metrics.brier_score_loss(y, method.predict(x)))
    # y_predict = method.predict(x)
    # proba_ = method.predict_proba(x)
    # print(label + ':')
    # print('精确度：', metrics.precision_score(y, method.predict(x)))
    # print('召回率：', metrics.recall_score(y, method.predict(x)))
    # print('F1-score：', metrics.f1_score(y, method.predict(x)))
    # print('AUC：', metrics.roc_auc_score(y, method.predict(x)))
    # plot_confusion_matrix(y, y_predict, label)
    # plot_roc_curve(y, proba_[:,1], label)
    

Softvote = VotingClassifier(estimators=[('Logistic Regression',LR),('SVC',SVC),('AdaBoost',AB),('GBDT',GBDT),('Random Forest',RF)], voting='soft', weights=weight)
Softvote.fit(x_train,y_train)

ytrain_predict = Softvote.predict(x_train)
ytest_predict = Softvote.predict(x_test)
y_predict = Softvote.predict(x)

print('Soft Voting: ')
print('训练集准确率为：\n', metrics.accuracy_score(y_train, ytrain_predict))
print('预测集准确率为：\n', metrics.accuracy_score(y_test, ytest_predict))
print('综合准确度为：\n', metrics.accuracy_score(y, Softvote.predict(x)))

proba = Softvote.predict_proba(x)

# 单一模型评估
plot_confusion_matrix(y, y_predict, 'Soft Voting')
plot_roc_curve(y, proba[:,1], 'Soft Voting')
plot_calibration_curve(Softvote, 'Soft Voting', 1)

calibrated_Softvote = CalibratedClassifierCV(Softvote, method='sigmoid').fit(x, y) # 或 CalibratedClassifierCV(Softvote, cv=2, method='isotonic')

prob = calibrated_Softvote.predict_proba(x)

# plot_calibration(y, proba[:,1], estimator)

#%% 违约概率
default_p=pd.DataFrame(index=enterprise_123Info.index, columns=['违约概率','信誉评级'])
default_p['违约概率']=prob[:,1]
default_p['信誉评级']=enterprise_123Info['信誉评级']

writer = pd.ExcelWriter('违约概率.xlsx')
default_p.to_excel(writer)
writer.save()
writer.close()

#%% 描述性统计

data = pd.read_excel('违约概率.xlsx', index_col='企业代号')

A = data[data['信誉评级']=='A']['违约概率'].describe()
B = data[data['信誉评级']=='B']['违约概率'].describe()
C = data[data['信誉评级']=='C']['违约概率'].describe()
D = data[data['信誉评级']=='D']['违约概率'].describe()

result = pd.DataFrame(data={'A':A,'B':B,'C':C,'D':D}).T
writer = pd.ExcelWriter('描述性统计.xlsx')
result.to_excel(writer)
writer.save()
writer.close()












