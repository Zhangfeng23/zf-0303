# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split

data_features_labels = pd.read_csv(r"D:/机器学习大作业/Data_features.csv")
data_features = data_features_labels.iloc[:,:-1].values
data_labels = data_features_labels.iloc[:,-1].values
X_train, X_val, y_train, y_val = train_test_split(data_features, data_labels, random_state=501) # 0501

# 数据准备
train_matrix = lgb.Dataset(X_train, label=y_train)
valid_matrix = lgb.Dataset(X_val, label=y_val)


# 构建lightgbm
params = {
    'learning_rate': 0.1,
    'boosting': 'gbdt', #设置提升类型
    'lambda_l2': 0.1,
    'max_depth': -1,
    'num_leaves': 512, #叶子节点数
    'bagging_fraction': 0.8, #建树的样本采样比例
    'feature_fraction':0.8, #建树的特征选择比例
    'metric': None, #评估函数
    'objective': 'multiclass', #目标函数
    'num_class': 4,
    'nthread': 10,
    'verbose': -1,
}

# 使用lightgbm训练
gbm = lgb.train(params,
          train_set=train_matrix,
          valid_sets=valid_matrix,
          num_boost_round=2000,   # 决策树提升循环次数
          verbose_eval=50,
          early_stopping_rounds=200,
          # feval=f1_score
          )
# 对模型进行保存
gbm.save_model(r"lightGBM_model.txt")
# 模型加载
gbm = lgb.Booster(model_file='lightGBM_model.txt')
# 对验证集进行预测
val_pre_lgb = gbm.predict(X_val, num_iteration=gbm.best_iteration)
preds = np.argmax(val_pre_lgb, axis=1)
score = f1_score(y_true=y_val, y_pred=preds, average='macro')
print('未调参前lightgbm单模型在验证集上的f1：{}'.format(score))


