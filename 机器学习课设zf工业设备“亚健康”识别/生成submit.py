import pandas as pd
import pandas as pd
from sklearn.metrics import f1_score
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
# 加载模型
import joblib
model = lgb.Booster(model_file="lightGBM_model.txt")

X_test_features = pd.read_csv(r"D:/机器学习大作业/X_test_features.csv")

test_pre_lgb = model.predict(X_test_features, num_iteration=model.best_iteration)
preds = np.argmax(test_pre_lgb, axis=1)

# 生成submint.csv文件
submit = pd.read_csv(r"D:/机器学习大作业/数据集/submit_sample.csv")
submit['label'] = preds
submit.to_csv(r"D:机器学习大作业/submit.csv", index=False)