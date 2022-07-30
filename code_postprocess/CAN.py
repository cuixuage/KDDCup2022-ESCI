"""
    CAN: 多分类F1指标校正
    https://zhuanlan.zhihu.com/p/428123727
    https://github.com/KMnP/can
"""

import pandas as pd
import string
import os
import sys
import numpy as np
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '12345'
    os.execv(sys.executable, [sys.executable] + sys.argv)


# Util 函数
def get_predict_probs(file_path):
    probs_list = []
    with open(file_path, mode='r') as fin:
        for line in fin:
            if 'index' in line: continue       
            tokens = line.strip().split('\t')
            probs = tokens[2].strip().split(',')
            probs = [float(i) for i in probs]
            probs_list.append(probs)
    return np.array(probs_list)

# 1.统计先验分布、模型预测概率值
label2id =  {
    "complement": 0,
    "exact": 1,
    "irrelevant": 2,
    "substitute": 3
  }
prior = [0.031,0.627,0.107,0.232]
prior = np.array(prior)
df_valid = pd.read_csv('./extra_data_with_task1/task2/flod_5_valid.csv')
y_true = np.array([label2id[item] for item in df_valid['esci_label'].to_list()])
y_pred = get_predict_probs('../v0.2_train/output/public_test/confident_learning_task01.valid_text.for_task2')
acc_original = np.mean([y_pred.argmax(1) == y_true])
print('original acc: %s' % acc_original)

# 2.评价每个预测结果的不确定性
k = 2
y_pred_topk = np.sort(y_pred, axis=1)[:, -k:]
print(type(y_pred_topk), y_pred_topk.shape)
y_pred_topk /= y_pred_topk.sum(axis=1, keepdims=True)
y_pred_uncertainty = -(y_pred_topk * np.log(y_pred_topk)).sum(1) / np.log(k)

# 3.选择阈值，划分高、低置信度两部分
threshold = 0.9
y_pred_confident = y_pred[y_pred_uncertainty < threshold]
y_pred_unconfident = y_pred[y_pred_uncertainty >= threshold]
y_true_confident = y_true[y_pred_uncertainty < threshold]
y_true_unconfident = y_true[y_pred_uncertainty >= threshold]

# 显示两部分各自的准确率
# 一般而言，高置信度集准确率会远高于低置信度的
acc_confident = (y_pred_confident.argmax(1) == y_true_confident).mean()
acc_unconfident = (y_pred_unconfident.argmax(1) == y_true_unconfident).mean()
print('confident acc: %s' % acc_confident)
print('unconfident acc: %s' % acc_unconfident)


# 4.逐个修改低置信度样本，并重新评价准确率
right, alpha, iters = 0, 1, 1
for i, y in enumerate(y_pred_unconfident):
    Y = np.concatenate([y_pred_confident, y[None]], axis=0)
    # print(y[None])
    for j in range(iters):
        Y = Y**alpha
        Y /= Y.mean(axis=0, keepdims=True)
        Y *= prior[None]
        Y /= Y.sum(axis=1, keepdims=True)
    y = Y[-1]
    if y.argmax() == y_true_unconfident[i]:
        right += 1
    # print(i, len(y_pred_unconfident))

# 5.输出修正后的准确率
acc_final = (acc_confident * len(y_pred_confident) + right) / len(y_pred)
print('new unconfident acc: %s' % (right / (i + 1.)))
print('final acc: %s' % acc_final)


def update_y_predict(y_pred_uncertainty, y_pred, y_pred_unconfident):
    y_pred_uncertainty = y_pred_uncertainty.to_list()
    y_pred = y_pred.to_list()
    y_pred_unconfident = y_pred_unconfident.to_list()
    assert len(y_pred_uncertainty) == len(y_pred)
    count = 0
    for idx, item in enumerate(y_pred):
        if y_pred_uncertainty[idx] >= threshold:
            item = y_pred_unconfident[count]
            count += 1
            print(item, type(item), y_pred_unconfident[count], type(y_pred_unconfident[count]))
            break
    return y_pred



####  获取Label分布
# train_file = "../v0.2_task2/data/processed/public/task_2_multiclass_product_classification/train-v0.2-with-task1.csv"
# df = pd.read_csv(train_file)

# exact_df = df[df['esci_label'] == 'exact']
# substitute_df = df[df['esci_label'] == 'substitute']
# complement_df = df[df['esci_label'] == 'complement']
# irrelevant_df = df[df['esci_label'] == 'irrelevant']
# print("class=", complement_df.shape, irrelevant_df.shape, substitute_df.shape, exact_df.shape, df.shape)   # 0.031  0.107  0.232 0.627
