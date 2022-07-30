"""
    2022.06.13 
    依赖cleanlab工具包，通过交叉验证数据、真实标签计算得到"可能标注错误"的数据
"""
from cleanlab.filter import find_label_issues
from cleanlab.rank import get_label_quality_scores
import pandas as pd
import string
import os
import sys
import numpy as np
hashseed = os.getenv('PYTHONHASHSEED')
if not hashseed:
    os.environ['PYTHONHASHSEED'] = '12345'
    os.execv(sys.executable, [sys.executable] + sys.argv)

label_to_id= {'complement': 0, 'exact': 1, 'irrelevant': 2, 'substitute': 3}
# label_dict = {'0':[1,0,0,0], '1':[0,1,0,0], '2':[0,0,1,0], '3':[0,0,0,1]}

def get_one_hot_labels(file_name):
    df = pd.read_csv(file_name)
    df = df.reset_index()
    one_hot_labels = []
    for index, row in df.iterrows():
        int_label = int(label_to_id[row['esci_label']])
        one_hot_labels.append(int_label)
    """
        batch_size * 1
    """
    return one_hot_labels

def get_probs(file_name):
    probs= []
    with open(file_name, mode='r') as fin:
        for line in fin:
            if 'index' in line or 'probs' in line:
                continue
            else:
                item = [float(i) for i in line.strip().split('\t')[-1].split(',') ]
                probs.append(item)
    """
        batch_size * num_classes
    """
    return probs

task01_labels = get_one_hot_labels('./extra_confident_learnging/flod_01_valid.csv')
task01_probs = get_probs('./extra_confident_learnging/confident_learning_task01.valid_text.for_task2')
task23_labels = get_one_hot_labels('./extra_confident_learnging/flod_23_valid.csv')
task23_probs = get_probs('./extra_confident_learnging/confident_learning_task23.valid_text.for_task2')
task45_labels = get_one_hot_labels('./extra_confident_learnging/flod_45_valid.csv')
task45_probs = get_probs('./extra_confident_learnging/confident_learning_task45.valid_text.for_task2')
task67_labels = get_one_hot_labels('./extra_confident_learnging/flod_67_valid.csv')
task67_probs = get_probs('./extra_confident_learnging/confident_learning_task67.valid_text.for_task2')
task89_labels = get_one_hot_labels('./extra_confident_learnging/flod_89_valid.csv')
task89_probs = get_probs('./extra_confident_learnging/confident_learning_task89.valid_text.for_task2')

labels = task01_labels + task23_labels + task45_labels + task67_labels + task89_labels
probs = task01_probs + task23_probs + task45_probs + task67_probs + task89_probs

labels_np = np.array(labels)
probs_np = np.array(probs)
print("init=", labels_np.shape , probs_np.shape)

# 1.获得"可能是标注错误"的样本
label_issue_file = './extra_confident_learnging/label_issue_index.csv'
ordered_label_issues = find_label_issues(
    labels=labels_np,
    pred_probs=probs_np,
    filter_by='prune_by_noise_rate',
    return_indices_ranked_by='normalized_margin',
)

print('output=', type(ordered_label_issues), ordered_label_issues.shape)
split_len = int(2069039 * 0.04)     # 82761
with open(label_issue_file, mode='w') as fout:
    fout.write('label_issue_index' + '\n')
    for idx, item in enumerate(ordered_label_issues.tolist()[:split_len]):
        fout.write(str(item) + '\n')


# 2.给Label计算置信度， 范围是由0到1
output_weight_file = './extra_confident_learnging/soft_weight_label.csv'
label_quality_scores = get_label_quality_scores(
    labels=labels_np,
    pred_probs=probs_np,
    method='normalized_margin'
)
print('output=', type(label_quality_scores), label_quality_scores.shape)
print(label_quality_scores[:100])
with open(output_weight_file, mode='w') as fout:
    fout.write('confident_learning_soft_weight' + '\n')
    for i, item in enumerate (label_quality_scores.tolist()):
        fout.write(str(item) + '\n')

