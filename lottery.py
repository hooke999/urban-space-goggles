import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

def predict(dummy_arg):
    # 讀取數據，轉換為狀態序列
    data = pd.read_excel('data.xlsx', index_col='期數')
    state_seq = data.iloc[:, :-1].values.flatten().tolist()

    # 定義狀態空間
    states = np.array(list(range(1, 50)))
    states = np.delete(states, np.where(states == 0))

    # 計算初始機率分佈
    init_dist = np.array([state_seq.count(s) for s in states]) / len(state_seq)

    # 計算轉移矩陣
    trans_mat = np.zeros((49, 49))
    for i in range(len(state_seq)-1):
        s1, s2 = state_seq[i], state_seq[i+1]
        trans_mat[s1-1][s2-1] += 1
    trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=True)
    trans_mat = np.delete(trans_mat, np.where(states == 0), axis=0)
    trans_mat = np.delete(trans_mat, np.where(states == 0), axis=1)

    # 計算機率分佈
    prob_dist = init_dist
    for i in range(7):
        prob_dist = prob_dist @ trans_mat
    # 找到機率最高的7個號碼
    top_7 = np.argsort(prob_dist)[-7:]
    top_7 = states[top_7]
    return top_7

if __name__ == '__main__':
    mp.freeze_support()
    pool = mp.Pool()
    num_predictions = 1
    results = list(tqdm(pool.imap(predict, [None] * num_predictions), total=num_predictions))
    pool.close()
    pool.join()

    # 找到機率最高的7個號碼
    top_7 = results[0]
    print(f"Predicted 7 numbers: {top_7}")
