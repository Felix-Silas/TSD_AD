import numpy as np


def dtw_reconstruction_error(x, x_):
    n, m = x.shape[0], x_.shape[0]

    # 경계 조건을 처리하기 위해 1씩 더 큰 크기로 초기화
    dtw_matrix = np.zeros((n + 1, m + 1))

    # inf로 초기화하여 DTW 경로 계산에서 값이 비정상적으로 작아지는 문제 방지
    for i in range(n + 1):
        for j in range(m + 1):
            dtw_matrix[i, j] = np.inf
    
    # 초기 값 설정
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # 현재 시점의 절대 차이를 cost로 계산
            cost = abs(x[i - 1] - x_[j - 1])
            # 이전 최소 비용 계산산
            last_min = np.min([dtw_matrix[i - 1, j],
                               dtw_matrix[i, j -1],
                               dtw_matrix[i - 1, j - 1]])
            # 누적 비용 저장
            dtw_matrix[i, j] = cost + last_min

    # 두 시계열 간의 최적 경로를 따라 계산된 총 비용이 최종 결과값
    return dtw_matrix[n][m]


def detect_anomaly(anomaly_score):
    window_size = len(anomaly_score) // 3
    step_size = len(anomaly_score) // (3 + 10)

    is_anomaly = np.zeros(len(anomaly_score))

    for i in range(0, len(anomaly_score) - window_size, step_size):
        window_elts = anomaly_score[i:i + window_size]
        window_mean = np.mean(window_elts)
        window_std = np.std(window_elts)

        for j, elt in enumerate(window_elts):
            if (window_mean - 4 * window_std) < elt < (window_mean + 4 * window_std):
                is_anomaly[i + j] = 0
            else:
                is_anomaly[i + j] = 1

    return is_anomaly


def prune_false_positive(is_anomaly, anomaly_score, change_threshold):
    seq_details = []
    delete_sequence = 0
    start_position = 0
    end_position = 0
    max_seq_element = anomaly_score[0]

    for i in range(1, len(is_anomaly)):
        if i+1 == len(is_anomaly): # 마지막 인덱스에 도달했을 때
            seq_details.append([start_position, i, max_seq_element, delete_sequence])
        elif is_anomaly[i] == 1 and is_anomaly[i+1] == 0: # 이상 시퀀스가 종료되었을 때
            end_position = i
            seq_details.append([start_position, end_position, max_seq_element, delete_sequence])
        elif is_anomaly[i] == 1 and is_anomaly[i-1] == 0: # 이상 시퀀스가 시작되었을 때
            start_position = i
            max_seq_element = anomaly_score[i]
        if is_anomaly[i] == 1 and is_anomaly[i-1] == 1 and anomaly_score[i] > max_seq_element: # 이상 시퀀스 내에 가장 높은 이상 점수 추적
            max_seq_element = anomaly_score[i]

    max_elements = list()
    for i in range(0, len(seq_details)):
        max_elements.append(seq_details[i][2])

    max_elements.sort(reverse=True)
    max_elements = np.array(max_elements)
    change_percent = abs(max_elements[1:] - max_elements[:-1]) / max_elements[1:]

    #Appending 0 for the 1 st element which is not change percent
    delete_seq = np.append(np.array([0]), change_percent < change_threshold)

    #Mapping max element and seq details
    for i, max_elt in enumerate(max_elements):
        for j in range(0, len(seq_details)):
            if seq_details[j][2] == max_elt:
                seq_details[j][3] = delete_seq[i]

    for seq in seq_details:
        if seq[3] == 1: #Delete sequence
            is_anomaly[seq[0]:seq[1]+1] = [0] * (seq[1] - seq[0] + 1)
 
    return is_anomaly

def find_scores(y_true, y_predict):
    tp = tn = fp = fn = 0

    for i in range(0, len(y_true)):
        if y_true[i] == 1 and y_predict[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_predict[i] == 0:
            fn += 1
        elif y_true[i] == 0 and y_predict[i] == 0:
            tn += 1
        elif y_true[i] == 0 and y_predict[i] == 1:
            fp += 1

    print('TP:', tp, ' TN:', tn, ' FP:', fp, ' FN:', fn)
    print ('Accuracy {:.2f}'.format((tp + tn)/(len(y_true))))

    
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    print('Recall {:.2f}'.format(recall))

    # F1 Score 계산 (분모가 0인 경우 처리)
    if precision + recall == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    print('F1 Score {:.2f}'.format(f1_score))
