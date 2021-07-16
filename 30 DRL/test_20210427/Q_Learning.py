import numpy as np
import random
#价值矩阵
Q = np.zeros((6,6))
#回报矩阵
R = np.array([[-1,-1,-1,-1,0,-1],
              [-1,-1,-1,0,-1,100],
              [-1,-1,-1,0,-1,-1],
              [-1,0,0,-1,0,-1],
              [0,-1,-1,0,-1,100],
              [-1,0,-1,-1,0,100]])
#学习过程
for i in range(2000):
    #对每个轮次的训练，选择一个随机的状态
    state = random.randint(0,5)
    while True:
        #选择当前状态下的所有可能的动作
        r_pos_action = []
        for action in range(6):
            if R[state,action] >=0:
                r_pos_action.append(action)
        #下一时刻的状态
        next_state = r_pos_action[random.randint(0,len(r_pos_action)-1)]
        #代入Q_Learning算法
        Q[state,next_state] = R[state,next_state] + 0.8 *(Q[next_state]).max()
        state = next_state
        #如果走出房间，游戏终止
        if state == 5:
            break
print(Q/5)