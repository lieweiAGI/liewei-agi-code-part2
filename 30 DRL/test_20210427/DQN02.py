import torch, gym, random
from torch import nn, optim


class QNet(nn.Sequential):

    def __init__(self):
        super().__init__(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )


class Game:

    def __init__(self, exp_pool_size, explore=0.9):
        self.env = gym.make('MountainCar-v0')

        self.exp_pool = []
        self.exp_pool_size = exp_pool_size

        self.q_net = QNet().cuda()

        self.explore = explore

        self.loss_fn = nn.MSELoss()

        self.opt = optim.Adam(self.q_net.parameters())

    def __call__(self):
        is_render = False
        avg = 0
        while True:
            # 数据采样
            state = self.env.reset()
            R = 0
            while True:
                if is_render: self.env.render()

                if len(self.exp_pool) >= self.exp_pool_size:
                    self.exp_pool.pop(0)  # 把旧的经验删除掉
                    self.explore += 0.00001
                    if random.random() > self.explore:
                        action = self.env.action_space.sample()
                    else:
                        _state = torch.tensor(state).float().cuda()
                        Qs = self.q_net(_state[None, ...])
                        action = Qs.argmax(dim=1)[0].item()

                else:  # 经验池没满就随机采样
                    action = self.env.action_space.sample()

                next_state, reward, done, _ = self.env.step(action)
                position,velocity = next_state
                #车开的越高，reward就越大
                reward = abs(position+0.5)
                R += reward
                self.exp_pool.append([state, reward, action, next_state, done])
                state = next_state
                if done:
                    avg = 0.95 * avg + 0.05 * R
                    print(avg, R, self.env.spec.reward_threshold)
                    if avg > 30:
                        is_render = True
                    break

            # 训练
            if len(self.exp_pool) >= self.exp_pool_size:
                exps = random.choices(self.exp_pool, k=10000)

                _state = torch.tensor([exp[0].tolist() for exp in exps]).cuda()
                _reward = torch.tensor([[exp[1]] for exp in exps]).cuda()
                _action = torch.tensor([[exp[2]] for exp in exps]).cuda()
                _next_state = torch.tensor([exp[3].tolist() for exp in exps]).cuda()
                _done = torch.tensor([[int(exp[4])] for exp in exps]).cuda()

                # 得到估计值
                _Qs = self.q_net(_state)
                _Q = torch.gather(_Qs, 1, _action)

                # 目标值
                _next_Qs = self.q_net(_next_state)
                _max_Q = _next_Qs.max(dim=1, keepdim=True)[0]
                _target_Q = _reward + (1 - _done) * 0.9 * _max_Q
                _target_Q = _target_Q.float()

                loss = self.loss_fn(_Q, _target_Q.detach())
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()


if __name__ == '__main__':
    game = Game(100000)
    game()
