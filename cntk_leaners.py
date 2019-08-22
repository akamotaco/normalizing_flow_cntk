import cntk as C
import numpy as np

from cntk.learners import UserLearner

class MySgd(UserLearner):
    def __init__(self, parameters, lr_schedule):
        super(MySgd, self).__init__(parameters, lr_schedule)

    def update(self, gradient_values, training_sample_count, sweep_end):
        eta = self.learning_rate() / training_sample_count
        for p, g in gradient_values.items():
            new_p = p - eta * C.constant(g)
            p.set_value(new_p.eval(as_numpy=False).data)
        return True

# https://github.com/LiyuanLucasLiu/RAdam/blob/master/radam.py

class RAdam(UserLearner):
    def __init__(self, parameters, lr_schedule, beta1: float, beta2: float, weight_decay: float = 0):
        super(RAdam, self).__init__(parameters, lr_schedule)
        self.state = {}
        self.betas = (C.Constant(beta1), C.Constant(beta2))
        self.buffer = [[None, None, None] for ind in range(10)]
        self.eps = C.Constant(1e-12)

    def update(self, gradient_values, training_sample_count, sweep_end):
        lr = self.learning_rate()
        for p, g in gradient_values.items():
            grad = C.Constant(g)
            p = p

            if p not in self.state:
                self.state[p] = {}
            
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = C.Constant(C.zeros_like(p).eval())
                state['exp_avg_sq'] = C.Constant(C.zeros_like(p).eval())
            else:
                state['exp_avg'] = state['exp_avg'] # .type_as(p)
                state['exp_avg_sq'] = state['exp_avg_sq'] # .type_as(p)

            exp_avg,  exp_avg_sq = state['exp_avg'],  state['exp_avg_sq']
            beta1, beta2 = self.betas

            exp_avg_sq.set_value( (exp_avg_sq*beta2 + (1-beta2)*grad*grad).eval(as_numpy=False).data )
            exp_avg.set_value( (exp_avg*beta1 + ((1-beta1)+grad)).eval(as_numpy=False).data )

            state['step'] += 1
            buffered = self.buffer[int(state['step']%10)]
            if state['step'] == buffered[0]:
                N_sma, step_size =  buffered[1], buffered[2]
            else:
                buffered[0] = state['step']
                beta2_t = C.pow(beta2, state['step'])
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = (N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)).eval()
                buffered[1] =  N_sma

                if N_sma >= 5:
                    step_size = lr * C.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - C.pow(beta1, state['step']))
                else:
                    step_size = lr / (1 - C.pow(beta1, state['step']))
                buffered[2] = step_size.eval()

            # if group['weight_decay'] != 0:
            #     p.add_(-group['weight_decay'] * group['lr'], p)

            if N_sma >= 5:
                denom = C.sqrt(exp_avg_sq) + self.eps
                new_p = p + (-step_size*(exp_avg/denom))
            else:
                new_p = p + (-step_size + exp_avg)
            
            p.set_value(new_p.eval(as_numpy=False).data)
        return True

def my_adagrad(parameters, gradients):
    accumulators = [C.constant(0, shape=p.shape, dtype=p.dtype, name='accum') for p in parameters]
    update_funcs = []
    for p, g, a in zip(parameters, gradients, accumulators):
        accum_new = C.assign(a, g * g)
        update_funcs.append(C.assign(p, p - 0.01 * g / C.sqrt(accum_new + 1e-6)))
    return C.combine(update_funcs)

a = C.input_variable(1)
b = C.layers.Dense(2)(a)
loss = C.reduce_mean(C.square(b))

# trainer = C.Trainer(b, (loss, None), C.universal(my_adagrad, b.parameters))
# trainer = C.Trainer(b, (loss, None), MySgd(b.parameters, C.learning_parameter_schedule(0.01)))
trainer = C.Trainer(b, (loss, None), RAdam(b.parameters, C.learning_parameter_schedule(0.01), 0.1, 0.9))
# trainer = C.Trainer(b, (loss, None), C.sgd(b.parameters, C.learning_parameter_schedule(0.01)))

trainer.train_minibatch({a:np.array([[1.0]])})
