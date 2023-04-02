import torch
import random
import numpy as np

def gen_dataset(gt_signal:torch.Tensor, out_name:str, num_train:int = 1000, num_cv:int = 100, num_test:int = 200, Q_mag=0.5, DC_bias=0):

    train_gt = gt_signal.repeat(num_train, 1, 1)
    for cnt, seq in enumerate(train_gt):
        shift = random.randint(0, seq.shape[1])
        #train_gt[cnt] = torch.roll(seq, shift,dims=1)
        train_gt[cnt][0] = train_gt[cnt][0] + DC_bias*2*(random.random()-0.5)
    train_measure = torch.clone(train_gt) + Q_mag*(torch.rand_like(train_gt, dtype=torch.float32)-0.5)

    cv_gt = gt_signal.repeat(num_cv, 1, 1)
    for cnt, seq in enumerate(cv_gt):
        shift = random.randint(0, seq.shape[1])
        #cv_gt[cnt] = torch.roll(seq, shift,dims=1)
        cv_gt[cnt][0] = cv_gt[cnt][0] + DC_bias*2*(random.random()-0.5)
    cv_measure = torch.clone(cv_gt) + Q_mag*(torch.rand_like(cv_gt, dtype=torch.float32)-0.5)

    test_gt = gt_signal.repeat(num_test, 1, 1)
    for cnt, seq in enumerate(test_gt):
        shift = random.randint(0, seq.shape[1])
        #test_gt[cnt] = torch.roll(seq, shift,dims=1)
        test_gt[cnt][0] = test_gt[cnt][0] + DC_bias*2*(random.random()-0.5)
    test_measure = torch.clone(test_gt) + Q_mag*(torch.rand_like(test_gt, dtype=torch.float32)-0.5)

    torch.save([train_measure, train_gt, cv_measure, cv_gt, test_measure, test_gt], out_name)

def f_parabolic_step(X_prev):
    delta_t = 1
    acc = -10
    x_prev, v_prev = X_prev
    
    v_curr = v_prev + acc*delta_t
    x_curr = x_prev + v_curr*delta_t + 0.5*acc*delta_t**2
    
    return np.array([x_curr,v_curr])

if __name__ == "__main__":
    print("Generating data")
    #step_pos = torch.tensor([0.5]*50+[-0.5]*50, dtype=torch.float32)
    #step_vel = torch.tensor([0]*len(step_pos), dtype=torch.float32)
    #step = torch.stack([step_pos, step_vel])
    #gen_dataset(step, './datasets/steps.pt')

    time_finish = 99 #seconds
    para_steps = np.arange(time_finish+1)
    para_gt = np.zeros([para_steps.shape[0], 2])
    para_prev = np.array([para_gt[0][0],50*5])

    for step in para_steps[1:]:
        para_curr = f_parabolic_step(para_prev)
        if para_curr[0] < 0:
            para_curr[0] = -para_curr[0]
            para_curr[1] = -para_curr[1]*0.8
        
        para_gt[step] = para_curr
        para_prev = np.copy(para_curr)
    para = torch.as_tensor(para_gt.T, dtype=torch.float32)
        
    gen_dataset(para, './datasets/para.pt', Q_mag=1000, DC_bias=2500)
    print("Completed data generation")
