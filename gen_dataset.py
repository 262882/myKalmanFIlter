import torch
import random

def gen_dataset(gt_signal:torch.Tensor, out_name:str, num_train:int = 1000, num_cv:int = 100, num_test:int = 200, Q_mag=0.1):
    
    train_gt = gt_signal.repeat(num_train, 1, 1)
    for cnt, seq in enumerate(train_gt):
        shift = random.randint(0, seq.shape[0])
        train_gt[cnt] = torch.roll(seq, shift,dims=1)
    train_measure = torch.clone(train_gt) + Q_mag*(torch.rand_like(train_gt, dtype=torch.float32)-0.5)

    cv_gt = gt_signal.repeat(num_cv, 1, 1)
    for cnt, seq in enumerate(cv_gt):
        shift = random.randint(0, seq.shape[0])
        cv_gt[cnt] = torch.roll(seq, shift,dims=1)
    cv_measure = torch.clone(cv_gt) + Q_mag*(torch.rand_like(cv_gt, dtype=torch.float32)-0.5)

    test_gt = gt_signal.repeat(num_test, 1, 1)
    for cnt, seq in enumerate(test_gt):
        shift = random.randint(0, seq.shape[0])
        test_gt[cnt] = torch.roll(seq, shift,dims=1)
    test_measure = torch.clone(test_gt) + Q_mag*(torch.rand_like(test_gt, dtype=torch.float32)-0.5)

    torch.save([train_measure, train_gt, cv_measure, cv_gt, test_measure, test_gt], out_name)

if __name__ == "__main__":
    print("Generating data")
    sig_pos = torch.tensor([0.5]*25+[-0.5]*25+[0.5]*25+[-0.5]*25, dtype=torch.float32)
    sig_vel = torch.tensor([0]*100, dtype=torch.float32)
    signal = torch.stack([sig_pos, sig_vel])
    gen_dataset(signal, './datasets/steps.pt')
    print("Completed data generation")
