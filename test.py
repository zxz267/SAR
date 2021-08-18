import torch
from tqdm import tqdm
from base import Tester

def main():
    torch.backends.cudnn.benchmark = True
    tester = Tester()
    tester._make_model()
    tester._make_batch_loader()
    eval_result = {}
    cur_sample_idx = 0
    for iteration, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_loader)):
        with torch.no_grad():
            outs = tester.model(inputs)

        outs = {k: v.cpu().numpy() for k, v in outs.items()}
        meta_info = {k: v.cpu().numpy() for k, v in meta_info.items()}

        # evaluate
        cur_eval_result = tester._evaluate(outs, meta_info, cur_sample_idx)
        for k, v in cur_eval_result.items():
            if k in eval_result:
                eval_result[k] += v
            else:
                eval_result[k] = v
        cur_sample_idx += inputs['img'].shape[0]
    tester._print_eval_result(eval_result)

if __name__ == '__main__':
    main()



