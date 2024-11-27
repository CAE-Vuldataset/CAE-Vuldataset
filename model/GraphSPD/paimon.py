from zxh_args import get_args
from src.zxh_train import Data_Prep, ValData_Prep, Train, Test
import time

if __name__ == '__main__':
    # dataset = 'PatchDB/uniqueness/0.7'
    dataset = 'Detect0day/data'
    opt = get_args(dataset)
    print(opt.conf_path)
    
    print(dataset)
    time.sleep(5)

    opt.task = 'train'
    Data_Prep(opt)
    ValData_Prep(opt)
    Train(opt)
    opt.task = 'test'
    Data_Prep(opt)
    Test(opt)

    # if opt.task == 'train':
    #     if not opt.train_only:
    #         Data_Prep(opt)
    #         ValData_Prep(opt)
    #     Train(opt)
    # elif opt.task == 'test':
    #     Data_Prep(opt)
    #     Test(opt)

    print(dataset)
