import  torch, os
import  numpy as np
from zIncipientFaultNShotMixup import IncipientFaultNShotMixup,IncipientFaultNShotMixup_test,\
    IncipientFaultNShotMixup_test_adaptationlam,IncipientFaultNShotMixup_taskInitial,IncipientFaultNShotMixup_test_fixlam
import  argparse
import matplotlib.pyplot as plt
from    meta import Meta_base
import wandb
wandb.init(project="MAMLIncipientFaultDetection-Results", entity="shilixian")
def main(args):

    # torch.manual_seed(222)
    # torch.cuda.manual_seed_all(222)
    # np.random.seed(222)
    seed = 222
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    print(args)
    #初始的参数
    config = [
        # ****************6层卷积层参数设置256*********************
        ('conv2d', [64, 3, 3, 3, 2, 0]),  # 64种1通道3乘3卷积核 步长2 padding 0
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 3, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [64, 64, 3, 3, 2, 0]),
        ('relu', [True]),
        ('bn', [64]),
        ('flatten', []),
        # ('linear', [16, 256]),
        ('relu', [True]),
        ('linear', [args.n_way, 256])
    ]

    device = torch.device('cuda')
    maml = Meta_base(args, config).to(device)

    tmp = filter(lambda x: x.requires_grad, maml.parameters()) ## filter
    num = sum(map(lambda x: np.prod(x.shape), tmp)) #计算数组元素的乘积，得到所有需要训练向量的个数
    print(maml)
    print('Total trainable tensors:', num)

    db_train = IncipientFaultNShotMixup('TrainData7.19Mixup',
                       batchsz=args.task_num,
                       n_way=args.n_way,
                       k_shot=args.k_spt,
                       k_query=args.k_qry,
                       imgsz=args.imgsz,
                       chan=args.imgc)
    mean, std = db_train.normalization()
    db_test = IncipientFaultNShotMixup_test_adaptationlam('Data5',
                                                          batchsz=args.task_num_test,
                                                          n_way=args.n_way,
                                                          k_shot=args.k_spt_test,
                                                          k_query=args.k_qry_test,
                                                          imgsz=args.imgsz,
                                                          chan=args.imgc, mean=mean, std=std,
                                                          )
    acc_array=[]
    for step in range(args.epoch):
        x_spt, y_spt, x_qry, y_qry = db_train.next()
        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                     torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
        # plt.imshow(x_spt[0][0][0]).plt.show()
        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        accs = maml(x_spt, y_spt, x_qry, y_qry)
        train_acc=accs[-1]
        # wandb.log({
        #     "train_acc": train_acc,
        # })
        wandb.log({
            "train_acc": train_acc,
        })
        if step % 50 == 0:
            print('step:', step, '\ttraining acc:', accs)

        if step % 100 == 0:
            accs = []
            accs_last = []
            macro_precisions = []
            macro_f1s = []
            macro_recalls = []
            # ccc=1000//args.task_num
            for _ in range(100//args.task_num):

                # test
                # x_spt, y_spt, x_qry, y_qry = db_test.next('test')
                x_spt, y_spt, x_qry, y_qry = db_test.next('test')
                x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                                             torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc,acc,macro_precision,macro_f1,macro_recall = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append( test_acc )
                    macro_precisions.append(macro_precision)
                    macro_f1s.append(macro_f1)
                    macro_recalls.append(macro_recall)
                    accs.append(test_acc)

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            accs_last = np.array(accs_last).mean(axis=0).astype(np.float16)
            macro_precisions = np.array(macro_precisions).mean(axis=0).astype(np.float16)
            macro_f1s = np.array(macro_f1s).mean(axis=0).astype(np.float16)
            macro_recalls = np.array(macro_recalls).mean(axis=0).astype(np.float16)
            print('Test acc:', accs)
            print('Test macro_precisions:', macro_precisions)
            print('Test macro_f1s:', macro_f1s)
            print('Test macro_recalls:', macro_recalls)
            test_acc = accs
            acc_array.append(test_acc[-1])
            wandb.log({
                "test_acc": accs[-1],
                "macro_precisions": macro_precisions,
                "macro_f1s": macro_f1s,
                "macro_recalls": macro_recalls,
            })
    y=range(0, args.update_step_test+1, 1)
    y=list(y)
    plt.plot(y, test_acc)
    plt.show()
    x=range(0, args.epoch, 100)
    x=list(x)
    plt.plot(x, acc_array)
    plt.show()

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    # argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
    # argparser.add_argument('--n_way', type=int, help='n way', default=5)
    # argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
    argparser.add_argument('--epoch', type=int, help='epoch number', default=3000) #我设置的是1000
    argparser.add_argument('--n_way', type=int, help='n way', default=3) #5
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3)
    argparser.add_argument('--k_spt_test', type=int, help='k shot for support set', default=4)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=2) #15
    argparser.add_argument('--k_qry_test', type=int, help='k shot for query set', default=10)# 15
    argparser.add_argument('--imgsz', type=int, help='imgsize', default=256)#28 64
    argparser.add_argument('--imgc', type=int, help='imgchannels', default=3)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
    argparser.add_argument('--task_num_test', type=int, help='meta batch size, namely task num', default=1)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)#0.4
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)#10
    args = argparser.parse_args()

    main(args)
