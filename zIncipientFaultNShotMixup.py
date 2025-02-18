
import  torchvision.transforms as transforms
from    PIL import Image
import  os.path
import  numpy as np
import torch
import pandas as pd
import math
import torchvision
import cv2
import matplotlib.pyplot as plt


class IncipientFaultNShotMixup:
    # BaseLine
    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz,chan):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz
        # if not os.path.isfile(os.path.join(root, 'omniglot.npy')):
        #     # if root/data.npy does not exist, just download it
        #     self.x = Omniglot(root, download=True,
        #                       transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
        #                                                     lambda x: x.resize((imgsz, imgsz)),
        #                                                     lambda x: np.reshape(x, (imgsz, imgsz, 1)),
        #                                                     lambda x: np.transpose(x, [2, 0, 1]),
        #                                                     lambda x: x/255.])
        #                       )
        #
        #     temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
        #     for (img, label) in self.x:
        #         if label in temp.keys():
        #             temp[label].append(img)
        #         else:
        #             temp[label] = [img]
        #
        #     self.x = []
        #     for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
        #         self.x.append(np.array(imgs))
        #
        #     # as different class may have different number of imgs
        #     self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
        #     # each character contains 20 imgs
        #     print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
        #     temp = []  # Free memory
        #     # save all dataset into npy file.
        #     np.save(os.path.join(root, 'omniglot.npy'), self.x)
        #     print('write into omniglot.npy.')
        # else:
        # if data.npy exists, just load it.
        self.imgsz=imgsz
        self.root=root
        self.chan=chan
        print('load Incipient fault')
        self.transform = transforms.Compose([  # 多个transform操作
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.resize),  # 从中心裁剪
            transforms.ToTensor()
            # transforms.Normalize(
            #     mean=torch.tensor([0.485, 0.456, 0.406]),
            #     std=torch.tensor([0.229, 0.224, 0.225]))
            # transforms.RandomVerticalFlip(p=0.2),
            #
            # transforms.Lambda(self.expand_greyscale)  # 自行定义transform操作

        ])
        # self.x = os.listdir(os.path.join(self.root))
        temp = self.get_img_info(self.root, '.png')
        self.x = []
        # xt=np.zeros((1,3,1,28,28))
        for label, imgs in temp.items():
            # self.x.append(np.array(imgs))
            imgs=np.array(imgs)
            # imgs=imgs.numpy()
            # imgs=np.expand_dims(imgs,axis=0)
            # imgs=imgs.reshape(1,3,1,28,28)
            self.x.append(imgs)
            # self.xt=np.append(self.xt,imgs,0)
            # xt = np.concatenate([xt, imgs], axis=0)
            # print(1)
        # a=tempp.shape
        # as different class may have different number of imgs
        # len_x = len(self.x)
        # self.x_1=np.zeros([len_x,3,1,28,28])
        # for i in range(len_x):
        #     self.x_1[i,:,:,:,:]=self.x[i]
        self.x = np.array(self.x).astype(np.float32)
        # self.x=np.reshape(self.x,(13,3,1,28,28))
        # self.x_fault = np.array(self.x_fault)
        # self.x = np.array(self.x).astype(np.float)
        # self.x=np.reshape(self.x,(13,3,1,28,28))
        # self.x=self.x.reshape(13,3,1,28,28)
        # for i in range
        # each character contains 2 imgs

        print('data shape:', self.x.shape)

        # [5,2,1,28,28]
        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        # self.x_train, self.x_test = self.x, self.x
        if self.x.shape[0]>20:
            # self.x_train, self.x_test = self.x[9:], self.x[:9]
            self.x_train, self.x_test = self.x[:], self.x[:]
        else:
            self.x_train, self.x_test = self.x, self.x

        # self.normalization()

        self.batchsz = batchsz #1
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

    def get_rms(self,data):
        """
        均方根值 反映的是有效值而不是平均值
        """
        return math.sqrt(sum([x ** 2 for x in data]) / len(data))

    def expand_greyscale(self,t):
        return t.expand(3, -1, -1)

    # def get_img_info(self, data_dir, char):
    #     data_info = list()
    #     data_info_sum = list()
    #     temp_list=list()
    #     temp=dict()
    #     j=0
    #     type_path=os.listdir(data_dir)
    #     for sub_dir in type_path:
    #         subtype_path = os.listdir(os.path.join(data_dir,sub_dir))
    #         j = 0
    #         temp_list=[]
    #         rms_list=[]
    #         for ssubdir in subtype_path:
    #             docum_path=os.listdir(os.path.join(data_dir, sub_dir,ssubdir))
    #             image_path=docum_path[:6]
    #             excel_path=docum_path[6:]
    #             for k in range(0, len(image_path)):
    #                 img_load = os.path.join(data_dir, sub_dir,ssubdir, image_path[k])
    #                 img= Image.open(img_load)
    #                 img = img.convert('RGB')
    #                 img=self.transform(img)
    #                 img = img.numpy()
    #                 temp_list.append(img)
    #             for k in range(0, len(excel_path)):
    #                 excel_load = os.path.join(data_dir, sub_dir, ssubdir, excel_path[k])
    #                 excel_data=pd.read_excel(excel_load)
    #                 excel_data=np.array(excel_data)
    #                 # aa=[3,4,5]
    #                 rms_value=np.zeros(1)
    #                 for l in range(3):
    #                     data=self.get_rms(excel_data[:,l])
    #                     rms_value=rms_value+data
    #                 rms_value=rms_value/3
    #                 rms_delta=rms_value-math.sqrt(2)/2
    #                 rms_list.append(rms_delta)
    #         j=j+1
    #         mixed_temp=list()
    #         if j==1:
    #             #现在是3个样本 如果样本多了这里要改
    #             for i in [0,1,2,3,4,5]:
    #                 x1=temp_list[i]
    #                 x2 = temp_list[i+6]
    #                 rms1=rms_list[i]
    #                 rms2 = rms_list[i+6]
    #                 lam1 = rms1 / (rms1 + rms2)
    #                 lam2 = rms2 / (rms1 + rms2)
    #                 #mixup
    #                 # alpha=1
    #                 # lam = np.random.beta(alpha, alpha)
    #                 # lam=1
    #                 # index = torch.randperm(x.size(0)).cuda()
    #                 mixed_x = lam1 * x1 +  lam2 * x2
    #                 mixed_temp.append(mixed_x)
    #         label = sub_dir
    #         temp[label] = mixed_temp
    #     return temp

    def get_img_info(self, data_dir, char):
        data_info = list()
        data_info_sum = list()
        temp_list=list()
        temp=dict()
        j=0
        type_path=os.listdir(data_dir)
        for sub_dir in type_path:
            subtype_path = os.listdir(os.path.join(data_dir,sub_dir))
            for ssubtype_path in subtype_path:
                sssubtype_path=os.listdir(os.path.join(data_dir, sub_dir,ssubtype_path))
                for ssssubtype_path in sssubtype_path:
                    sssssubtype_path = os.listdir(os.path.join(data_dir, sub_dir, ssubtype_path,ssssubtype_path))
                    j = 0
                    temp_list=[]
                    rms_list=[]
                    for ssubdir in sssssubtype_path:
                        docum_path=os.listdir(os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir))
                        image_path=docum_path[:6]
                        excel_path=docum_path[6:]
                        for k in range(0, len(image_path)):
                            img_load = os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir,image_path[k])
                            img= Image.open(img_load)
                            img = img.convert('RGB')
                            img=self.transform(img)
                            img = img.numpy()
                            temp_list.append(img)
                        for k in range(0, len(excel_path)):
                            excel_load = os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir,excel_path[k])
                            excel_data=pd.read_excel(excel_load)
                            excel_data=np.array(excel_data)
                            # aa=[3,4,5]
                            rms_value=np.zeros(1)
                            for l in range(3):
                                data=self.get_rms(excel_data[:,l])
                                rms_value=rms_value+data
                            #three phase rms value average
                            rms_value=rms_value/3
                            rms_delta=rms_value-math.sqrt(2)/2
                            rms_delta=abs(rms_delta)
                            rms_list.append(rms_delta)
                    j=j+1
                    mixed_temp=list()
                    if j==1:
                        #现在是3个样本 如果样本多了这里要改
                        for i in [0,1,2,3,4,5]:
                            x1=temp_list[i]
                            x2 = temp_list[i+6]
                            rms1=rms_list[i]
                            rms2 = rms_list[i+6]
                            lam1 = rms1 / (rms1 + rms2)
                            lam2 = rms2 / (rms1 + rms2)
                            # lam1 = 0.5
                            # lam2 = 0.5
                            #mixup
                            # alpha=1
                            # lam = np.random.beta(alpha, alpha)
                            # lam=1
                            # index = torch.randperm(x.size(0)).cuda()
                            mixed_x = lam1 * x1 + lam2 * x2
                            mixed_temp.append(mixed_x)
                    label = sub_dir+ssubtype_path+ssssubtype_path
                    temp[label] = mixed_temp
        return temp


    # def normalization(self):
    #     """
    #     Normalizes our data, to have a mean of 0 and sdt of 1
    #     """
    #     self.mean = np.mean(self.x_train)
    #     self.std = np.std(self.x_train)
    #     self.max = np.max(self.x_train)
    #     self.min = np.min(self.x_train)
    #     # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
    #     self.x_train = (self.x_train - self.mean) / self.std
    #     self.x_test = (self.x_test - self.mean) / self.std
    #
    #     self.mean = np.mean(self.x_train)
    #     self.std = np.std(self.x_train)
    #     self.max = np.max(self.x_train)
    #     self.min = np.min(self.x_train)
    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        return self.mean,self.std
    # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                # selected_cls = np.array(list([0,1,2,3,4]))
                # selected_cls = np.array(list([0, 1, 2]))
                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(6, self.k_shot + self.k_query, False)
                    # selected_img=[0,1,2]

                    # selected_img = np.array([0,1,2,3,4,5])
                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                # perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.resize, self.resize, self.chan)
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)
                # perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.resize, self.resize, self.chan)
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)
                # print(1)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, self.resize, self.resize, self.chan)
            x_spts = np.transpose(x_spts, (0, 1, 4, 2, 3))
            y_spts = np.array(y_spts).astype(np.int64).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.resize, self.resize, self.chan)
            x_qrys = np.transpose(x_qrys, (0, 1, 4, 2, 3))
            y_qrys = np.array(y_qrys).astype(np.int64).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch

class IncipientFaultNShotMixup_taskInitial:
    # incipient fault initial parameters RMS weight
    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz,chan):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """

        self.resize = imgsz
        # if not os.path.isfile(os.path.join(root, 'omniglot.npy')):
        #     # if root/data.npy does not exist, just download it
        #     self.x = Omniglot(root, download=True,
        #                       transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
        #                                                     lambda x: x.resize((imgsz, imgsz)),
        #                                                     lambda x: np.reshape(x, (imgsz, imgsz, 1)),
        #                                                     lambda x: np.transpose(x, [2, 0, 1]),
        #                                                     lambda x: x/255.])
        #                       )
        #
        #     temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
        #     for (img, label) in self.x:
        #         if label in temp.keys():
        #             temp[label].append(img)
        #         else:
        #             temp[label] = [img]
        #
        #     self.x = []
        #     for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
        #         self.x.append(np.array(imgs))
        #
        #     # as different class may have different number of imgs
        #     self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
        #     # each character contains 20 imgs
        #     print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
        #     temp = []  # Free memory
        #     # save all dataset into npy file.
        #     np.save(os.path.join(root, 'omniglot.npy'), self.x)
        #     print('write into omniglot.npy.')
        # else:
        # if data.npy exists, just load it.
        self.imgsz=imgsz
        self.root=root
        self.chan=chan
        print('load Incipient fault')
        self.transform = transforms.Compose([  # 多个transform操作
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.resize),  # 从中心裁剪
            transforms.ToTensor()
            # transforms.Normalize(
            #     mean=torch.tensor([0.485, 0.456, 0.406]),
            #     std=torch.tensor([0.229, 0.224, 0.225]))
            # transforms.RandomVerticalFlip(p=0.2),
            #
            # transforms.Lambda(self.expand_greyscale)  # 自行定义transform操作

        ])
        # self.x = os.listdir(os.path.join(self.root))
        temp,temp_rms = self.get_img_info(self.root, '.png')
        self.x = []
        # xt=np.zeros((1,3,1,28,28))
        for label, imgs in temp.items():
            # self.x.append(np.array(imgs))
            imgs=np.array(imgs)

            # imgs=imgs.numpy()
            # imgs=np.expand_dims(imgs,axis=0)
            # imgs=imgs.reshape(1,3,1,28,28)
            self.x.append(imgs)
            # self.xt=np.append(self.xt,imgs,0)
            # xt = np.concatenate([xt, imgs], axis=0)
            # print(1)
        # a=tempp.shape
        # as different class may have different number of imgs
        # self.x_temp=[]
        # self.rms_temp = []
        # for i in range (len(self.x)):
        #     aa=self.x[i][:6]
        #     rms=self.x[i][6]
        #     self.rms_temp.append(rms)
        #     self.x_temp.append(aa)
        self.x = np.array(self.x).astype(np.float32)
        self.temp_rms=np.array(temp_rms).astype(np.float32)
        # each character contains 2 imgs

        print('data shape:', self.x.shape)
        # [5,2,1,28,28]
        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        # self.x_train, self.x_test = self.x, self.x
        if self.x.shape[0]>10:
            # self.x_train, self.x_test = self.x[9:], self.x[:9]
            self.x_train, self.x_test, self.x_train_rms,self.x_test_rms= self.x[:], self.x[:],temp_rms,temp_rms
        else:
            self.x_train, self.x_test = self.x, self.x

        self.mean,self.std=self.normalization()
        self.batchsz = batchsz #1
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("Train task DB: train", self.x_train.shape, "test", self.x_test.shape)
        data_cache_train,x_rmss_train=self.load_data_cache(self.datasets["train"], self.temp_rms)
        data_cache_test, x_rmss_test= self.load_data_cache(self.datasets["test"], self.temp_rms)
        self.datasets_cache = {"train": data_cache_train,  # current epoch data cached
                               "test": data_cache_test}
        self.rmss_cache={"train": x_rmss_train,  # current epoch data cached
                               "test": x_rmss_test
        }

    def get_rms(self,data):
        """
        均方根值 反映的是有效值而不是平均值
        """
        return math.sqrt(sum([x ** 2 for x in data]) / len(data))

    def expand_greyscale(self,t):
        return t.expand(3, -1, -1)

    # def get_img_info(self, data_dir, char):
    #     data_info = list()
    #     data_info_sum = list()
    #     temp_list=list()
    #     temp=dict()
    #     j=0
    #     type_path=os.listdir(data_dir)
    #     for sub_dir in type_path:
    #         subtype_path = os.listdir(os.path.join(data_dir,sub_dir))
    #         j = 0
    #         temp_list=[]
    #         rms_list=[]
    #         for ssubdir in subtype_path:
    #             docum_path=os.listdir(os.path.join(data_dir, sub_dir,ssubdir))
    #             image_path=docum_path[:6]
    #             excel_path=docum_path[6:]
    #             for k in range(0, len(image_path)):
    #                 img_load = os.path.join(data_dir, sub_dir,ssubdir, image_path[k])
    #                 img= Image.open(img_load)
    #                 img = img.convert('RGB')
    #                 img=self.transform(img)
    #                 img = img.numpy()
    #                 temp_list.append(img)
    #             for k in range(0, len(excel_path)):
    #                 excel_load = os.path.join(data_dir, sub_dir, ssubdir, excel_path[k])
    #                 excel_data=pd.read_excel(excel_load)
    #                 excel_data=np.array(excel_data)
    #                 # aa=[3,4,5]
    #                 rms_value=np.zeros(1)
    #                 for l in range(3):
    #                     data=self.get_rms(excel_data[:,l])
    #                     rms_value=rms_value+data
    #                 rms_value=rms_value/3
    #                 rms_delta=rms_value-math.sqrt(2)/2
    #                 rms_list.append(rms_delta)
    #         j=j+1
    #         mixed_temp=list()
    #         if j==1:
    #             #现在是3个样本 如果样本多了这里要改
    #             for i in [0,1,2,3,4,5]:
    #                 x1=temp_list[i]
    #                 x2 = temp_list[i+6]
    #                 rms1=rms_list[i]
    #                 rms2 = rms_list[i+6]
    #                 lam1 = rms1 / (rms1 + rms2)
    #                 lam2 = rms2 / (rms1 + rms2)
    #                 #mixup
    #                 # alpha=1
    #                 # lam = np.random.beta(alpha, alpha)
    #                 # lam=1
    #                 # index = torch.randperm(x.size(0)).cuda()
    #                 mixed_x = lam1 * x1 +  lam2 * x2
    #                 mixed_temp.append(mixed_x)
    #         label = sub_dir
    #         temp[label] = mixed_temp
    #     return temp

    def get_img_info(self, data_dir, char):
        data_info = list()
        data_info_sum = list()
        temp_list=list()
        temp=dict()
        temp_rms=[]
        j=0
        # k=0
        type_path=os.listdir(data_dir)
        for sub_dir in type_path:
            subtype_path = os.listdir(os.path.join(data_dir,sub_dir))
            for ssubtype_path in subtype_path:
                sssubtype_path=os.listdir(os.path.join(data_dir, sub_dir,ssubtype_path))
                for ssssubtype_path in sssubtype_path:
                    sssssubtype_path = os.listdir(os.path.join(data_dir, sub_dir, ssubtype_path,ssssubtype_path))
                    j = 0
                    temp_list=[]
                    rms_list=[]
                    for ssubdir in sssssubtype_path:
                        docum_path=os.listdir(os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir))
                        image_path=docum_path[:6]
                        excel_path=docum_path[6:]
                        for k in range(0, len(image_path)):
                            img_load = os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir,image_path[k])
                            img= Image.open(img_load)
                            img = img.convert('RGB')
                            img=self.transform(img)
                            img = img.numpy()
                            temp_list.append(img)
                        for k in range(0, len(excel_path)):
                            excel_load = os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir,excel_path[k])
                            excel_data=pd.read_excel(excel_load)
                            excel_data=np.array(excel_data)
                            # aa=[3,4,5]
                            rms_value=np.zeros(1)
                            for l in range(3):
                                data=self.get_rms(excel_data[:,l])
                                rms_value=rms_value+data
                            rms_value=rms_value/3
                            rms_delta=rms_value-math.sqrt(2)/2
                            rms_delta=abs(rms_delta)
                            rms_list.append(rms_delta)
                    j=j+1
                    mixed_temp=list()
                    if j==1:
                        #现在是3个样本 如果样本多了这里要改
                        for i in [0,1,2,3,4,5]:
                            x1=temp_list[i]
                            x2 = temp_list[i+6]
                            rms1=rms_list[i]
                            rms2 = rms_list[i+6]
                            lam1 = rms1 / (rms1 + rms2)
                            lam2 = rms2 / (rms1 + rms2)
                            # lam1 = 0.1
                            # lam2 = 0.9
                            #mixup
                            # alpha=1
                            # lam = np.random.beta(alpha, alpha)
                            # lam=1
                            # index = torch.randperm(x.size(0)).cuda()
                            mixed_x = lam1 * x1 + lam2 * x2
                            mixed_temp.append(mixed_x)
                    label = sub_dir+ssubtype_path+ssssubtype_path
                    temp[label] = mixed_temp
                    # label2='zz'+'rms'+str(k)
                    # k=k+1
                    temp_rms.append(rms1)
                    # deltarms='Deltarms'
                    # temp[deltarms]=
        return temp,temp_rms


    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        # self.mean = np.mean(self.x_train)
        # self.std = np.std(self.x_train)
        # self.max = np.max(self.x_train)
        # self.min = np.min(self.x_train)
        return self.mean,self.std


    def load_data_cache(self, data_pack,rms):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []
        rmss_cache=[]
        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            x_rmss=[]
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                x_rms = []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                # selected_cls = np.array(list([0,1,2,3,4]))
                # selected_cls = np.array(list([0, 1, 2]))
                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(6, self.k_shot + self.k_query, False)
                    # selected_img=[0,1,2]

                    # selected_img = np.array([0,1,2,3,4,5])
                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                    x_rms.append(rms[cur_class])

                # shuffle inside a batch
                # perm = np.random.permutation(self.n_way * self.k_shot)
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.resize, self.resize, self.chan)
                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)
                # perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.resize, self.resize, self.chan)
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)
                x_rms=np.array(x_rms)

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)
                x_rmss.append(x_rms)
                # print(1)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, self.resize, self.resize, self.chan)
            x_spts = np.transpose(x_spts, (0, 1, 4, 2, 3))
            y_spts = np.array(y_spts).astype(np.int64).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.resize, self.resize, self.chan)
            x_qrys = np.transpose(x_qrys, (0, 1, 4, 2, 3))
            y_qrys = np.array(y_qrys).astype(np.int64).reshape(self.batchsz, querysz)
            x_rmss=np.array(x_rmss).astype(np.float32).reshape(self.batchsz, self.n_way)
            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])
            rmss_cache.append([x_rmss])

        return data_cache,rmss_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode],self.rmss_cache[mode]= self.load_data_cache(self.datasets[mode],self.temp_rms)

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        rms_batch = self.rmss_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch,rms_batch
# class IncipientFaultNShotMixup_test:
#     # 按单个样本自适应加权 去除了归一化
#     def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz,chan):
#         """
#         Different from mnistNShot, the
#         :param root:
#         :param batchsz: task num
#         :param n_way:
#         :param k_shot:
#         :param k_qry:
#         :param imgsz:
#         """
#
#         self.resize = imgsz
#         # if not os.path.isfile(os.path.join(root, 'omniglot.npy')):
#         #     # if root/data.npy does not exist, just download it
#         #     self.x = Omniglot(root, download=True,
#         #                       transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
#         #                                                     lambda x: x.resize((imgsz, imgsz)),
#         #                                                     lambda x: np.reshape(x, (imgsz, imgsz, 1)),
#         #                                                     lambda x: np.transpose(x, [2, 0, 1]),
#         #                                                     lambda x: x/255.])
#         #                       )
#         #
#         #     temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
#         #     for (img, label) in self.x:
#         #         if label in temp.keys():
#         #             temp[label].append(img)
#         #         else:
#         #             temp[label] = [img]
#         #
#         #     self.x = []
#         #     for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
#         #         self.x.append(np.array(imgs))
#         #
#         #     # as different class may have different number of imgs
#         #     self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
#         #     # each character contains 20 imgs
#         #     print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
#         #     temp = []  # Free memory
#         #     # save all dataset into npy file.
#         #     np.save(os.path.join(root, 'omniglot.npy'), self.x)
#         #     print('write into omniglot.npy.')
#         # else:
#         # if data.npy exists, just load it.
#         self.imgsz=imgsz
#         self.root=root
#         self.chan=chan
#         print('load Incipient fault')
#         self.transform = transforms.Compose([  # 多个transform操作
#             transforms.Resize(self.resize),
#             transforms.CenterCrop(self.resize),  # 从中心裁剪
#             transforms.ToTensor()
#             # transforms.Normalize(
#             #     mean=torch.tensor([0.485, 0.456, 0.406]),
#             #     std=torch.tensor([0.229, 0.224, 0.225]))
#             # transforms.RandomVerticalFlip(p=0.2),
#             #
#             # transforms.Lambda(self.expand_greyscale)  # 自行定义transform操作
#
#         ])
#         # self.x = os.listdir(os.path.join(self.root))
#         temp = self.get_img_info(self.root, '.png')
#         self.x = []
#         # xt=np.zeros((1,3,1,28,28))
#         for label, imgs in temp.items():
#             # self.x.append(np.array(imgs))
#             imgs=np.array(imgs)
#             # imgs=imgs.numpy()
#             # imgs=np.expand_dims(imgs,axis=0)
#             # imgs=imgs.reshape(1,3,1,28,28)
#             self.x.append(imgs)
#             # self.xt=np.append(self.xt,imgs,0)
#             # xt = np.concatenate([xt, imgs], axis=0)
#             # print(1)
#         # a=tempp.shape
#         # as different class may have different number of imgs
#         # len_x = len(self.x)
#         # self.x_1=np.zeros([len_x,3,1,28,28])
#         # for i in range(len_x):
#         #     self.x_1[i,:,:,:,:]=self.x[i]
#         self.x = np.array(self.x).astype(np.float)
#         # self.x=np.reshape(self.x,(13,3,1,28,28))
#         # self.x_fault = np.array(self.x_fault)
#         # self.x = np.array(self.x).astype(np.float)
#         # self.x=np.reshape(self.x,(13,3,1,28,28))
#         # self.x=self.x.reshape(13,3,1,28,28)
#         # for i in range
#         # each character contains 2 imgs
#
#         print('data shape:', self.x.shape)
#
#         # [5,2,1,28,28]
#         # [1623, 20, 84, 84, 1]
#         # TODO: can not shuffle here, we must keep training and test set distinct!
#         # self.x_train, self.x_test = self.x, self.x
#         if self.x.shape[0]>20:
#             # self.x_train, self.x_test = self.x[9:], self.x[:9]
#             self.x_train, self.x_test = self.x[:], self.x[:]
#         else:
#             self.x_train, self.x_test = self.x, self.x
#
#         # self.normalization()
#
#         self.batchsz = batchsz #1
#         self.n_cls = self.x.shape[0]  # 1623
#         self.n_way = n_way  # n way
#         self.k_shot = k_shot  # k shot
#         self.k_query = k_query  # k query
#         assert (k_shot + k_query) <=20
#
#         # save pointer of current read batch in total cache
#         self.indexes = {"train": 0, "test": 0}
#         self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
#         print("DB: train", self.x_train.shape, "test", self.x_test.shape)
#
#         self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
#                                "test": self.load_data_cache(self.datasets["test"])}
#
#     def get_rms(self,data):
#         """
#         均方根值 反映的是有效值而不是平均值
#         """
#         return math.sqrt(sum([x ** 2 for x in data]) / len(data))
#
#     def expand_greyscale(self,t):
#         return t.expand(3, -1, -1)
#
#     # def get_img_info(self, data_dir, char):
#     #     data_info = list()
#     #     data_info_sum = list()
#     #     temp_list=list()
#     #     temp=dict()
#     #     j=0
#     #     type_path=os.listdir(data_dir)
#     #     for sub_dir in type_path:
#     #         subtype_path = os.listdir(os.path.join(data_dir,sub_dir))
#     #         j = 0
#     #         temp_list=[]
#     #         rms_list=[]
#     #         for ssubdir in subtype_path:
#     #             docum_path=os.listdir(os.path.join(data_dir, sub_dir,ssubdir))
#     #             image_path=docum_path[:6]
#     #             excel_path=docum_path[6:]
#     #             for k in range(0, len(image_path)):
#     #                 img_load = os.path.join(data_dir, sub_dir,ssubdir, image_path[k])
#     #                 img= Image.open(img_load)
#     #                 img = img.convert('RGB')
#     #                 img=self.transform(img)
#     #                 img = img.numpy()
#     #                 temp_list.append(img)
#     #             for k in range(0, len(excel_path)):
#     #                 excel_load = os.path.join(data_dir, sub_dir, ssubdir, excel_path[k])
#     #                 excel_data=pd.read_excel(excel_load)
#     #                 excel_data=np.array(excel_data)
#     #                 # aa=[3,4,5]
#     #                 rms_value=np.zeros(1)
#     #                 for l in range(3):
#     #                     data=self.get_rms(excel_data[:,l])
#     #                     rms_value=rms_value+data
#     #                 rms_value=rms_value/3
#     #                 rms_delta=rms_value-math.sqrt(2)/2
#     #                 rms_list.append(rms_delta)
#     #         j=j+1
#     #         mixed_temp=list()
#     #         if j==1:
#     #             #现在是3个样本 如果样本多了这里要改
#     #             for i in [0,1,2,3,4,5]:
#     #                 x1=temp_list[i]
#     #                 x2 = temp_list[i+6]
#     #                 rms1=rms_list[i]
#     #                 rms2 = rms_list[i+6]
#     #                 lam1 = rms1 / (rms1 + rms2)
#     #                 lam2 = rms2 / (rms1 + rms2)
#     #                 #mixup
#     #                 # alpha=1
#     #                 # lam = np.random.beta(alpha, alpha)
#     #                 # lam=1
#     #                 # index = torch.randperm(x.size(0)).cuda()
#     #                 mixed_x = lam1 * x1 +  lam2 * x2
#     #                 mixed_temp.append(mixed_x)
#     #         label = sub_dir
#     #         temp[label] = mixed_temp
#     #     return temp
#
#     def get_img_info(self, data_dir, char):
#         data_info = list()
#         data_info_sum = list()
#         temp_list=list()
#         temp=dict()
#         j=0
#         type_path=os.listdir(data_dir)
#         for sub_dir in type_path:
#             subtype_path = os.listdir(os.path.join(data_dir,sub_dir))
#             for ssubtype_path in subtype_path:
#                 sssubtype_path=os.listdir(os.path.join(data_dir, sub_dir,ssubtype_path))
#                 for ssssubtype_path in sssubtype_path:
#                     sssssubtype_path = os.listdir(os.path.join(data_dir, sub_dir, ssubtype_path,ssssubtype_path))
#                     j = 0
#                     temp_list=[]
#                     rms_list=[]
#                     for ssubdir in sssssubtype_path:
#                         docum_path=os.listdir(os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir))
#                         image_path=docum_path[:5]
#                         excel_path=docum_path[5:]
#                         for k in range(0, len(image_path)):
#                             img_load = os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir,image_path[k])
#                             # img= Image.open(img_load)
#                             # img = img.convert('RGB')
#                             img = cv2.imread(img_load)
#                             cv2.imshow("show_img",img)
#                             img=img.reshape(3,656,875)
#                             plt.imshow(img)
#                             # cv2.imshow("show_img", img)
#                             img=self.transform(img)
#                             # img = img.numpy()
#                             input_tensor = img.to(torch.device('cpu'))
#                             torchvision.utils.save_image(input_tensor, ".\out_cv.jpg")
#                             # img = img.numpy() #***************************************************
#                             temp_list.append(img)
#                         nn=0
#                         rms_value = np.zeros(1)
#                         for k in range(0, len(excel_path)):
#                             nn = nn + 1
#                             excel_load = os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir,excel_path[k])
#                             excel_data=pd.read_excel(excel_load)
#                             excel_data=np.array(excel_data)
#                             data = self.get_rms(excel_data)
#                             rms_value=rms_value+data
#                             if nn%3==0:
#                                 # aa=[3,4,5]
#                                 rms_value=rms_value/3
#                                 rms_delta = rms_value - math.sqrt(2) / 2
#                                 rms_delta=abs(rms_delta)
#                                 rms_list.append(rms_delta)
#                                 rms_value = np.zeros(1)
#
#                     j=j+1
#                     mixed_temp=list()
#                     if j==1:
#                         #现在是5个样本 如果样本多了这里要改
#                         for i in [0,1,2,3,4]:
#                             x1=temp_list[i]
#                             x2 = temp_list[i+5]
#                             input_tensor = x1.to(torch.device('cpu'))
#                             torchvision.utils.save_image(input_tensor, "ox1.jpg")
#                             input_tensor = x2.to(torch.device('cpu'))
#                             torchvision.utils.save_image(input_tensor, "ox2.jpg")
#                             rms1=rms_list[i]
#                             rms2 = rms_list[i+5]
#                             lam1 = rms1 / (rms1 + rms2)
#                             lam2 = rms2 / (rms1 + rms2)
#                             lam1 = 0.8
#                             lam2 = 0.2
#                             #mixup
#                             # alpha=1
#                             # lam = np.random.beta(alpha, alpha)
#                             # lam=1
#                             # index = torch.randperm(x.size(0)).cuda()
#                             mixed_x = lam1 * x1 + lam2 * x2
#                             lamimg1 = lam1 * x1
#                             lamimg2 = lam2 * x2
#                             torch.save(mixed_x, 'mixed_x0.80.2.pt')
#
#                             lamimg2_change=lamimg2[:,:64,:64]
#                             input_tensor = mixed_x.to(torch.device('cpu'))
#                             torchvision.utils.save_image(input_tensor, "ox3.jpg")
#                             input_tensor = lamimg1.to(torch.device('cpu'))
#                             torchvision.utils.save_image(input_tensor, "olamimg1.jpg")
#                             input_tensor = lamimg2.to(torch.device('cpu'))
#                             torchvision.utils.save_image(input_tensor, "olamimg2.jpg")
#                             input_tensor = lamimg2_change.to(torch.device('cpu'))
#                             torchvision.utils.save_image(input_tensor, "lamimg2_change.jpg")
#                             # mixed_x=transforms.Normalize(
#                             #         mean=torch.tensor([0.485, 0.456, 0.406]),
#                             #         std=torch.tensor([0.229, 0.224, 0.225]))(mixed_x)
#                             # torchvision.utils.save_image(mixed_x, "ox3归一化后.jpg")
#                             mixed_x=mixed_x.numpy()
#                             mixed_temp.append(mixed_x)
#                     label = sub_dir+ssubtype_path+ssssubtype_path
#                     temp[label] = mixed_temp
#         return temp
#
#
#     def normalization(self):
#         """
#         Normalizes our data, to have a mean of 0 and sdt of 1
#         """
#         self.mean = np.mean(self.x_train)
#         self.std = np.std(self.x_train)
#         self.max = np.max(self.x_train)
#         self.min = np.min(self.x_train)
#         # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
#         self.x_train = (self.x_train - self.mean) / self.std
#         self.x_test = (self.x_test - self.mean) / self.std
#
#         self.mean = np.mean(self.x_train)
#         self.std = np.std(self.x_train)
#         self.max = np.max(self.x_train)
#         self.min = np.min(self.x_train)
#
#     # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
#
#     def load_data_cache(self, data_pack):
#         """
#         Collects several batches data for N-shot learning
#         :param data_pack: [cls_num, 20, 84, 84, 1]
#         :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
#         """
#         #  take 5 way 1 shot as example: 5 * 1
#         setsz = self.k_shot * self.n_way
#         querysz = self.k_query * self.n_way
#         data_cache = []
#
#         # print('preload next 50 caches of batchsz of batch.')
#         for sample in range(10):  # num of episodes
#
#             x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
#             for i in range(self.batchsz):  # one batch means one set
#
#                 x_spt, y_spt, x_qry, y_qry = [], [], [], []
#                 selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
#                 # selected_cls = np.array(list([0,1,2,3,4]))
#                 # selected_cls = np.array(list([0, 1, 2]))
#                 for j, cur_class in enumerate(selected_cls):
#
#                     selected_img = np.random.choice(5, self.k_shot + self.k_query, False)
#                     # selected_img=[0,1,2]
#
#                     # selected_img = np.array([0,1,2,3,4,5])
#                     # meta-training and meta-test
#                     x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
#                     x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
#                     y_spt.append([j for _ in range(self.k_shot)])
#                     y_qry.append([j for _ in range(self.k_query)])
#
#                 # shuffle inside a batch
#                 # perm = np.random.permutation(self.n_way * self.k_shot)
#                 x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.chan, self.resize, self.resize)
#                 y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)
#                 # perm = np.random.permutation(self.n_way * self.k_query)
#                 x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.chan, self.resize, self.resize)
#                 y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)
#
#                 # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
#                 x_spts.append(x_spt)
#                 y_spts.append(y_spt)
#                 x_qrys.append(x_qry)
#                 y_qrys.append(y_qry)
#                 # print(1)
#
#
#             # [b, setsz, 1, 84, 84]
#             x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, self.chan, self.resize, self.resize)
#             y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
#             # [b, qrysz, 1, 84, 84]
#             x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.chan, self.resize, self.resize)
#             y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)
#
#             data_cache.append([x_spts, y_spts, x_qrys, y_qrys])
#
#         return data_cache
#
#     def next(self, mode='train'):
#         """
#         Gets next batch from the dataset with name.
#         :param mode: The name of the splitting (one of "train", "val", "test")
#         :return:
#         """
#         # update cache if indexes is larger cached num
#         if self.indexes[mode] >= len(self.datasets_cache[mode]):
#             self.indexes[mode] = 0
#             self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])
#
#         next_batch = self.datasets_cache[mode][self.indexes[mode]]
#         self.indexes[mode] += 1
#
#         return next_batch
class IncipientFaultNShotMixup_test_adaptationlam:
    # 按类别自适应加权 去除了归一化  又加上了归一化

    def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz,chan,mean,std):
        """
        Different from mnistNShot, the
        :param root:
        :param batchsz: task num
        :param n_way:
        :param k_shot:
        :param k_qry:
        :param imgsz:
        """
        self.sample_num = 14
        self.resize = imgsz
        # if not os.path.isfile(os.path.join(root, 'omniglot.npy')):
        #     # if root/data.npy does not exist, just download it
        #     self.x = Omniglot(root, download=True,
        #                       transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
        #                                                     lambda x: x.resize((imgsz, imgsz)),
        #                                                     lambda x: np.reshape(x, (imgsz, imgsz, 1)),
        #                                                     lambda x: np.transpose(x, [2, 0, 1]),
        #                                                     lambda x: x/255.])
        #                       )
        #
        #     temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
        #     for (img, label) in self.x:
        #         if label in temp.keys():
        #             temp[label].append(img)
        #         else:
        #             temp[label] = [img]
        #
        #     self.x = []
        #     for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
        #         self.x.append(np.array(imgs))
        #
        #     # as different class may have different number of imgs
        #     self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
        #     # each character contains 20 imgs
        #     print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
        #     temp = []  # Free memory
        #     # save all dataset into npy file.
        #     np.save(os.path.join(root, 'omniglot.npy'), self.x)
        #     print('write into omniglot.npy.')
        # else:
        # if data.npy exists, just load it.
        self.imgsz=imgsz
        self.root=root
        self.chan=chan
        self.mean=mean
        self.std=std
        print('load Incipient fault')
        self.transform = transforms.Compose([  # 多个transform操作
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.resize),  # 从中心裁剪
            transforms.ToTensor()
            # transforms.Normalize(
            #     mean=torch.tensor([0.485, 0.456, 0.406]),
            #     std=torch.tensor([0.229, 0.224, 0.225]))
            # transforms.RandomVerticalFlip(p=0.2),
            #
            # transforms.Lambda(self.expand_greyscale)  # 自行定义transform操作
        ])
        # self.x = os.listdir(os.path.join(self.root))
        # TODO zyg
        # temp = self.get_img_info(self.root, '.png')
        temp, self.temp_dict = self.get_img_info(self.root, '.png')
        self.x = []
        # xt=np.zeros((1,3,1,28,28))
        # for label, imgs in temp.items():
        for label, imgs_dict in self.temp_dict.items():
            # self.x.append(np.array(imgs))
            imgs = np.array([list(item) for item in imgs_dict.values()]) # zyg
            # imgs=np.array(imgs) # raw
            # imgs=imgs.numpy()
            # imgs=np.expand_dims(imgs,axis=0)
            # imgs=imgs.reshape(1,3,1,28,28)
            self.x.append(imgs)
            # self.xt=np.append(self.xt,imgs,0)
            # xt = np.concatenate([xt, imgs], axis=0)
            # print(1)
        # a=tempp.shape
        # as different class may have different number of imgs
        # len_x = len(self.x)
        # self.x_1=np.zeros([len_x,3,1,28,28])
        # for i in range(len_x):
        #     self.x_1[i,:,:,:,:]=self.x[i]
        self.x = np.array(self.x).astype(np.float32)
        # self.x=np.reshape(self.x,(13,3,1,28,28))
        # self.x_fault = np.array(self.x_fault)
        # self.x = np.array(self.x).astype(np.float)
        # self.x=np.reshape(self.x,(13,3,1,28,28))
        # self.x=self.x.reshape(13,3,1,28,28)
        # for i in range
        # each character contains 2 imgs

        print('data shape:', self.x.shape)

        # [5,2,1,28,28]
        # [1623, 20, 84, 84, 1]
        # TODO: can not shuffle here, we must keep training and test set distinct!
        # self.x_train, self.x_test = self.x, self.x
        if self.x.shape[0]>20:
            # self.x_train, self.x_test = self.x[9:], self.x[:9]
            self.x_train, self.x_test = self.x[:], self.x[:]
        else:
            self.x_train, self.x_test = self.x, self.x

        # plt.figure()
        # plt.imshow(self.x_train[0][0]/255.0)
        # plt.show()

        self.normalization()

        self.batchsz = batchsz #1
        self.n_cls = self.x.shape[0]  # 1623
        self.n_way = n_way  # n way
        self.k_shot = k_shot  # k shot
        self.k_query = k_query  # k query
        assert (k_shot + k_query) <=20

        # save pointer of current read batch in total cache
        self.indexes = {"train": 0, "test": 0}
        self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
        print("Test task DB: train", self.x_train.shape, "test", self.x_test.shape)

        self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
                               "test": self.load_data_cache(self.datasets["test"])}

        # print(len(self.datasets_cache['test'][0]),len(self.datasets_cache['test'][0][0]), self.datasets_cache['test'][0][0].shape)
        # plt.figure()
        # plt.imshow((self.datasets_cache['test'][0][0]*self.std+self.mean)/255.0)
        # plt.show()

    def get_rms(self,data):
        """
        均方根值 反映的是有效值而不是平均值
        """
        return math.sqrt(sum([x ** 2 for x in data]) / len(data))

    def expand_greyscale(self,t):
        return t.expand(3, -1, -1)
    def get_img_info(self, data_dir, char):
        data_info = list()
        data_info_sum = list()
        temp_list=list()
        temp=dict()
        temp_dict=dict()
        path_img = dict()
        j=0
        mixed_img_dir = 'MixedImg'

        type_path=os.listdir(data_dir)
        # type_path = ['Incipient Fault']
        for sub_dir in type_path:
            subtype_path = os.listdir(os.path.join(data_dir,sub_dir))
            for ssubtype_path in subtype_path:
                sssubtype_path=os.listdir(os.path.join(data_dir, sub_dir,ssubtype_path))
                for ssssubtype_path in sssubtype_path:
                    sssssubtype_path = os.listdir(os.path.join(data_dir, sub_dir, ssubtype_path,ssssubtype_path))
                    # print(sssssubtype_path)-->['current', 'voltage']
                    j = 0
                    temp_list=[] # 为了合成一张包含I、V的新图片
                    # temp_dict={}  # save both path and image_tensor, including all the images in 'cable2' dir
                    rms_list=[]
                    for ssubdir in sssssubtype_path:
                        docum_path=os.listdir(os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir))
                        docum_path.sort()
                        # docum_path = [item for item in os.listdir(os.path.join(data_dir, sub_dir, ssubtype_path, ssssubtype_path, ssubdir)) if item.endswith('.png')]
                        image_path=docum_path[:self.sample_num ]  # self.sample_num = 14
                        excel_path=docum_path[self.sample_num :]
                        for k in range(0, len(image_path)):
                            img_load = os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir,image_path[k])
                            img= Image.open(img_load)
                            img = img.convert('RGB')
                            img=self.transform(img)
                            img = img.numpy()  # image has been transformed to numpy here
                            temp_list.append(img)
                            # temp_dict.update({img_load:img})  # for record img's path ad its value
                        nn=0
                        rms_value = np.zeros(1)
                        for k in range(0, len(excel_path)):
                            nn = nn + 1
                            excel_load = os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir,excel_path[k])
                            excel_data=pd.read_excel(excel_load)
                            excel_data=np.array(excel_data)
                            data = self.get_rms(excel_data)
                            rms_value=rms_value+data
                            if nn%3==0:
                                # aa=[3,4,5]
                                rms_value=rms_value/3
                                rms_delta = rms_value - math.sqrt(2) / 2
                                rms_delta=abs(rms_delta)
                                rms_list.append(rms_delta)
                                rms_value = np.zeros(1)
                    current_rms_list=rms_list[:self.sample_num ]
                    voltage_rms_list = rms_list[self.sample_num :]
                    current_rms_summm=0
                    voltage_rms_summm = 0
                    for l1 in range(len(current_rms_list)):
                        current_rms_summm=current_rms_list[l1]+current_rms_summm

                    for l1 in range(len(voltage_rms_list)):
                        voltage_rms_summm = voltage_rms_list[l1] + voltage_rms_summm
                    # current_rms_summm=math.log(current_rms_summm)
                    # voltage_rms_summm = math.log(voltage_rms_summm)
                    lam1 = current_rms_summm / (current_rms_summm + voltage_rms_summm)
                    lam2 = voltage_rms_summm / (current_rms_summm + voltage_rms_summm)
                    j=j+1
                    mixed_temp=list()
                    mixed_temp_dict = dict()
                    if j==1:
                        #现在是5个样本 如果样本多了这里要改
                        for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13]:
                            x1 = temp_list[i]
                            x2 = temp_list[i+self.sample_num]
                            rms1 = rms_list[i]
                            rms2 = rms_list[i+self.sample_num]
                            # lam1 = rms1 / (rms1 + rms2)
                            # lam2 = rms2 / (rms1 + rms2)
                            # lam1 = 0
                            # lam2 = 1
                            #mixup
                            # alpha=1
                            # lam = np.random.beta(alpha, alpha)
                            # lam=1
                            # index = torch.randperm(x.size(0)).cuda()
                            mixed_x = lam1 * x1 + lam2 * x2  # 合并电流电压为一张图像
                            mixed_temp.append(mixed_x)
                            # TODO zyg
                            mixed_img = np.uint8((np.transpose(mixed_x, (1, 2, 0)))*255.0)
                            mixed_PIL_img = Image.fromarray(mixed_img)  # numpy 被转为图，命名为path并保存
                            save_path = os.path.join(data_dir, mixed_img_dir)
                            save_name = ''+str(ssssubtype_path)+''+str(i)+'mixed_img.jpg'
                            save_path_name = os.path.join(mixed_img_dir, save_name)
                            mixed_PIL_img.save(save_path_name)
                            path_img[save_path_name] = mixed_img # test task 的全部图和路径对应的字典
                            mixed_temp_dict[save_path_name] = mixed_img

                    label = '/'.join((sub_dir,ssubtype_path,ssssubtype_path))
                    temp[label] = mixed_temp
                    temp_dict[label] = mixed_temp_dict
            # print(temp)
            # print(temp_dict)
            # temp_dict = {'cable2':{'cable20':array,'cable21':array,...},
            #              'Disturbance':{'Disturbance0':array,'Disturbance1':array,...},
            #              'Normal':{'Normal0':array,'Normal1':array,...}}
        return temp, temp_dict


    def normalization(self):
        """
        Normalizes our data, to have a mean of 0 and sdt of 1
        """
        self.mean = np.mean(self.x_train)
        self.std = np.std(self.x_train)
        self.max = np.max(self.x_train)
        self.min = np.min(self.x_train)
        self.x_train = (self.x_train - self.mean) / self.std
        self.x_test = (self.x_test - self.mean) / self.std

        # self.mean = np.mean(self.x_train)
        # self.std = np.std(self.x_train)
        # self.max = np.max(self.x_train)
        # self.min = np.min(self.x_train)

    def denormalization(self, x):
        """
        反归一化
        """
        # x = x * self.std + self.mean
        x = x * self.std + self.mean
        return x

    def load_data_cache(self, data_pack):
        """
        Collects several batches data for N-shot learning
        :param data_pack: [cls_num, 20, 84, 84, 1]
        :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
        """
        #  take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        querysz = self.k_query * self.n_way
        data_cache = []

        # print('preload next 50 caches of batchsz of batch.')
        for sample in range(10):  # num of episodes

            x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
            for i in range(self.batchsz):  # one batch means one set

                x_spt, y_spt, x_qry, y_qry = [], [], [], []
                selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
                # selected_cls = np.array(list([0, 1, 2]))
                # selected_cls = np.array(list([0,1,2,3,4]))
                # selected_cls = np.array(list([0, 1, 2]))
                for j, cur_class in enumerate(selected_cls):

                    selected_img = np.random.choice(self.sample_num, self.k_shot + self.k_query, False)
                    # selected_img=[0,1,2]

                    # selected_img = np.array([0,1,2,3,4,5])
                    # meta-training and meta-test
                    x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
                    x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
                    y_spt.append([j for _ in range(self.k_shot)])
                    y_qry.append([j for _ in range(self.k_query)])

                # shuffle inside a batch
                # perm = np.random.permutation(self.n_way * self.k_shot)   
                x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.resize, self.resize, self.chan)
                # x_spt = np.moveaxis(x_spt,[0,1,2,3],[0,3,1,2])
                #x_spt = np.array(np.moveaxis(x_spt,[0,1,2,3,4],[0,1,4,2,3])).reshape(self.n_way * self.k_shot, self.chan, self.resize, self.resize)
                

                y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)
                # perm = np.random.permutation(self.n_way * self.k_query)
                x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.resize, self.resize, self.chan)
                y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)

                # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
                x_spts.append(x_spt)
                y_spts.append(y_spt)
                x_qrys.append(x_qry)
                y_qrys.append(y_qry)
                # print(1)


            # [b, setsz, 1, 84, 84]
            x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, self.resize, self.resize, self.chan)
            x_spts = np.transpose(x_spts, (0, 1, 4, 2, 3))
            y_spts = np.array(y_spts).astype(np.int64).reshape(self.batchsz, setsz)
            # [b, qrysz, 1, 84, 84]
            x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.resize, self.resize, self.chan)
            x_qrys = np.transpose(x_qrys, (0, 1, 4, 2, 3))
            y_qrys = np.array(y_qrys).astype(np.int64).reshape(self.batchsz, querysz)

            data_cache.append([x_spts, y_spts, x_qrys, y_qrys])

        return data_cache

    def next(self, mode='train'):
        """
        Gets next batch from the dataset with name.
        :param mode: The name of the splitting (one of "train", "val", "test")
        :return:
        """
        # update cache if indexes is larger cached num
        if self.indexes[mode] >= len(self.datasets_cache[mode]):
            self.indexes[mode] = 0
            self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])

        next_batch = self.datasets_cache[mode][self.indexes[mode]]
        self.indexes[mode] += 1

        return next_batch
# class IncipientFaultNShotMixup_test_fixlam:
#     # # 固定图像融合 加上了归一化
#
#     def __init__(self, root, batchsz, n_way, k_shot, k_query, imgsz,chan,mean,std):
#         """
#         Different from mnistNShot, the
#         :param root:
#         :param batchsz: task num
#         :param n_way:
#         :param k_shot:
#         :param k_qry:
#         :param imgsz:
#         """
#         self.sample_num = 14
#         self.resize = imgsz
#         # if not os.path.isfile(os.path.join(root, 'omniglot.npy')):
#         #     # if root/data.npy does not exist, just download it
#         #     self.x = Omniglot(root, download=True,
#         #                       transform=transforms.Compose([lambda x: Image.open(x).convert('L'),
#         #                                                     lambda x: x.resize((imgsz, imgsz)),
#         #                                                     lambda x: np.reshape(x, (imgsz, imgsz, 1)),
#         #                                                     lambda x: np.transpose(x, [2, 0, 1]),
#         #                                                     lambda x: x/255.])
#         #                       )
#         #
#         #     temp = dict()  # {label:img1, img2..., 20 imgs, label2: img1, img2,... in total, 1623 label}
#         #     for (img, label) in self.x:
#         #         if label in temp.keys():
#         #             temp[label].append(img)
#         #         else:
#         #             temp[label] = [img]
#         #
#         #     self.x = []
#         #     for label, imgs in temp.items():  # labels info deserted , each label contains 20imgs
#         #         self.x.append(np.array(imgs))
#         #
#         #     # as different class may have different number of imgs
#         #     self.x = np.array(self.x).astype(np.float)  # [[20 imgs],..., 1623 classes in total]
#         #     # each character contains 20 imgs
#         #     print('data shape:', self.x.shape)  # [1623, 20, 84, 84, 1]
#         #     temp = []  # Free memory
#         #     # save all dataset into npy file.
#         #     np.save(os.path.join(root, 'omniglot.npy'), self.x)
#         #     print('write into omniglot.npy.')
#         # else:
#         # if data.npy exists, just load it.
#         self.imgsz=imgsz
#         self.root=root
#         self.chan=chan
#         self.mean=mean
#         self.std=std
#         print('load Incipient fault')
#         self.transform = transforms.Compose([  # 多个transform操作
#             transforms.Resize(self.resize),
#             transforms.CenterCrop(self.resize),  # 从中心裁剪
#             transforms.ToTensor()
#             # transforms.Normalize(
#             #     mean=torch.tensor([0.485, 0.456, 0.406]),
#             #     std=torch.tensor([0.229, 0.224, 0.225]))
#             # transforms.RandomVerticalFlip(p=0.2),
#             #
#             # transforms.Lambda(self.expand_greyscale)  # 自行定义transform操作
#
#         ])
#         # self.x = os.listdir(os.path.join(self.root))
#         temp = self.get_img_info(self.root, '.png')
#         self.x = []
#         # xt=np.zeros((1,3,1,28,28))
#         for label, imgs in temp.items():
#             # self.x.append(np.array(imgs))
#             imgs=np.array(imgs)
#             # imgs=imgs.numpy()
#             # imgs=np.expand_dims(imgs,axis=0)
#             # imgs=imgs.reshape(1,3,1,28,28)
#             self.x.append(imgs)
#             # self.xt=np.append(self.xt,imgs,0)
#             # xt = np.concatenate([xt, imgs], axis=0)
#             # print(1)
#         # a=tempp.shape
#         # as different class may have different number of imgs
#         # len_x = len(self.x)
#         # self.x_1=np.zeros([len_x,3,1,28,28])
#         # for i in range(len_x):
#         #     self.x_1[i,:,:,:,:]=self.x[i]
#         self.x = np.array(self.x).astype(np.float)
#         # self.x=np.reshape(self.x,(13,3,1,28,28))
#         # self.x_fault = np.array(self.x_fault)
#         # self.x = np.array(self.x).astype(np.float)
#         # self.x=np.reshape(self.x,(13,3,1,28,28))
#         # self.x=self.x.reshape(13,3,1,28,28)
#         # for i in range
#         # each character contains 2 imgs
#
#         print('data shape:', self.x.shape)
#
#         # [5,2,1,28,28]
#         # [1623, 20, 84, 84, 1]
#         # TODO: can not shuffle here, we must keep training and test set distinct!
#         # self.x_train, self.x_test = self.x, self.x
#         if self.x.shape[0]>20:
#             # self.x_train, self.x_test = self.x[9:], self.x[:9]
#             self.x_train, self.x_test = self.x[:], self.x[:]
#         else:
#             self.x_train, self.x_test = self.x, self.x
#
#         self.normalization()
#
#         self.batchsz = batchsz #1
#         self.n_cls = self.x.shape[0]  # 1623
#         self.n_way = n_way  # n way
#         self.k_shot = k_shot  # k shot
#         self.k_query = k_query  # k query
#         assert (k_shot + k_query) <=20
#
#         # save pointer of current read batch in total cache
#         self.indexes = {"train": 0, "test": 0}
#         self.datasets = {"train": self.x_train, "test": self.x_test}  # original data cached
#         print("DB: train", self.x_train.shape, "test", self.x_test.shape)
#
#         self.datasets_cache = {"train": self.load_data_cache(self.datasets["train"]),  # current epoch data cached
#                                "test": self.load_data_cache(self.datasets["test"])}
#
#     def get_rms(self,data):
#         """
#         均方根值 反映的是有效值而不是平均值
#         """
#         return math.sqrt(sum([x ** 2 for x in data]) / len(data))
#
#     def expand_greyscale(self,t):
#         return t.expand(3, -1, -1)
#
#     # def get_img_info(self, data_dir, char):
#     #     data_info = list()
#     #     data_info_sum = list()
#     #     temp_list=list()
#     #     temp=dict()
#     #     j=0
#     #     type_path=os.listdir(data_dir)
#     #     for sub_dir in type_path:
#     #         subtype_path = os.listdir(os.path.join(data_dir,sub_dir))
#     #         j = 0
#     #         temp_list=[]
#     #         rms_list=[]
#     #         for ssubdir in subtype_path:
#     #             docum_path=os.listdir(os.path.join(data_dir, sub_dir,ssubdir))
#     #             image_path=docum_path[:6]
#     #             excel_path=docum_path[6:]
#     #             for k in range(0, len(image_path)):
#     #                 img_load = os.path.join(data_dir, sub_dir,ssubdir, image_path[k])
#     #                 img= Image.open(img_load)
#     #                 img = img.convert('RGB')
#     #                 img=self.transform(img)
#     #                 img = img.numpy()
#     #                 temp_list.append(img)
#     #             for k in range(0, len(excel_path)):
#     #                 excel_load = os.path.join(data_dir, sub_dir, ssubdir, excel_path[k])
#     #                 excel_data=pd.read_excel(excel_load)
#     #                 excel_data=np.array(excel_data)
#     #                 # aa=[3,4,5]
#     #                 rms_value=np.zeros(1)
#     #                 for l in range(3):
#     #                     data=self.get_rms(excel_data[:,l])
#     #                     rms_value=rms_value+data
#     #                 rms_value=rms_value/3
#     #                 rms_delta=rms_value-math.sqrt(2)/2
#     #                 rms_list.append(rms_delta)
#     #         j=j+1
#     #         mixed_temp=list()
#     #         if j==1:
#     #             #现在是3个样本 如果样本多了这里要改
#     #             for i in [0,1,2,3,4,5]:
#     #                 x1=temp_list[i]
#     #                 x2 = temp_list[i+6]
#     #                 rms1=rms_list[i]
#     #                 rms2 = rms_list[i+6]
#     #                 lam1 = rms1 / (rms1 + rms2)
#     #                 lam2 = rms2 / (rms1 + rms2)
#     #                 #mixup
#     #                 # alpha=1
#     #                 # lam = np.random.beta(alpha, alpha)
#     #                 # lam=1
#     #                 # index = torch.randperm(x.size(0)).cuda()
#     #                 mixed_x = lam1 * x1 +  lam2 * x2
#     #                 mixed_temp.append(mixed_x)
#     #         label = sub_dir
#     #         temp[label] = mixed_temp
#     #     return temp
#
#     def get_img_info(self, data_dir, char):
#         data_info = list()
#         data_info_sum = list()
#         temp_list=list()
#         temp=dict()
#         j=0
#
#         type_path=os.listdir(data_dir)
#         for sub_dir in type_path:
#             subtype_path = os.listdir(os.path.join(data_dir,sub_dir))
#             for ssubtype_path in subtype_path:
#                 sssubtype_path=os.listdir(os.path.join(data_dir, sub_dir,ssubtype_path))
#                 for ssssubtype_path in sssubtype_path:
#                     sssssubtype_path = os.listdir(os.path.join(data_dir, sub_dir, ssubtype_path,ssssubtype_path))
#                     j = 0
#                     temp_list=[]
#                     rms_list=[]
#                     for ssubdir in sssssubtype_path:
#                         docum_path=os.listdir(os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir))
#                         image_path=docum_path[:self.sample_num ]
#                         excel_path=docum_path[self.sample_num :]
#                         for k in range(0, len(image_path)):
#                             img_load = os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir,image_path[k])
#                             img= Image.open(img_load)
#                             img = img.convert('RGB')
#                             img=self.transform(img)
#                             img = img.numpy()
#                             temp_list.append(img)
#                         nn=0
#                         rms_value = np.zeros(1)
#                         for k in range(0, len(excel_path)):
#                             nn = nn + 1
#                             excel_load = os.path.join(data_dir, sub_dir,ssubtype_path,ssssubtype_path,ssubdir,excel_path[k])
#                             excel_data=pd.read_excel(excel_load)
#                             excel_data=np.array(excel_data)
#                             data = self.get_rms(excel_data)
#                             rms_value=rms_value+data
#                             if nn%3==0:
#                                 # aa=[3,4,5]
#                                 rms_value=rms_value/3
#                                 rms_delta = rms_value - math.sqrt(2) / 2
#                                 rms_delta=abs(rms_delta)
#                                 rms_list.append(rms_delta)
#                                 rms_value = np.zeros(1)
#                     current_rms_list=rms_list[:self.sample_num ]
#                     voltage_rms_list = rms_list[self.sample_num :]
#                     current_rms_summm=0
#                     voltage_rms_summm = 0
#                     for l1 in range(len(current_rms_list)):
#                         current_rms_summm=current_rms_list[l1]+current_rms_summm
#
#                     for l1 in range(len(voltage_rms_list)):
#                         voltage_rms_summm = voltage_rms_list[l1] + voltage_rms_summm
#                     lam1 = current_rms_summm / (current_rms_summm + voltage_rms_summm)
#                     lam2 = voltage_rms_summm / (current_rms_summm + voltage_rms_summm)
#
#                     j=j+1
#                     mixed_temp=list()
#                     if j==1:
#                         #现在是5个样本 如果样本多了这里要改
#                         for i in [0,1,2,3,4,5,6,7,8,9,10,11,12,13]:
#                             x1=temp_list[i]
#                             x2 = temp_list[i+self.sample_num ]
#                             rms1 = rms_list[i]
#                             rms2 = rms_list[i+self.sample_num ]
#                             # lam1 = rms1 / (rms1 + rms2)
#                             # lam2 = rms2 / (rms1 + rms2)
#                             lam1 = 0.1
#                             lam2 = 0.9
#                             #mixup
#                             # alpha=1
#                             # lam = np.random.beta(alpha, alpha)
#                             # lam=1
#                             # index = torch.randperm(x.size(0)).cuda()
#                             mixed_x = lam1 * x1 + lam2 * x2
#                             mixed_temp.append(mixed_x)
#                     label = sub_dir+ssubtype_path+ssssubtype_path
#                     temp[label] = mixed_temp
#         return temp
#
#
#     def normalization(self):
#         """
#         Normalizes our data, to have a mean of 0 and sdt of 1
#         """
#         # self.mean = np.mean(self.x_train)
#         # self.std = np.std(self.x_train)
#         # self.max = np.max(self.x_train)
#         # self.min = np.min(self.x_train)
#         # print("before norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
#         # self.x_train = (self.x_train - self.mean) / self.std
#         self.x_test = (self.x_test - self.mean) / self.std
#
#         # self.mean = np.mean(self.x_train)
#         # self.std = np.std(self.x_train)
#         # self.max = np.max(self.x_train)
#         # self.min = np.min(self.x_train)
#
#     # print("after norm:", "mean", self.mean, "max", self.max, "min", self.min, "std", self.std)
#
#     def load_data_cache(self, data_pack):
#         """
#         Collects several batches data for N-shot learning
#         :param data_pack: [cls_num, 20, 84, 84, 1]
#         :return: A list with [support_set_x, support_set_y, target_x, target_y] ready to be fed to our networks
#         """
#         #  take 5 way 1 shot as example: 5 * 1
#         setsz = self.k_shot * self.n_way
#         querysz = self.k_query * self.n_way
#         data_cache = []
#
#         # print('preload next 50 caches of batchsz of batch.')
#         for sample in range(10):  # num of episodes
#
#             x_spts, y_spts, x_qrys, y_qrys = [], [], [], []
#             for i in range(self.batchsz):  # one batch means one set
#
#                 x_spt, y_spt, x_qry, y_qry = [], [], [], []
#                 selected_cls = np.random.choice(data_pack.shape[0], self.n_way, False)
#                 # selected_cls = np.array(list([0,1,2,3,4]))
#                 # selected_cls = np.array(list([0, 1, 2]))
#                 for j, cur_class in enumerate(selected_cls):
#
#                     selected_img = np.random.choice(self.sample_num, self.k_shot + self.k_query, False)
#                     # selected_img=[0,1,2]
#
#                     # selected_img = np.array([0,1,2,3,4,5])
#                     # meta-training and meta-test
#                     x_spt.append(data_pack[cur_class][selected_img[:self.k_shot]])
#                     x_qry.append(data_pack[cur_class][selected_img[self.k_shot:]])
#                     y_spt.append([j for _ in range(self.k_shot)])
#                     y_qry.append([j for _ in range(self.k_query)])
#
#                 # shuffle inside a batch
#                 # perm = np.random.permutation(self.n_way * self.k_shot)
#                 x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.chan, self.resize, self.resize)
#                 y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)
#                 # perm = np.random.permutation(self.n_way * self.k_query)
#                 x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.chan, self.resize, self.resize)
#                 y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)
#
#                 # append [sptsz, 1, 84, 84] => [b, setsz, 1, 84, 84]
#                 x_spts.append(x_spt)
#                 y_spts.append(y_spt)
#                 x_qrys.append(x_qry)
#                 y_qrys.append(y_qry)
#                 # print(1)
#
#
#             # [b, setsz, 1, 84, 84]
#             x_spts = np.array(x_spts).astype(np.float32).reshape(self.batchsz, setsz, self.chan, self.resize, self.resize)
#             y_spts = np.array(y_spts).astype(np.int).reshape(self.batchsz, setsz)
#             # [b, qrysz, 1, 84, 84]
#             x_qrys = np.array(x_qrys).astype(np.float32).reshape(self.batchsz, querysz, self.chan, self.resize, self.resize)
#             y_qrys = np.array(y_qrys).astype(np.int).reshape(self.batchsz, querysz)
#
#             data_cache.append([x_spts, y_spts, x_qrys, y_qrys])
#
#         return data_cache
#
#     def next(self, mode='train'):
#         """
#         Gets next batch from the dataset with name.
#         :param mode: The name of the splitting (one of "train", "val", "test")
#         :return:
#         """
#         # update cache if indexes is larger cached num
#         if self.indexes[mode] >= len(self.datasets_cache[mode]):
#             self.indexes[mode] = 0
#             self.datasets_cache[mode] = self.load_data_cache(self.datasets[mode])
#
#         next_batch = self.datasets_cache[mode][self.indexes[mode]]
#         self.indexes[mode] += 1
#
#         return next_batch
