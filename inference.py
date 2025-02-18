# %%
import pickle
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.modules.container import ParameterList
from meta import Meta
import  argparse
import matplotlib.pyplot as plt
import shap

# %% [markdown]
# # Set parameters

# %%
argparser = argparse.ArgumentParser()
# argparser.add_argument('--epoch', type=int, help='epoch number', default=40000)
# argparser.add_argument('--n_way', type=int, help='n way', default=5)
# argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=32)
argparser.add_argument('--epoch', type=int, help='epoch number', default=200) #我设置的是1000
argparser.add_argument('--n_way', type=int, help='n way', default=3) #5
argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=3)
argparser.add_argument('--k_spt_test', type=int, help='k shot for support set', default=4)
argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=2) #15
argparser.add_argument('--k_qry_test', type=int, help='k shot for query set', default=10)  # 10
argparser.add_argument('--imgsz', type=int, help='imgsize', default=256)#28 64
argparser.add_argument('--imgc', type=int, help='imgchannels', default=3)
argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=4)
argparser.add_argument('--task_num_test', type=int, help='meta batch size, namely task num', default=1)
argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.1)#0.4
argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=5)
argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=10)#10
# args = argparser.parse_args()
# Filter out unrecognized arguments
args, unknown = argparser.parse_known_args()

# %%
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
    # ****************6层卷积层参数设置256*********************
]

# %%
# Allowlist the ParameterList class
device = torch.device('cuda')
maml = Meta(args, config).to(device)
fw = torch.load("net_vars.pth", weights_only=False, map_location=device)

# %%
batchsz=args.task_num
n_way=args.n_way
k_shot=args.k_spt
k_query=args.k_qry
imgsz=args.imgsz
chan=args.imgc

sample_num = 14
resize = imgsz

# %% [markdown]
# # Load data

# %%
with open('db_test_temp_dict.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)

# %%
loaded_dict

# %%
x = []
indices = []
labels = ['Normal', 'Disturbance', 'cable2']
mapping = {'Normal': 0, 'Disturbance': 1, 'cable2': 2}
for label, imgs_dict in loaded_dict.items():
    # self.x.append(np.array(imgs))
    imgs = np.array([list(item) for item in imgs_dict.values()]) # zyg
    x.append(imgs)
    label_name = label.split('/')[-1]

    for name in imgs_dict.keys():
        indices.append(mapping[label_name])
indices = np.array(indices)
print(indices)

# %%
x = np.array(x).astype(np.float32)
print('data shape:', x.shape)
if x.shape[0]>20:
    x_train, x_test = x[:], x[:]
else:
    x_train, x_test = x, x

# %%
x_test = (x_test - 2.9900356e-09) / 1.0
x_test.shape

# %%
x_test = np.moveaxis(x_test, -1, 2)
x_test.shape

# %%
x_test_reshaped = torch.tensor(x_test.reshape(-1, chan, resize, resize)).to(device)
x_test_reshaped.shape

# %%
# image = x_test_reshaped[2].cpu().numpy()
# plt.imshow(image.transpose(1, 2, 0)/255.0)
# plt.axis('off')
# plt.show()

# %% [markdown]
# # Test inference

# %%
# result_pred = maml.net(x_test_reshaped, fw, bn_training=False)
result_pred = maml.net(x_test_reshaped, fw, bn_training=False)
print(result_pred)

# %% [markdown]
# ## Convert logits to probabilities

# %%
result_prob = F.softmax(result_pred, dim=1)
result_indices = np.argmax(result_prob.cpu().detach().numpy(), axis=1)

# %%
result_indices

# %%
indices == result_indices

# %% [markdown]
# # Explain a single image

# %%
img_ID = 0
input_image = x_test_reshaped[img_ID].permute(1, 2, 0)
input_label = labels[indices[img_ID]]
input_indices = indices[img_ID]
print(f"{input_label}: {input_indices}")

# %% [markdown]
# ## Display input image

# %%
plt.imshow(input_image.cpu().numpy()/255.0)
plt.title(f"{input_label}: {input_indices}")
plt.axis('off')
plt.show()

# %%
def predict(image):
    # Convert numpy array to Tensor
    if isinstance(image, np.ndarray):
        image = torch.from_numpy(image).float()
    
    # Ensure the image is on the correct device (e.g., GPU if available)
    image = image.to(device)
    
    return F.softmax(maml.net(image.permute(0, -1, 1, 2), fw, bn_training=True), dim=1)

# %%
input_image = input_image.unsqueeze(0)
predict(input_image)

# %%
batch_size = 50
n_evals = 5000 # 迭代次数越大，显著性分析粒度越精细，计算消耗时间越长
# 定义 mask，遮盖输入图像上的局部区域
masker_blur = shap.maskers.Image("blur(64, 64)", input_image[0].shape)
# 创建可解释分析算法
explainer = shap.Explainer(predict, masker_blur, output_names=labels)

# %%
shap_values = explainer(input_image, max_evals=n_evals, batch_size=batch_size, outputs=[0,1,2])

# %%
# 整理张量维度
shap_values.data = (shap_values.data).cpu().numpy()[0]/255 # 原图
shap_values.values = [val for val in np.moveaxis(shap_values.values[0],-1, 0)] # shap值热力图

# %%
# 可视化
shap.image_plot(shap_values=shap_values.values,
                pixel_values=shap_values.data,
                labels=shap_values.output_names)

# %% [markdown]
# # Explain all images

# %%
for img_ID in range(42):
    input_image = x_test_reshaped[img_ID].permute(1, 2, 0)
    input_label = labels[indices[img_ID]]
    input_indices = indices[img_ID]
    print(f"{input_label}: {input_indices}")

    input_image = input_image.unsqueeze(0)
    print(predict(input_image))

    batch_size = 50
    n_evals = 5000 # 迭代次数越大，显著性分析粒度越精细，计算消耗时间越长
    # 定义 mask，遮盖输入图像上的局部区域
    masker_blur = shap.maskers.Image("blur(32, 32)", input_image[0].shape)
    # 创建可解释分析算法
    explainer = shap.Explainer(predict, masker_blur, output_names=labels)

    shap_values = explainer(input_image, max_evals=n_evals, batch_size=batch_size, outputs=[0,1,2])

    # 整理张量维度
    shap_values.data = (shap_values.data).cpu().numpy()[0]/255 # 原图
    shap_values.values = [val for val in np.moveaxis(shap_values.values[0],-1, 0)] # shap值热力图

    # 可视化
    shap.image_plot(shap_values=shap_values.values,
                    pixel_values=shap_values.data,
                    labels=shap_values.output_names)

# %% [markdown]
# ---

# %%
x_test_reshaped.shape

# %%
with torch.no_grad():
    print(maml.net(x_test_reshaped[0:10], fw, bn_training=True))

# %%
with torch.no_grad():
    print(maml.net(x_test_reshaped[0:2], fw, bn_training=True))


