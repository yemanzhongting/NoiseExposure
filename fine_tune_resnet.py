import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
df=pd.read_csv('merge_data.csv')

df['label'] = pd.factorize(df['predict'])[0]

import os
path='D:\pics_one'
# id  pic_date_y ponaid .jpg
def get_path(id, pic_date_y,ponaid):
    img=str(id)+'_'+str(pic_date_y)+'_'+str(ponaid)+'.jpg'
    return os.path.join(path,img)

df['image_path'] = df.apply(lambda x: get_path(x['point_id'], x['pic_date_y'],x['ponaid']), axis=1)   
# 检查图片文件是否存在，如果不存在则标记为False
df['img_exists'] = df['image_path'].apply(os.path.exists)
# 仅保留存在的图片对应的行
df_filtered = df[df['img_exists']].drop('img_exists', axis=1)
# 现在df_filtered将仅包含那些图片文件实际存在的行
print(df_filtered.shape)

tmp=df_filtered[['new_label','image_path']]

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 1]  # 图片路径
        image = Image.open(img_path).convert('RGB')
        label = self.dataframe.iloc[idx, 0]  # 标签
        if self.transform:
            image = self.transform(image)
        return image, label

# 数据转换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 创建数据集和数据加载器
dataset = CustomDataset(dataframe=tmp, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 加载预训练的ResNet模型
model = models.resnet18(pretrained=True)

# 修改分类器
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)  # 假设有4个类别

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 检查GPU是否可用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

import copy
# 验证和保存最佳模型
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

# 训练模型
num_epochs=3
for epoch in range(num_epochs):  # num_epochs是您希望训练的周期数
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 验证过程
    model.eval()  # 设置模型为评估模式
    running_corrects = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():  # 在评估时不计算梯度
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    epoch_acc = running_corrects.double() / len(dataloader.dataset)

    # 打印统计信息
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    print('Accuracy: {:.4f}'.format(epoch_acc))

    # 深拷贝模型
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

print('Best val Acc: {:4f}'.format(best_acc))

# 载入最佳模型权重
model.load_state_dict(best_model_wts)

# 保存模型
torch.save(model.state_dict(), 'best_model2.pth')

####
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image, normalize
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

# 加载训练好的模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 修改为二类分类
model.load_state_dict(torch.load('best_model2.pth'))
model.eval()  # 设为评估模式

# 选择一个目标层（通常是最后一个卷积层）
target_layer = model.layer4[-1]

# 实例化CAM提取器
cam_extractor = SmoothGradCAMpp(model, target_layer)
# 加载并处理一个图像
img_path = r'D:\pics_one\3_201507_09000200011507220420201096I.jpg'
# D:\\pics_one\\2684_202106_0900020012210605151814910HS.jpg' 工业噪音
#  'D:\\pics_one\\2040_201903_09002200121903071501596637P.jpg' 生活噪音
# tmp['image_path'].values.tolist()[ 276] 几栋楼，但是是交通噪音
# D:\\pics_one\\1858_201905_09000200121905111512149302P.jpg 生活噪音（窗户
#'
img_path =tmp['image_path'].values.tolist()[ 276]
img_path = 'D:\\pics_one\\1858_201905_09000200121905111512149302P.jpg'
original_img = Image.open(img_path).convert('RGB')
original_size = original_img.size  # 获取原始图像大小
input_tensor = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])(original_img).unsqueeze(0)  # 添加一个batch维度
# 预测类别得分
out = model(input_tensor)
print(out)
# 得到CAM并转换为PIL图像
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
cam_image = to_pil_image(activation_map[0], mode='F').resize(original_size)  # 使用原始图像大小
# 叠加原始图像和CAM
result = overlay_mask(original_img, cam_image, alpha=0.5)
# 显示结果
result.save('窗户噪音.jpg')
result.show()

from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import to_pil_image, normalize
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask

# 加载训练好的模型
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)  # 修改为二类分类
model.load_state_dict(torch.load('best_model2.pth'))
model.eval()  # 设为评估模式

# 选择一个目标层（通常是最后一个卷积层）
target_layer = model.layer4[-1]

# 实例化CAM提取器
cam_extractor = SmoothGradCAMpp(model, target_layer)
# 加载并处理一个图像

# D:\\pics_one\\2684_202106_0900020012210605151814910HS.jpg' 工业噪音
#  'D:\\pics_one\\2040_201903_09002200121903071501596637P.jpg' 生活噪音
# tmp['image_path'].values.tolist()[ 276] 几栋楼，但是是交通噪音
# D:\\pics_one\\1858_201905_09000200121905111512149302P.jpg 生活噪音（窗户
#'

img_path = 'D:\\pics_one\\1858_201905_09000200121905111512149302P.jpg'
img_path = r'C:\Users\20143\Downloads\noise.jpg'#SVI

for i in tmp['image_path'].values.tolist():
    img_path=i
    original_img = Image.open(img_path).convert('RGB')
    original_size = original_img.size  # 获取原始图像大小
    input_tensor = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(original_img).unsqueeze(0)  # 添加一个batch维度
    # 预测类别得分
    out = model(input_tensor)
    _, preds = torch.max(out, 1)
    if preds!=pred_tmp:
        print(i,out,preds, _)
