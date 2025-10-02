import torch
import torch.nn as nn
import torch.nn.functional as F

class DigitClassifier(nn.Module):
    """手写数字分类器神经网络"""
    
    def __init__(self):
        super(DigitClassifier, self).__init__()
        # 第一个卷积层：输入1通道(灰度图)，输出32通道
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 第二个卷积层：32通道 -> 64通道
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 丢弃层，防止过拟合
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 经过两次池化后：28x28 -> 14x14 -> 7x7
        self.fc2 = nn.Linear(128, 10)  # 输出10个类别（0-9）
    
    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout1(x)
        
        # 第二个卷积块
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout1(x)
        
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x

def create_model(device='cuda'):
    """创建并返回模型"""
    model = DigitClassifier().to(device)
    print("✅ 模型创建成功！")
    print(f"📊 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"🎯 使用设备: {device}")
    return model

if __name__ == "__main__":
    # 测试模型
    model = create_model()
    print(model)
