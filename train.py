import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from model import create_model
import time
import os

class Trainer:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = create_model(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.7)
        
        # 创建保存目录
        os.makedirs('checkpoints', exist_ok=True)
        
    def load_data(self):
        """加载MNIST数据集"""
        print("📥 正在下载MNIST数据集...")
        
        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # 下载训练集和测试集
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        
        # 创建数据加载器
        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2)
        
        print(f"✅ 数据加载完成！")
        print(f"📊 训练样本: {len(train_dataset):,}")
        print(f"📊 测试样本: {len(test_dataset):,}")
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'训练周期: {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)}'
                      f' ({100. * batch_idx / len(self.train_loader):.0f}%)]'
                      f'\t损失: {loss.item():.6f}')
        
        accuracy = 100. * correct / total
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss, accuracy
    
    def test(self):
        """测试模型性能"""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(self.test_loader)
        return avg_loss, accuracy
    
    def train(self, epochs=10):
        """完整的训练过程"""
        print("🚀 开始训练...")
        self.load_data()
        
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        
        start_time = time.time()
        
        for epoch in range(1, epochs + 1):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # 测试
            test_loss, test_acc = self.test()
            test_accuracies.append(test_acc)
            
            # 更新学习率
            self.scheduler.step()
            
            print(f'\n📈 Epoch {epoch} 结果:')
            print(f'   训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
            print(f'   测试损失: {test_loss:.4f}, 测试准确率: {test_acc:.2f}%')
            print(f'   学习率: {self.scheduler.get_last_lr()[0]:.6f}')
            
            # 保存模型
            if epoch % 5 == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'test_accuracy': test_acc
                }
                torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pth')
                print(f'💾 模型已保存: checkpoints/model_epoch_{epoch}.pth')
        
        training_time = time.time() - start_time
        print(f'\n✅ 训练完成！总用时: {training_time:.2f}秒')
        
        # 保存最终模型
        torch.save(self.model.state_dict(), 'checkpoints/final_model.pth')
        print('💾 最终模型已保存: checkpoints/final_model.pth')
        
        # 绘制训练曲线
        self.plot_training_curve(train_losses, train_accuracies, test_accuracies)
        
        return max(test_accuracies)
    
    def plot_training_curve(self, train_losses, train_accuracies, test_accuracies):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, 'b-', label='训练损失')
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, 'g-', label='训练准确率')
        plt.plot(test_accuracies, 'r-', label='测试准确率')
        plt.title('准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curve.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # 开始训练！
    print("=" * 60)
    print("🎯 PyTorch 手写数字识别项目")
    print("=" * 60)
    
    trainer = Trainer()
    best_accuracy = trainer.train(epochs=10)
    
    print(f"\n🎉 最佳测试准确率: {best_accuracy:.2f}%")
    print("项目完成！你可以运行 test.py 来测试模型性能。")