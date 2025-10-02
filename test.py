import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from model import create_model

class Tester:
    def __init__(self, model_path='checkpoints/final_model.pth', device='cuda'):
        self.device = device
        self.model = create_model(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        print(f"✅ 模型加载成功: {model_path}")
    
    def load_data(self):
        """加载测试数据"""
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        return test_dataset
    
    def evaluate(self):
        """全面评估模型性能"""
        test_dataset = self.load_data()
        
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = 100. * correct / total
        print(f"\n📊 模型评估结果:")
        print(f"   测试样本数: {total:,}")
        print(f"   正确预测数: {correct:,}")
        print(f"   准确率: {accuracy:.2f}%")
        
        return all_predictions, all_targets, accuracy
    
    def show_examples(self, num_examples=10):
        """显示一些预测例子"""
        test_dataset = self.load_data()
        
        plt.figure(figsize=(12, 6))
        indices = np.random.choice(len(test_dataset), num_examples, replace=False)
        
        for i, idx in enumerate(indices):
            image, true_label = test_dataset[idx]
            
            # 预测
            with torch.no_grad():
                output = self.model(image.unsqueeze(0).to(self.device))
                pred_label = output.argmax(dim=1).item()
            
            plt.subplot(2, 5, i+1)
            plt.imshow(image.squeeze(), cmap='gray')
            color = 'green' if pred_label == true_label else 'red'
            plt.title(f'True: {true_label}, Pred: {pred_label}', color=color)
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('prediction_examples.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    print("🧪 模型测试开始...")
    
    tester = Tester()
    
    # 评估模型
    predictions, targets, accuracy = tester.evaluate()
    
    # 显示预测例子
    tester.show_examples()
    
    print(f"\n🎯 最终测试准确率: {accuracy:.2f}%")