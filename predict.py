import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from model import create_model
import numpy as np

class Predictor:
    def __init__(self, model_path='checkpoints/final_model.pth', device='cuda'):
        self.device = device
        self.model = create_model(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()
        
        # 数据预处理（与训练时相同）
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        print("✅ 预测器准备就绪！")
    
    def predict_image(self, image_path):
        """预测单张图片"""
        # 加载并预处理图片
        image = Image.open(image_path).convert('L')  # 转为灰度图
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = output.argmax(dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 显示结果
        plt.figure(figsize=(8, 4))
        
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title(f'输入图片\n预测: {predicted_class} (置信度: {confidence:.2%})')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        classes = list(range(10))
        probs = probabilities[0].cpu().numpy()
        plt.bar(classes, probs, color='skyblue')
        plt.xlabel('数字')
        plt.ylabel('概率')
        plt.title('预测概率分布')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return predicted_class, confidence

def create_sample_images():
    """创建一些示例手写数字图片（用于测试）"""
    samples = []
    
    for digit in range(10):
        # 创建一个28x28的黑色背景图片
        img = Image.new('L', (28, 28), color=0)
        draw = ImageDraw.Draw(img)
        
        # 简单绘制数字（实际应用中应该用真实手写图片）
        # 这里只是示例，你可以用自己的手写图片替换
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        # 在图片中心绘制数字
        bbox = draw.textbbox((0, 0), str(digit), font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        x = (28 - text_width) // 2
        y = (28 - text_height) // 2
        
        draw.text((x, y), str(digit), fill=255, font=font)
        samples.append(img)
    
    return samples

if __name__ == "__main__":
    print("🔮 手写数字预测器")
    print("=" * 40)
    
    predictor = Predictor()
    
    # 创建示例图片并预测
    print("创建示例图片...")
    sample_images = create_sample_images()
    
    for i, img in enumerate(sample_images):
        # 保存临时图片
        img_path = f'temp_digit_{i}.png'
        img.save(img_path)
        
        # 预测
        predicted, confidence = predictor.predict_image(img_path)
        print(f"数字 {i} -> 预测: {predicted}, 置信度: {confidence:.2%}")