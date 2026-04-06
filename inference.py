# coding=utf-8
"""
FPViT 推理脚本
用于对单张或批量图像进行推理预测，并打印置信度等信息
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# 确保可以导入 models 模块
current_file = Path(__file__).resolve()
project_root = current_file.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from models.foveated_vit import FoveatedViT, get_foveated_b16_full_config


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# 类别名称
CLASSES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']


def get_transform(img_size=224):
    """获取图像预处理transform"""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def load_model(checkpoint_path, device):
    """加载训练好的模型"""
    from models.foveated_vit import FoveatedViT, get_foveated_b16_full_config

    config = get_foveated_b16_full_config()
    model = FoveatedViT(
        config=config,
        img_size=224,
        num_classes=len(CLASSES),
        zero_head=True
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        logger.info(f"Checkpoint accuracy: {checkpoint.get('accuracy', 'unknown'):.4f}" if 'accuracy' in checkpoint else "")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def predict_single_image(model, image_path, device, img_size=224):
    """对单张图像进行推理"""
    transform = get_transform(img_size)

    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        logits, _ = model(image_tensor)

    # 计算概率
    probs = torch.softmax(logits, dim=1)[0]
    conf, pred_idx = torch.max(probs, dim=0)

    return {
        'image_path': image_path,
        'predicted_class': CLASSES[pred_idx.item()],
        'confidence': conf.item(),
        'probabilities': {CLASSES[i]: probs[i].item() for i in range(len(CLASSES))},
        'all_logits': logits[0].cpu().numpy()
    }


def predict_batch(model, image_paths, device, img_size=224, batch_size=16):
    """批量推理"""
    transform = get_transform(img_size)

    results = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []

        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            batch_tensors.append(transform(image))

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            logits, _ = model(batch)

        probs = torch.softmax(logits, dim=1)

        for j, path in enumerate(batch_paths):
            conf, pred_idx = torch.max(probs[j], dim=0)
            results.append({
                'image_path': path,
                'predicted_class': CLASSES[pred_idx.item()],
                'confidence': conf.item(),
                'probabilities': {CLASSES[k]: probs[j][k].item() for k in range(len(CLASSES))},
                'all_logits': logits[j].cpu().numpy()
            })

    return results


def print_result(result, show_all_probs=True):
    """打印单个预测结果"""
    print(f"\n{'='*60}")
    print(f"图像: {result['image_path']}")
    print(f"{'='*60}")
    print(f"预测类别: {result['predicted_class']}")
    print(f"置信度: {result['confidence']*100:.2f}%")

    if show_all_probs:
        print(f"\n各类别概率:")
        for cls, prob in result['probabilities'].items():
            bar_length = int(prob * 40)
            bar = '█' * bar_length + '░' * (40 - bar_length)
            print(f"  {cls:10s}: {prob*100:6.2f}% |{bar}|")
    print()


def print_summary(results):
    """打印预测结果汇总"""
    print(f"\n{'='*60}")
    print(f"预测汇总")
    print(f"{'='*60}")
    print(f"总图像数: {len(results)}")

    class_counts = {}
    for r in results:
        cls = r['predicted_class']
        class_counts[cls] = class_counts.get(cls, 0) + 1

    print(f"\n各类别预测数量:")
    for cls in CLASSES:
        count = class_counts.get(cls, 0)
        print(f"  {cls:10s}: {count}")

    avg_conf = np.mean([r['confidence'] for r in results])
    print(f"\n平均置信度: {avg_conf*100:.2f}%")

    # 找出置信度最低的样本（可能更容易出错）
    sorted_results = sorted(results, key=lambda x: x['confidence'])
    print(f"\n置信度最低的5张图像:")
    for r in sorted_results[:5]:
        print(f"  {os.path.basename(r['image_path']):30s} -> {r['predicted_class']:10s} (conf: {r['confidence']*100:.2f}%)")

    print()


def main():
    parser = argparse.ArgumentParser(description='FPViT 推理脚本')

    # 模型路径
    parser.add_argument('--checkpoint', type=str, default='output/best_model.pt',
                       help='模型checkpoint路径')
    parser.add_argument('--img_dir', type=str, default='img',
                       help='测试图像目录')
    parser.add_argument('--img_file', type=str, default=None,
                       help='单张图像路径（优先于img_dir）')

    # 推理参数
    parser.add_argument('--img_size', type=int, default=224,
                       help='图像大小')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批量推理时的batch size')
    parser.add_argument('--show_all_probs', action='store_true', default=True,
                       help='是否显示所有类别的概率')

    # 设备
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cuda/cpu), 默认为自动检测')

    args = parser.parse_args()

    # 设置设备
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info(f"使用设备: {device}")

    # 加载模型
    logger.info(f"加载模型: {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # 获取图像路径
    if args.img_file:
        if not os.path.exists(args.img_file):
            logger.error(f"图像文件不存在: {args.img_file}")
            return
        image_paths = [args.img_file]
    else:
        img_dir = Path(args.img_dir)
        if not img_dir.exists():
            logger.error(f"图像目录不存在: {args.img_dir}")
            return

        # 注意: Windows文件系统不区分大小写，*.jpg和*.JPG会匹配相同文件
        # 因此只使用小写扩展名
        supported_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_paths = []
        seen = set()  # 用于去重
        for ext in supported_exts:
            for p in img_dir.glob(f'*{ext}'):
                # 使用绝对路径去重
                abs_path = str(p.resolve())
                if abs_path not in seen:
                    seen.add(abs_path)
                    image_paths.append(abs_path)

        if not image_paths:
            logger.error(f"在 {args.img_dir} 中未找到图像文件")
            return

    logger.info(f"找到 {len(image_paths)} 张图像")

    # 推理
    if len(image_paths) == 1:
        result = predict_single_image(model, image_paths[0], device, args.img_size)
        print_result(result, args.show_all_probs)
        results = [result]
    else:
        logger.info("开始批量推理...")
        results = predict_batch(model, image_paths, device, args.img_size, args.batch_size)

        # 打印所有结果
        for result in results:
            print_result(result, args.show_all_probs)

        # 打印汇总
        print_summary(results)


if __name__ == '__main__':
    main()