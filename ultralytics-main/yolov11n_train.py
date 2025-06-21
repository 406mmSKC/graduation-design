import warnings
import traceback
from ultralytics import YOLO

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    try:
        model = YOLO('ultralytics/cfg/models/11/yolo11n.yaml')
        model.load('yolo11n.pt')

        model.train(
            data='data.yaml',
            imgsz=640,
            epochs=300,
            batch=200,
            workers=45,
            device=[0, 1, 2,3],
            optimizer='SGD',
            amp=True,
            cache=True
        )

    except Exception as e:
        error_msg = f"""
        [训练崩溃] 异常类型: {type(e).__name__}
        ------------------------------
        错误详情: {str(e)}
        ------------------------------
        堆栈追踪:
        {traceback.format_exc()}
        """
        print("\033[91m" + error_msg + "\033[0m")  # 红色高亮输出

        # 智能诊断建议（综合网页1、3、5）
        if "CUDA" in str(e):
            print("\n\033[93m建议排查:\033[0m")
            print("1. 显存不足 → 降低batch_size或启用梯度累积(accumulate=2)")
            print("2. 环境冲突 → 检查CUDA版本与PyTorch的兼容性")
            print("3. 碎片优化 → 设置环境变量: `export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128`")
        elif "DataLoader" in str(e):
            print("\n\033[93m建议排查:\033[0m")
            print("1. workers过高 → 降低至CPU核心数的75%（如40→30）")
            print("2. 数据集路径错误 → 检查data.yaml中的路径是否存在空格或中文")
        elif "AMP" in str(e):
            print("\n\033[93m建议排查:\033[0m")
            print("1. 关闭混合精度 → 设置amp=False")
            print("2. 检查输入数据范围 → 确保归一化到[0,1]或[-1,1]")

        exit(-1)