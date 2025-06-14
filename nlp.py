import json
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from tqdm import tqdm


def load_model_and_tokenizer(base_model_path, lora_model_path):
    """
    加载基础模型和LoRA适配器

    参数:
    base_model_path -- 基础模型路径
    lora_model_path -- LoRA适配器路径

    返回:
    model -- 加载LoRA后的模型
    tokenizer -- 对应的分词器
    """
    # 加载基础模型的分词器
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用半精度减少内存占用
        device_map="auto"  # 自动分配设备(GPU/CPU)
    )

    # 通过Peft加载LoRA适配器
    model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map="auto"
    )

    # 合并LoRA权重到基础模型(优化推理速度)
    model = model.merge_and_unload()

    # 设置为评估模式
    model.eval()

    return model, tokenizer


def generate_predictions(model, tokenizer, input_data):
    """
    生成模型预测结果

    参数:
    model -- 加载好的模型
    tokenizer -- 分词器
    input_data -- 输入数据列表

    返回:
    outputs -- 包含所有预测结果的列表
    """
    outputs = []

    # 使用无梯度计算以节省内存
    with torch.no_grad():
        # 遍历每个输入样本
        for item in tqdm(input_data, desc="处理中"):
            # 获取输入文本(假设每个字典有两个元素，第二个是文本)
            input_text = list(item.values())[1]

            # 使用分词器准备模型输入
            inputs = tokenizer(
                input_text,
                return_tensors="pt",  # 返回PyTorch张量
                padding=True,  # 自动填充
                truncation=True,  # 自动截断
                max_length=512  # 最大长度限制
            ).to(model.device)  # 移动到模型所在设备

            # 生成文本(调整参数可优化性能)
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=20,
                # 替换为Qwen兼容的参数
                top_k=40,  # 替代temperature的采样控制
                top_p=0.9,  # 替代temperature的采样控制
                do_sample=True,  # 需要启用采样
                repetition_penalty=1.1
            )

            # 解码生成的token，跳过特殊token
            prediction = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )

            # 仅保留新生成的文本(移除输入部分)
            # 注意: 根据您的任务可能需要调整此逻辑
            generated_text = prediction[len(input_text):].strip()

            outputs.append(generated_text)

    return outputs


def main():
    # 配置路径(请根据实际情况修改)
    BASE_MODEL_PATH = "qwen3"  # 基础模型文件夹路径
    LORA_MODEL_PATH = "qwen-2-l"  # LoRA模型文件夹路径
    TEST_FILE_PATH = "test1-1.json"  # 测试文件路径
    OUTPUT_FILE_PATH = "output.json"  # 输出文件路径

    # 步骤1: 加载模型和分词器
    print("正在加载模型...")
    model, tokenizer = load_model_and_tokenizer(BASE_MODEL_PATH, LORA_MODEL_PATH)
    print(f"模型已加载到设备: {model.device}")

    # 步骤2: 读取测试数据
    print("正在读取测试数据...")
    with open(TEST_FILE_PATH, 'r', encoding='utf-8') as f:
        test_data = json.load(f)  # 加载JSON数据

    # 验证数据格式
    if not isinstance(test_data, list) or len(test_data) == 0:
        raise ValueError("测试数据格式错误: 应为非空列表")

    print(f"成功加载 {len(test_data)} 条测试数据")

    # 步骤3: 生成预测结果
    print("开始生成预测...")
    predictions = generate_predictions(model, tokenizer, test_data)

    # 步骤4: 保存结果到JSON文件
    with open(OUTPUT_FILE_PATH, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"预测结果已保存至: {OUTPUT_FILE_PATH}")


if __name__ == "__main__":
    main()