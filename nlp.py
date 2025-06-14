import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm  # 用于显示进度条

def generate_predictions(model, tokenizer, input_data):
    outputs = []
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
                top_k=35,  # 替代temperature的采样控制
                top_p=0.9,  # 替代temperature的采样控制
                do_sample=True,  # 需要启用采样
                repetition_penalty=1.1
            )

            # 解码生成的token，跳过特殊token
            prediction = tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True
            )

            generated_text = prediction[len(input_text):].strip()

            outputs.append(generated_text)

    return outputs
def load_model_and_tokenizer(base_model_path, lora_model_path):
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


def main():
    basemodel = "qwen3"  # 基础模型文件夹路径
    lora = "qwen-1-h"  # LoRA模型文件夹路径
    test = "test.json"  # 测试文件路径
    output = "output1.json"  # 输出文件路径

    # 步骤1: 加载模型和分词器
    print("正在加载模型...")
    model, tokenizer = load_model_and_tokenizer(basemodel, lora)
    print(f"模型已加载到设备: {model.device}")

    # 步骤2: 读取测试数据
    print("正在读取测试数据...")
    with open(test, 'r', encoding='utf-8') as f:
        test_data = json.load(f)  # 加载JSON数据

    # 验证数据格式
    if not isinstance(test_data, list) or len(test_data) == 0:
        raise ValueError("测试数据格式错误: 应为非空列表")

    print(f"成功加载 {len(test_data)} 条测试数据")

    # 步骤3: 生成预测结果
    print("开始生成预测...")
    predictions = generate_predictions(model, tokenizer, test_data)

    # 步骤4: 保存结果到JSON文件
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"预测结果已保存至: {output}")


if __name__ == "__main__":
    main()