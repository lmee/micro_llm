# micro_llm, 一个极简的LLM项目,用来学习和理解LLM的原理和实现

## 简介
项目结构

- `mini_llm/`  
  - `data/`                  # 数据存储目录  
  - `models/`                # 模型定义  
    - `__init__.py`  
    - `mini_transformer.py`  # 我们的极小Transformer模型  
  - `training/`              # 训练相关代码  
    - `__init__.py`  
    - `pretrain.py`          # 预训练代码  
    - `sft.py`               # 监督微调代码  
    - `lora.py`              # LoRA实现  
    - `dpo.py`               # DPO实现  
    - `distill.py`           # 蒸馏实现  
  - `utils/`                 # 工具函数  
    - `__init__.py`  
    - `data_utils.py`        # 数据处理工具  
    - `train_utils.py`       # 训练工具  
  - `main.py`                # 主入口文件