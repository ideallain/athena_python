import json

import torch.cuda
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_prepare import samples



class TrainModel:
    model_name = "../model/deepseekr1-1.5b"
    dataset_file = "../dataset/train_model_datasets.json"

    @classmethod
    def prepare_dataset(self):
        with open(self.dataset_file, "w", encoding="utf-8") as f:
            for line in samples:
                json_line = json.dumps(line, ensure_ascii=False)
                f.write(json_line + "\n")

            print("this json has write into json file")

        # 第三步 准备训练数据和测试数据
        from datasets import load_dataset
        dataset = load_dataset(path="json", data_files={"train": self.dataset_file}, split="train")
        print(f"the number of data {len(dataset)}")
        train_test_split = dataset.train_test_split(test_size=0.1)
        train_dataset = train_test_split["train"]
        test_dataset = train_test_split["test"]
        print(f"the line of train dataset {len(train_dataset)} ,the lint of test_dataset {len(test_dataset)}")
        return train_dataset, test_dataset


    # 第四步， 编写tokenizer处理工具
    @classmethod
    def prepare_tokenizer(self, many_samples):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        texts = [f"{prompt}\n{completion}" for prompt, completion in
                 zip(many_samples["prompt"], many_samples["completion"])]
        tokens = tokenizer(texts, truncation=True, max_length=512, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    @classmethod
    def train_model(self):
        train_dataset,test_dataset = self.prepare_dataset()
        tokenized_train_dataset = train_dataset.map(self.prepare_tokenizer, batched=True)
        tokenized_eval_dataset = test_dataset.map(self.prepare_tokenizer, batched=True)
        print(f"the tokenized_train_dataset {tokenized_train_dataset[0]} , \nthe tokenized_eval_dataset {tokenized_eval_dataset[0]}")

        from transformers import BitsAndBytesConfig
        # 使用CPU时不要设置这个参数
        quantization_config = BitsAndBytesConfig(load_in_8bit = True)
        model_kwargs = {
            # "trust_remote_code": model_params.trust_remote_code,
            "device_map": "auto",
            "quantization_config": quantization_config,
            "trust_remote_code": True
        }
        if torch.cuda.is_available():
            print(f"the cuda is available")
        else:
            print(f"will use cpu")
            model_kwargs["torch_dtype"] = torch.float32
            model_kwargs["quantization_config"] = ""
        print(f"the transformer version is {transformers.__version__}")
        # 到这里运行失败，因为模型一直加载失败，回来再改吧
        # 必须还使用CUDA版本才可以运行这个模型
        model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        print(f"load model success,model will use {next(model.parameters()).device.type}")

        from peft import get_peft_model,LoraConfig,TaskType
        lora_config = LoraConfig(r = 8, lora_alpha = 16, lora_dropout = 0.05, task_type = TaskType.CAUSAL_LM)
        model = get_peft_model(model,lora_config)
        model.print_trainable_parameters()

        #设置训练参数,并且训练
        from transformers import TrainingArguments, Trainer
        training_args = TrainingArguments(
            output_dir = "./finetuned_models",
            num_train_epochs = 10,
            per_device_train_batch_size = 4,
            gradient_accumulation_steps = 8,
            fp16 = True,
            logging_steps = 16,
            save_steps = 100,
            eval_strategy = "steps",
            eval_steps = 10,
            learning_rate = 3e-5,
            logging_dir = "./logs",
            run_name = "deepseek_r1-distill-finetune"
        )

        print(f"train parameter config finished")
        trainer = Trainer(model = model,args = training_args, train_dataset = tokenized_train_dataset, eval_dataset = tokenized_eval_dataset)
        print("begin train")
        trainer.train()
        print("train finished")

# trainModel = TrainModel()
# TrainModel.train_model()
TrainModel.train_model()


import torch
import sys
print("=== PyTorch CUDA 诊断信息 ===")
print(f"Python 版本: {sys.version}")
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
print(f"可用 GPU 数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
