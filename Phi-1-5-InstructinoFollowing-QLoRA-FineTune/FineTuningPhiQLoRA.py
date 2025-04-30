import os
import torch

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    pipeline,
    logging,
    BitsAndBytesConfig)

from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class DataPrep():
    def __init__(self,pr=AttrDict()):  
        super(DataPrep, self).__init__()    

    def load_dataset(self):
        self.dataset = load_dataset('tatsu-lab/alpaca')
    
    def train_test_separation(self):
        full_dataset = self.dataset['train'].train_test_split(test_size=0.05, shuffle=True)
        self.dataset_train = full_dataset['train']
        self.dataset_valid = full_dataset['test']

    def preprocess_function(self, example):
        """
        Formatting function returning a list of samples (kind of necessary for SFT API).
        """
        text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        return text


class Mdl():
    def __init__(self,Pr=AttrDict()):        
        super(Mdl, self).__init__()  
        # Quantization configuration.
        if Pr.bf16:
            self.compute_dtype = getattr(torch, 'bfloat16')
        else: # FP16
            self.compute_dtype = getattr(torch, 'float16')

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=True
            )   
        self.model = AutoModelForCausalLM.from_pretrained(Pr.model_name,
                                                    quantization_config=self.quant_config)

        self.tokenizer = AutoTokenizer.from_pretrained(Pr.model_name, 
                                                    trust_remote_code=True,
                                                    use_fast=False)
        
        self.tokenizer.pad_token = self.tokenizer.eos_token


    def model_info(self)                                                    :
        print(model)
        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")



if __name__ == "__main__":

    Pr = AttrDict()

    # Configuration Training 
    Pr.batch_size = 1
    Pr.num_workers = os.cpu_count()
    Pr.epochs = 1
    Pr.bf16 = True
    Pr.fp16 = False
    Pr.gradient_accumulation_steps = 16
    Pr.context_length = 1024
    Pr.learning_rate = 0.0002
    Pr.model_name = 'microsoft/phi-1_5'
    Pr.out_dir = 'outputs/phi_1_5_alpaca_qlora'
    

    dt = DataPrep(Pr)
    md = Mdl(Pr)    
    
    

    dt = DataPrep()
    dt.load_dataset()
    dt.train_test_separation()
        
    



    peft_params = LoraConfig(
                            lora_alpha=16,
                            lora_dropout=0.1,
                            r=16,
                            bias='none',
                            task_type='CAUSAL_LM',)

    training_args = TrainingArguments(
                                    output_dir=f"{Pr.out_dir}/logs",
                                    eval_strategy='epoch',
                                    weight_decay=0.01,
                                    load_best_model_at_end=True,
                                    per_device_train_batch_size=Pr.batch_size,
                                    per_device_eval_batch_size=Pr.batch_size,
                                    logging_strategy='epoch',
                                    save_strategy='epoch',
                                    num_train_epochs=Pr.epochs,
                                    save_total_limit=2,
                                    bf16=Pr.bf16,
                                    fp16=Pr.fp16,
                                    report_to='tensorboard',
                                    dataloader_num_workers=Pr.num_workers,
                                    gradient_accumulation_steps=Pr.gradient_accumulation_steps,
                                    learning_rate=Pr.learning_rate,
                                    lr_scheduler_type='constant')
    sft_config = SFTConfig(
                        max_seq_length=Pr.context_length,  # Move it here
                        per_device_train_batch_size=Pr.batch_size,
                        gradient_accumulation_steps=Pr.gradient_accumulation_steps,
                        learning_rate=Pr.learning_rate,
                        num_train_epochs=Pr.epochs,
                        packing=True)

    trainer = SFTTrainer(
                        model=md.model,
                        train_dataset=dt.dataset_train,
                        eval_dataset=dt.dataset_valid,    
                        args=sft_config,  # Use SFTConfig instead of TrainingArguments    
                        peft_config=peft_params,
                        formatting_func=dt.preprocess_function)

    dataloader = trainer.get_train_dataloader()



    history = trainer.train()
    trainer.model.save_pretrained(f"{Pr.out_dir}/best_model")
    trainer.tokenizer.save_pretrained(f"{Pr.out_dir}/best_model")

    

    


    
    
