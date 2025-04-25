import torch
import pandas as pd
import numpy as np
import random

from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

class BERTFullFineTune():
    def __init__(self, path_to_data=None, Train_mode=True):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.df = pd.read_csv(path_to_data, names=['id', 'text', 'catagory'])
        self.Train_mode = Train_mode

    def f1_score_func(self, preds, labels):
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return f1_score(labels_flat, preds_flat, average='weighted')

    def accuracy_per_class(self, preds, labels):
        label_dit_inverse = {v: k for k, v in self.label_dict.items()}
        
        preds_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        
        for label in np.unique(labels_flat):
            y_pred = preds_flat[labels_flat==label]
            y_true = labels_flat[labels_flat==label]
        
            print(f'class: {label_dit_inverse}')
            print(f'accuracy: {len(y_pred[y_pred==label])}/{len(y_true)}\n')
        

    def train_test_prep(self):
        self.df.set_index('id', inplace=True)
        self.df = self.df[~self.df.catagory.str.contains('\\|')]
        possible_labels = self.df.catagory.unique()

        self.label_dict = {}
        for index, possible_label in enumerate(possible_labels):
            self.label_dict[possible_label] = index
        
        self.df['label'] = self.df.catagory.replace(self.label_dict)

        X_train, X_val, y_train, y_test = train_test_split(self.df.index.values,
                                                   self.df.label.values,
                                                   test_size = 0.10, 
                                                   random_state = 17,
                                                   stratify = self.df.label.values)

        self.df['data_type']=['not_set']*self.df.shape[0]
        self.df.loc[X_train,'data_type'] = 'train'
        self.df.loc[X_val,'data_type'] = 'val'

    def dataTokenize(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_caser=True)

        encoded_data_train = tokenizer.batch_encode_plus(
            self.df[self.df.data_type=='train'].text.values,
            add_special_tokens=True, 
            return_attention_mask=True,
            padding=True,         # Ensure long sequences get truncated
            max_length=256,          # Set a fixed maximum length
            return_tensors='pt'      # Convert output to PyTorch tensors
            )

        encoded_data_val = tokenizer.batch_encode_plus(
            self.df[self.df.data_type=='val'].text.values,
            add_special_tokens=True, 
            return_attention_mask=True,
            padding=True,            
            max_length=256,
            return_tensors='pt'
        )

        input_ids_train = encoded_data_train['input_ids']
        attention_masks_trian = encoded_data_train['attention_mask']
        labels_train = torch.tensor(self.df[self.df.data_type=='train'].label.values)

        input_ids_val = encoded_data_val['input_ids']
        attention_masks_val = encoded_data_val['attention_mask']
        labels_val = torch.tensor(self.df[self.df.data_type=='val'].label.values)


        self.dataset_train = TensorDataset(input_ids_train,
                           attention_masks_trian, labels_train)
        self.dataset_val = TensorDataset(input_ids_val,
                           attention_masks_val, labels_val)

    def evaluate(self, dataloader_val):

        self.model.eval()
        
        loss_val_total = 0
        predictions, true_vals = [], []
        
        for batch in dataloader_val:
            
            batch = tuple(b.to(self.device) for b in batch)
            
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'labels':         batch[2],
                    }

            with torch.no_grad():        
                outputs = self.model(**inputs)
                
            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)
        
        loss_val_avg = loss_val_total/len(dataloader_val) 
        
        predictions = np.concatenate(predictions, axis=0)
        true_vals = np.concatenate(true_vals, axis=0)
                
        return loss_val_avg, predictions, true_vals



    def modelTrain(self):
        seed_val = 17
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)
        batch_size = 32
        epochs=10


        self.dataloader_train =DataLoader(self.dataset_train,
                                        sampler=RandomSampler(self.dataset_train),
                                        batch_size=batch_size)

        self.dataloader_val =DataLoader(self.dataset_val,
                                    sampler=RandomSampler(self.dataset_val),
                                    batch_size=32)

        self.model = BertForSequenceClassification.from_pretrained(
                                                                'bert-base-uncased',
                                                                num_labels=len(self.label_dict),
                                                                output_attentions=False,
                                                                output_hidden_states=False
                                                                ).to(self.device)

        

        optimizer = AdamW(self.model.parameters(), lr=1e-6, eps=1e-8)
        schedular = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=10,
                                                    num_training_steps=len(self.dataloader_train)*epochs
                                                    )
       

        
        if self.Train_mode:
            for epoch in tqdm(range(1, epochs+1)):
        
                self.model.train()
                
                loss_trian_total = 0
                
                progress_bar = tqdm(self.dataloader_train, desc='Epoch {:1d}'.format(epoch),
                                leave=False,
                                disable=False)
                for batch in progress_bar:
                    self.model.zero_grad()
                    
                    batch = tuple(b.to(self.device) for b in batch)
                    
                    inputs = {
                        'input_ids'        :batch[0],
                        'attention_mask'   :batch[1],
                        'labels'           :batch[2]
                    }
                    
                    outputs = self.model(**inputs)
                    
                    loss = outputs[0]
                    loss_trian_total += loss.item()
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.)
                    
                    optimizer.step()
                    schedular.step()
                    
                    progress_bar.set_postfix({'training_los': '{:.3}'.format(loss.item()/len(batch))})
                    
                torch.save(self.model.state_dict(), f'Models/BERT_ft_epoch{epoch}.model')
                
                tqdm_train_avg = loss_trian_total/len(self.dataloader_train)
                
                val_loss, predictions, true_vals = self.evaluate(self.dataloader_val)
                val_f1 = self.f1_score_func(predictions, true_vals)
                tqdm.write(f'validation loss: {val_loss}')
                tqdm.write(f'F1 score (weighted): {val_f1}')

                

    def LoadTestModel(self):
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                      num_labels=len(self.label_dict),
                                                      output_attentions=False,
                                                      output_hidden_states=False).to(self.device)

        self.model.load_state_dict(torch.load('Models//BERT_ft_epoch10.model'))
        
        _, predictions, true_vals = self.evaluate(self.dataloader_val)
        self.accuracy_per_class(predictions, true_vals)



if __name__ == "__main__":
    
    import os

    print(os.listdir("Models"))  # Lists all files in the directory
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)

    dataPath = f'{current_dir}/data/smile-annotations-final.csv'

    bfft = BERTFullFineTune(dataPath)

    bfft.Train_mode = True
    print("\n\n Model sample created! \n\n")


    bfft.train_test_prep()
    print("\n\n Test/Train separated! \n\n")

    bfft.dataTokenize()
    print("\n\n Data Tokenized! \n\n")


    bfft.modelTrain()
    print("\n\n Model Trained! \n\n")
    
    bfft.LoadTestModel()
    print("\n\n Model Evaluated! \n\n")








