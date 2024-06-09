import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import ElectraModel, ElectraTokenizer, ElectraConfig


def Pre_train_texts(df, model_path):
    Pre_train_model_config = ElectraConfig.from_pretrained(model_path)  
    tokenizer = ElectraTokenizer.from_pretrained(model_path, do_lower_case=True)
    Pre_train_model = ElectraModel.from_pretrained(model_path, config=Pre_train_model_config)

    texts = df[['Name', 'MethodName']]  # ClassName or MethodName
    embeddings = []  
    token_counts = []

    for row in range(len(texts)):   # len(texts)
        input_text = texts.iloc[row, :].astype(str).values.tolist()  

        text_dict = tokenizer.encode_plus(input_text[0], text_pair=input_text[1],
                                          add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
        input_ids = text_dict['input_ids']  
        token_type_ids = text_dict['token_type_ids']  
        attention_mask_ids = text_dict['attention_mask']  

        token = input_ids.squeeze(0).numpy()
        token_counts.append(len(token))

        Pre_train_model.eval()
        device = 'cuda'  
        tokens_tensor = input_ids.to(device)
        segments_tensors = token_type_ids.to(device)
        attention_mask_ids_tensors = attention_mask_ids.to(device)
        Pre_train_model.to(device)

        # Get the embeddings
        with torch.no_grad():
            encode_outputs = Pre_train_model(tokens_tensor, segments_tensors, attention_mask_ids_tensors)

        # shape = [batch, Num_token_id, Dim], Electra-large: Layers-24、N-max=1024、D-1024
        row_embeddings, _ = torch.max(encode_outputs[0], dim=1) 
        # row_embeddings_pooled = encode_outputs[1].squeeze(0)   
        # row_embeddings_mean = torch.mean(encode_outputs[0], dim=1).squeeze(0)  
        row_embeddings_max = row_embeddings.squeeze(0)
        embeddings.append(row_embeddings_max)

    unique_elements = np.unique(token_counts) 
    return embeddings

