import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import ElectraModel, ElectraTokenizer, ElectraConfig


def Pre_train_texts(df, model_path):
    # 预训练模型加载
    Pre_train_model_config = ElectraConfig.from_pretrained(model_path)  # 还可以增加配置参数，来覆盖修改默认config
    tokenizer = ElectraTokenizer.from_pretrained(model_path, do_lower_case=True)
    Pre_train_model = ElectraModel.from_pretrained(model_path, config=Pre_train_model_config)

    texts = df[['Name', 'MethodName']]  # 取data中指定两列的文本  ClassName
    embeddings = []  # 定义嵌入列表保存每条数据文本向量
    token_counts = []

    for row in range(len(texts)):   # len(texts)
        input_text = texts.iloc[row, :].astype(str).values.tolist()  # 转化为列表格式

        # 分词并编码文本
        text_dict = tokenizer.encode_plus(input_text[0], text_pair=input_text[1],
                                          add_special_tokens=True, return_attention_mask=True, return_tensors='pt')
        input_ids = text_dict['input_ids']  # token序列，词索引ID  shape = [1, Num_ids]
        token_type_ids = text_dict['token_type_ids']  # 两句文本之间的关系索引
        attention_mask_ids = text_dict['attention_mask']  # 注意力掩码

        token = input_ids.squeeze(0).numpy()
        token_counts.append(len(token))

        Pre_train_model.eval()
        device = 'cuda'  # 采用GPU运算
        tokens_tensor = input_ids.to(device)
        segments_tensors = token_type_ids.to(device)
        attention_mask_ids_tensors = attention_mask_ids.to(device)
        Pre_train_model.to(device)

        # Get the embeddings
        with torch.no_grad():
            encode_outputs = Pre_train_model(tokens_tensor, segments_tensors, attention_mask_ids_tensors)

        # 词向量shape = [batch, Num_token_id, Dim] 所选Electra-large模型Layers-24、N-max=1024、D-1024
        row_embeddings, _ = torch.max(encode_outputs[0], dim=1)  # 最大池化
        # row_embeddings_pooled = encode_outputs[1].squeeze(0)    # 全连接+池化+’tanh‘
        # row_embeddings_mean = torch.mean(encode_outputs[0], dim=1).squeeze(0)  # 平均池化
        row_embeddings_max = row_embeddings.squeeze(0)
        embeddings.append(row_embeddings_max)

    unique_elements = np.unique(token_counts)  # 获取数组中元素的类别数
    # 查看整个数据集的每条文本经过分词后切出token个数的分布情况
    plt.hist(token_counts, bins=len(unique_elements), alpha=0.5, color='b')
    plt.xlabel('Token IDs Count')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Token IDs Count')
    plt.show()
    return embeddings

