# DeepCSS: Severity Classification for Code Smell Based on Deep Learning
Code smell severity refers to the different levels of impact extent that smelly instances may have upon a specific project when more than one kind of code smell exists. Severity classification helps developers better understand a code smell and prioritize multiple refactoring operations more efficiently, thus improving the efficiency of software maintenance. However, existing studies on code smell severity assessment and classification suffer from insufficient quantitative evaluation and low accuracy. 

To this end, this paper proposes DeepCSS, a novel approach to classify code smell severity based on deep learning. To evaluate the severity of code smells reasonably and accurately, a quantitative evaluation framework is proposed to evaluate the importance of assessing each related metric. With this evaluation framework, datasets are constructed for four types of code smell (including data class, god class, long method, and feature envy) extracted from 100 real-world projects. 

DeepCSS acquires structural and semantic information from which features are extracted by leveraging BiLSTM-Attention and the improved CNN model. Then the final classification is done by a fully connected network containing the Attention mechanism and \textit{softmax} functions. The experimental results show that DeepCSS can achieve an accuracy ranging from 95.11\% to 98.97\%. Compared to other studies, \textit{DeepCSS} obtained an average improvement of 6.97\% and 1.39\% in MCC, demonstrating its effectiveness. <be>

## DeepCSS Model
The main code for DeepCSS is available at the following URL [DeepCSS_model.py](https://github.com/AAAAaBae/DeepCSS/blob/main/DeepCSS_model.py) <br>
The Pre_train code is available at the following URL [Pre_train.py](https://github.com/AAAAaBae/DeepCSS/blob/main/Pre_train.py) <br>
The implementation code of the Baseline Method is available at the following URL [ML_Model.py](https://github.com/AAAAaBae/DeepCSS/blob/main/ML_Model.py) <be>
## Dataset
* Identifiers text such as ProjectName, PackageName, and ClassName MethodName are filtered by symbols and stop words.
* Features such as is_constructor, is_abstract, is_inner, and is_static are of bool type and are converted into binary values of 0 or 1.
* Missing and outlier values are deleted. <br>

The datasets are preprocessed after the above steps, which the pre-trained model can directly train to convert the text into a word vector. The processed datasets are available at the following URL [Datasets.zip](https://github.com/AAAAaBae/DeepCSS/blob/main/Datasets.zip)

## Pipeline
The Python scripting tool for screening out the smelly instances is available at the following URL [Class.py](https://github.com/AAAAaBae/DeepCSS/blob/main/Class.py) and [Method.py](https://github.com/AAAAaBae/DeepCSS/blob/main/Method.py)<br>
