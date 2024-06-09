# DeepCSS: Severity Classification for Code Smell Based on Deep Learning
Code smell severity refers to the different impact extent of smelly instances on a specific project when more than one kind of code smell exists. The severity classification helps developers better understand code smell and prioritize multiple refactoring operations, thus improving the efficiency of software maintenance. <be>

However, existing works on the severity classification of code smell suffer from insufficient quantitative evaluation and low accuracy. To this end, this paper proposes a novel approach called DeepCSS to classify code smell severity based on deep learning.<be>

To support the classification, we present a quantitative evaluation framework for code smell severity by assessing the importance of each metric. To support the training of the deep learning model, we construct datasets for four types of code smell (including data class, god class, long method, and feature envy) from 100 real-world projects. With this evaluation framework, severity labels of these samples are built.<be>

DeepCSS is evaluated on four types of code smell severity. The experimental results show that DeepCSS can achieve an accuracy ranging from 95.95\% to 99.39\%. Furthermore, it is compared against two existing works and obtains an improvement in F1-score of 6.38\% and 0.84\% on average, demonstrating the effectiveness of our approach.
## DeepCSS Model
The main code for DeepCSS is available at the following URL [DeepCSS_model.py](https://github.com/AAAAaBae/DeepCSS/blob/main/DeepCSS_model.py) <br>
The Pre_train code is available at the following URL [Pre_train.py](https://github.com/AAAAaBae/DeepCSS/blob/main/Pre_train.py) <br>
## Dataset
* Identifiers such as ProjectName, PackageName, and ClassName MethodName are filtered by symbols and stop words.
* Features such as is_constructor, is_abstract, is_inner, and is_static are of bool type and are converted into binary values of 0 or 1.
* Missing and outlier values are deleted. <br>

The datasets are preprocessed after the above steps, which the pre-trained model can directly train to convert the text into word-vector. The processed datasets are available at the following URL [Severity_DataSets.zip](https://github.com/AAAAaBae/DeepCSS/blob/main/Severity_DataSets.zip)
