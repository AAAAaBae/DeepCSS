import pandas as pd
import os

def Get_files_path(folder):
    files_path = []
    keyword = 'result'
    for current_dir, dirs, files in os.walk(folder):
        for file in files:
            if keyword not in file:
                continue
            file_path = os.path.join(current_dir, file)
            files_path.append(file_path)
    return files_path


def LongMethod(data):
    if (data['CYCLO'] >= 3) and (data['LOC_x'] >= 20) and (data['NOP'] >= 1) and (data['MAXNESTING_x'] >= 2):
        return 1
    elif (data['LOC_x'] >= 30) and (data['NOP'] >= 4):
        return 1
    elif (data['LOC_x'] >= 30) and (data['NOLV'] >= 4):
        return 1
    elif data['LOC_x'] > 50:
        return 1
    else:
        return 0


def FeatureEnvy(data):
    if (data['ATFD'] > 2) and (data['FDP'] <= 2) and (data['LAA'] < 0.33):
        return 1
    elif data['cbo'] > 5:
        return 1
    elif (data['ATFD'] >= 5) and (data['LAA'] < 0.33):
        return 1
    else:
        return 0


def BrainMethod(data):
    if (data['CYCLO'] >= 4) and (data['LOC_x'] >= 33) and (data['MAXNESTING_x'] >= 4) and (data['NOLV'] >= 5):
        return 1
    elif (data['NOLV'] >= 5) and (data['ALD'] >= 4):
        return 1
    else:
        return 0


def LongParameterList(data):
    if data['NOP'] >= 5:
        return 1
    else:
        return 0


root_dir = "E:/Result_method_data"
f_path = Get_files_path(root_dir)
print(len(f_path))

count_all = []
count_LM = []
count_FE = []
count_BM = []
count_LPL = []

for path in f_path:
    # c_data = pd.read_excel(path)
    c_data = pd.read_csv(path, encoding='UTF-8')
    count_all.append(len(c_data))

    c_data.loc[:, 'is_LongMethod'] = c_data.apply(LongMethod, axis=1)
    c_data.loc[:, 'is_FeatureEnvy'] = c_data.apply(FeatureEnvy, axis=1)
    c_data.loc[:, 'is_BrainMethod'] = c_data.apply(BrainMethod, axis=1)
    c_data.loc[:, 'is_LongParameterList'] = c_data.apply(LongParameterList, axis=1)

    print(c_data['is_LongMethod'].value_counts())
    print(c_data['is_FeatureEnvy'].value_counts())
    print(c_data['iis_BrainMethod'].value_counts())
    print(c_data['is_LongParameterList'].value_counts())

    c_data.to_csv(path, index=False)

    count_LM.append((c_data['is_LongMethod'] == 1).sum())
    count_FE.append((c_data['is_FeatureEnvy'] == 1).sum())
    count_BM.append((c_data['is_BrainMethod'] == 1).sum())
    count_LPL.append((c_data['is_LongParameterList'] == 1).sum())

print(sum(count_all))
print(sum(count_LM))
print(sum(count_FE))
print(sum(count_BM))
print(sum(count_LPL))
