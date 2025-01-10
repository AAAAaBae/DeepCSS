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


def DataClass(data):
    if (data['NOAM'] >= 3) and (data['WOC'] <= 0.33):
        return 1
    elif (data['NOPA'] >= 2) and (data['WOC'] <= 0.33):
        return 1
    elif (data['WOC'] <= 0.33) and (data['NOAM'] + data['NOPA'] > 4) and (data['WMC_y'] < 47):
        return 1
    elif (data['WOC'] <= 0.33) and (data['NOAM'] + data['NOPA'] > 2) and (data['WMC_y'] < 31):
        return 1
    elif (data['NOAM'] >= 10) and (data['lcom*'] > 0.66):
        return 1
    elif (data['NOAM'] >= 3) and (data['WOC'] <= 0.33) and (data['rfc'] <= 43):
        return 1
    else:
        return 0


def LargeClass(data):
    if (data['totalFieldsQty'] > 9) or (data['NOM'] > 15) or (data['loc'] > 150):
        return 1
    elif (data['WMC_y'] >= 47) and (data['ATFD'] >= 4) and (data['TCC'] < 0.33):
        return 1
    elif data['NOM'] + data['totalFieldsQty'] > 20:
        return 1
    elif (data['lcom*'] > 0.66) and (data['WMC_y'] >= 47) and (data['NOM'] > 8) and (data['totalFieldsQty'] > 14):
        return 1
    else:
        return 0


def GodClass(data):
    if (data['ATFD'] >= 4) and (data['WMC_y'] > 20) and (data['TCC'] <= 0.33):
        return 1
    elif (data['ATFD'] > 2) and (data['WMC_y'] > 47) and (data['TCC'] <= 0.33):
        return 1
    elif (data['loc'] >= 176) and (data['WMC_y'] >= 22) and (data['NOM'] >= 18) and (data['ATFD'] >= 6) and (data['TCC'] <= 0.33):
        return 1
    else:
        return 0


def ShotgunSurgery(data):
    if (data['CC'] >= 5) and (data['CM'] >= 6) and (data['fanout'] >= 3):
        return 1
    elif (data['CM'] >= 10) and (data['CC'] >= 5):
        return 1
    else:
        return 0


root_directory = "E:/Result_class_data"
f_path = Get_files_path(root_directory)

count_all = []
count_DC = []
count_LC = []
count_GC = []
count_SS = []

for path in f_path:
    c_data = pd.read_csv(path, encoding='UTF-8')
    count_all.append(len(c_data))

    c_data.loc[:, 'is_DC_D'] = c_data.apply(DataClass, axis=1)
    c_data.loc[:, 'is_LC_D'] = c_data.apply(LargeClass, axis=1)
    c_data.loc[:, 'is_GC_D'] = c_data.apply(GodClass, axis=1)
    c_data.loc[:, 'is_SS_D'] = c_data.apply(ShotgunSurgery, axis=1)

    print(c_data['is_DC_D'].value_counts())
    print(c_data['is_LC_D'].value_counts())
    print(c_data['is_GC_D'].value_counts())
    print(c_data['is_SS_D'].value_counts())

    c_data.to_csv(path, index=False)

    count_DC.append((c_data['is_DC_D'] == 1).sum())
    count_LC.append((c_data['is_LC_D'] == 1).sum())
    count_GC.append((c_data['is_GC_D'] == 1).sum())
    count_SS.append((c_data['is_SS_D'] == 1).sum())

print(sum(count_all))
print(sum(count_DC))
print(sum(count_LC))
print(sum(count_GC))
print(sum(count_SS))

