import pandas as pd
import numpy as np


def entropy_calc(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


def InfoGain(data, split_attribute_name, target_name):
    total_entropy = entropy_calc(data[target_name])
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy_calc(
            data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


def ID3(data, originaldata, features, target_attribute_name, parent_node_class=None):
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]
    elif len(features) == 0:
        return parent_node_class
    else:
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in
                       features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature: {}}
        features = [i for i in features if i != best_feature]
        for value in np.unique(data[best_feature]):
            value = value
            sub_data = data.where(data[best_feature] == value).dropna()
            subtree = ID3(sub_data, dataset, features, target_attribute_name, parent_node_class)
            tree[best_feature][value] = subtree
        return (tree)


def predict(query, tree_elm, default=1):
    for key in list(query.keys()):
        if key in list(tree_elm.keys()):
            try:
                result = tree_elm[key][query[key]]
            except:
                return default
            result = tree_elm[key][query[key]]
            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def train_test_split(dataset_tmp):
    count_row = dataset_tmp.shape[0]
    training_cnt = int(count_row * 80 / 100)
    training_dataset = dataset_tmp.iloc[:training_cnt].reset_index(drop=True)
    testing_dataset = dataset_tmp.iloc[training_cnt:].reset_index(drop=True)
    return training_dataset, testing_dataset


def test(data, tree_elm, target_attr):
    print("test")
    queries = data.iloc[:, :-1].to_dict(orient="records")
    data['predict'] = pd.DataFrame
    for i in range(len(data)):
        data.loc[i, "predicted"] = predict(queries[i], tree_elm, 1.0)
    print('The prediction accuracy is: ', (np.sum(data["predicted"] == data[target_attr]) / len(data)) * 100, '%')


def test_2(data, tree, target_attr):
    pass


dataset = pd.read_csv('playTennis.data', sep=',', header = None)
df = pd.DataFrame(dataset)
## For Header ##
df_col = []
for e in range(len(df.columns)):
    df_col.append('A' + str(e))
df.columns = df_col
################
df_train = df.sample(frac=.8, random_state=10).reset_index(drop=True)
df_test = df.drop(df_train.index).reset_index(drop=True)
print("ID3 starts")
f = df_train.columns[:-1]
t_name = df_train.columns[-1]
print(f)
print(t_name)
tree = ID3(df_train, df_train, f, t_name)
print(tree)
test(df_test, tree, t_name)
