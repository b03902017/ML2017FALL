import argparse
import csv
import numpy as np
import xgboost as xgb

# Global param setting
normalization = True
valid_num = 0

def extract_features(row):
    row = [float(x) for x in row]
    features = [i for i in row]
    # features.extend([row[i] for i in [0,1,2,3,5]])
    # 加入['age','fnlwgt','capital_gain','hours_per_week']等連續features
    features.extend(row[i]**2 for i in [0,1,3,5])
    features.extend([row[i]**3 for i in [0,1,3,5]])
    return features

def normalize(train_x, test_x):
    x = np.concatenate((train_x, test_x))
    mean = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    np.place(sigma, sigma==0, 1)

    mean = np.tile(mean, (x.shape[0], 1))
    sigma = np.tile(sigma, (x.shape[0], 1))
    x = (x - mean) / sigma
    return x[:train_x.shape[0]], x[train_x.shape[0]:]

def load_data(train_x_path, train_y_path, test_x_path,):
    train_x = []
    with open(train_x_path) as f_train:
        for row_n, row in enumerate(csv.reader(f_train)):
            if row_n != 0: # row0 contains no data
                features = extract_features(row)
                train_x.append(features)
    train_x = np.array(train_x)

    test_x = []
    with open(test_x_path) as f_test:
        for row_n, row in enumerate(csv.reader(f_test)):
            if row_n != 0: # row0 contains no data
                features = extract_features(row)
                test_x.append(features)
    test_x = np.array(test_x)

    if normalization:
        train_x, test_x = normalize(train_x, test_x)
    valid_x = train_x[:valid_num]
    train_x = train_x[valid_num:]

    train_y = []
    with open(train_y_path) as f_train:
        for row_n, row in enumerate(csv.reader(f_train)):
            if row_n != 0: # row0 contains no data
                row = int(row[0])
                train_y.append(row)
    valid_y = np.array(train_y[:valid_num])
    train_y = np.array(train_y[valid_num:])
    return train_x, train_y, valid_x, valid_y, test_x

def evaluate(model, train_x, train_y, valid_x, valid_y):
    pred_y = model.predict(train_x)
    print(f'train accu: {((train_y == pred_y).sum()) / train_y.shape[0] : .4f}')
    if valid_num > 0:
        pred_y = model.predict(valid_x)
        print(f'valid accu: {((valid_y == pred_y).sum()) / valid_num : .4f}')

def save_prediction(pred_y, output_path):
    with open(output_path, 'w') as f_out:
        print('id,label', file=f_out)
        for i in range(len(pred_y)):
            print(f'{i+1},{pred_y[i]}', file=f_out)

class gradient_boosting:
    def __init__(self):
        self._params = {
            'max_depth': 10,
            'eta': 0.01,
            'min_child_weight' : 1,
            'max_features': 17,
            'min_samples_split': 300,
            'gamma': 0.5,
            'subsample': 0.75,
            'colsample_bytree': 0.8,
            'objective': 'binary:logistic',
            'nthread': 4,
            'silent': 1,
            'verbose_eval': False,
            'warm_start': False,
            'eval_metric': 'auc' }
        self._num_round = 1100
        self._bst = None

    def predict(self, X):
        self._bst = xgb.Booster(model_file='models/best.model')
        pred_prob = self._bst.predict(xgb.DMatrix(X))
        return [0 if i <0.5 else 1 for i in pred_prob]

    def fit(self, train_x, train_y):
        dtrain = xgb.DMatrix(train_x, label=train_y)
        self._bst = xgb.train(self._params, dtrain, self._num_round)
        self._bst.save_model('models/best.model')

def main(args):
    global valid_num
    valid_num = args.valid_num
    train_x, train_y, valid_x, valid_y, test_x = \
    load_data(args.train_x_path, args.train_y_path, args.test_x_path)

    if args.output_path is None:
        print('Error: Argument --output_path for the path of prediction result')
        return

    model = gradient_boosting()
    if args.train:
        model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    save_prediction(pred_y, args.output_path)

    if args.evaluate:
        evaluate(model, train_x, train_y, valid_x, valid_y)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', action='store_true', default=False,
                        dest='train', help='Input --train to train xgboost')
    parser.add_argument('-x', '--train_data_path', type=str,
                        default='features/X_train', dest='train_x_path',
                        help='Path to training data')
    parser.add_argument('-y', '--train_label_path', type=str,
                        default='features/Y_train', dest='train_y_path',
                        help='Path to training data\'s label')
    parser.add_argument('-i', '--test_data_path', type=str,
                        default='features/X_test', dest='test_x_path',
                        help='Path to testing data')
    parser.add_argument('-o', '--output_path', type=str, dest='output_path',
                        help='Path to save the prediction result')
    parser.add_argument('-e', '--evaluate', action='store_true', default=False,
                        dest='evaluate', help='Input --evaluate to evaluate results')
    parser.add_argument('-v', '--valid_num', type=int, default=0,
                         dest='valid_num', help='Size of validation data')
    main(parser.parse_args())
