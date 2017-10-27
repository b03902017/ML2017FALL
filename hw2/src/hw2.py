import argparse
import csv
import numpy as np
from numpy.linalg import pinv
from math import floor

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

def sigmoid(x):
    x = np.clip(x, -30, 30)
    return (1 / (1 + np.exp(-x)))

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

class generative_model:
    def __init__(self):
        self._N_1, self._N_2 = None, None
        self._mean_1, self._mean_2 = None, None
        self._icov = None
        pass

    def predict(self, X):
        N_1, N_2 = self._N_1, self._N_2
        mean_1, mean_2 = self._mean_1, self._mean_2
        icov = self._icov
        z = np.dot(mean_1-mean_2, np.dot(icov, X.T))\
            - 0.5*np.dot(mean_1, np.dot(icov, mean_1.T))\
            + 0.5*np.dot(mean_2, np.dot(icov, mean_2.T)) + np.log(N_1/N_2)
        return [0 if i >0.5 else 1 for i in sigmoid(z)]

    def fit(self, train_x, train_y):
        c1_train_x = train_x[train_y == 0, :] # N * dim
        c2_train_x = train_x[train_y == 1, :]
        self._N_1 = c1_train_x.shape[0]
        self._N_2 = c2_train_x.shape[0]
        self._mean_1 = np.mean(c1_train_x, axis = 0) # 1 * dim
        self._mean_2 = np.mean(c2_train_x, axis = 0)
        cov_1 = np.cov(c1_train_x.T) # dim * dim
        cov_2 = np.cov(c2_train_x.T)
        N_sum = self._N_1 + self._N_2
        cov = (self._N_1/N_sum) * cov_1 + (self._N_2/N_sum) * cov_2
        self._icov = pinv(cov)

class logistic_regression:
    def __init__(self):
        self._params = {
            'l_rate': 0.1,
            'epochs': 250,
            'lamb': 0, # regularization param
            'batch_size': 32 }
        self._w = None

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0],1)),X), axis=1)
        z = np.dot(self._w, X.T)
        return [0 if i <0.5 else 1 for i in sigmoid(z)]

    def fit(self, train_x, train_y):
        params = self._params
        train_x = np.concatenate((np.ones((train_x.shape[0],1)),train_x), axis=1)
        N, dim = train_x.shape
        w = np.zeros(dim)
        square_g = np.zeros(dim)+(1e-5)
        step_num = int(floor(N / params['batch_size']))
        for epoch in range(1, params['epochs']+1):
            for start in range(step_num):
                X = train_x[start*params['batch_size'] : (start+1)*params['batch_size']]
                Y = train_y[start*params['batch_size'] : (start+1)*params['batch_size']]
                z = np.dot(w, X.T)
                gra = np.dot((sigmoid(z) - Y), X)\
                 + 2 * params['lamb'] * np.concatenate((np.array([0]),w[1:]))
                square_g += gra ** 2
                w -= params['l_rate'] * gra / np.sqrt(square_g)

        self._w = w

def main(args):
    global valid_num
    valid_num = args.valid_num
    train_x, train_y, valid_x, valid_y, test_x = \
    load_data(args.train_x_path, args.train_y_path, args.test_x_path)

    if args.output_path is None:
        print('Error: Argument --output_path for the path of prediction result')
        return
    if args.model is None:
        print('Error: Argument --model [model] to decide the applied model')
        return
    elif args.model not in ['generative', 'logistic']:
        print(f'Error: Selected model {args.model} does not exist')
        return

    if args.model == 'generative':
        model = generative_model()
    elif args.model == 'logistic':
        model = logistic_regression()

    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)
    save_prediction(pred_y, args.output_path)

    if args.evaluate:
        evaluate(model, train_x, train_y, valid_x, valid_y)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, dest='model',
                        help='Which model to apply')
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
