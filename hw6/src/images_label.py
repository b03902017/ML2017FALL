import argparse
import numpy as np

def load_data(test_path):
    test = []
    with open(test_path) as f:
        for n, line in enumerate(f):
            if n == 0:
                continue
            else:
                line = line.strip('\n').split(',')
                test.append([int(line[1]), int(line[2])])
    test = np.array(test)
    return test

def predict(images_labels, test, output_path):
    with open(output_path, 'w') as f_out:
        print('ID,Ans', file=f_out)
        for i in range(test.shape[0]):
            image1, image2 = test[i]
            ans = 1 if images_labels[image1] == images_labels[image2] else 0
            print('%d,%d'%(i,ans), file=f_out)

def main(args):
    images_labels = np.load(args.label_path)
    print('labels.shape:'+str(images_labels.shape))

    test = load_data(args.test_path)
    print('test.shape:'+str(test.shape))

    if args.output_path is None:
        print('Error: Argument --output_path for the path of prediction result')
        return

    predict(images_labels, test, args.output_path)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_data_path', type=str,
                        default='data/test_case.csv', dest='test_path',
                        help='Path to testing data')
    parser.add_argument('-l', '--label_path', type=str,
                        default='images_labels.npy', dest='label_path',
                        help='Path to load the images labels')
    parser.add_argument('-o', '--output_path', type=str, dest='output_path',
                        help='Path to save the prediction result')
    main(parser.parse_args())
