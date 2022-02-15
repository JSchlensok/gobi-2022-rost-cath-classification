from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
import os
import numpy as np
import Parameters
import PrepareData
import argparse
from keras.models import model_from_json


def xroc(res, cutoff):
    """
    :type res: List[List[label, score]]
    :type curoff: all or 50
    """
    area, height, fp, tp = 0.0, 0.0, 0.0, 0.0
    for x in res:
        label = x
        if cutoff > fp:
            if label == 1:
                height += 1
                tp += 1
            else:
                area += height
                fp += 1
        else:
            if label == 1:
                tp += 1
    lroc = 0
    if fp != 0 and tp != 0:
        lroc = area / (fp * tp)
    elif fp == 0 and tp != 0:
        lroc = 1
    elif fp != 0 and tp == 0:
        lroc = 0
    return lroc


def get_roc(y_true, y_pred, cutoff):
    '''

    :param y_true: 
    :param y_pred: 
    :param cutoff: 
    :return: 
    '''
    score = []
    label = []

    for i in range(y_pred.shape[0]):
        label.append(y_true[i])
        score.append(y_pred[i][0])

    index = np.argsort(score)
    index = index[::-1]
    t_score = []
    t_label = []
    for i in index:
        t_score.append(score[i])
        t_label.append(label[i])

    score = xroc(t_label, cutoff)
    return score


def get_graph():
    sequence_length = Parameters.sequence_length
    latent_dim = Parameters.nb_of_cells
    window_size = Parameters.window_size
    model = Sequential()
    model.add(Bidirectional(LSTM(latent_dim, dropout=0.2, recurrent_dropout=0.2, return_sequences=True),
                            merge_mode='concat', weights=None,input_shape=(sequence_length, 20 * window_size,)),)
    model.add(TimeDistributed(Dense(1, activation='relu')))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='RMSprop',
                  metrics=['accuracy'])
    return model


def writeResults(test_names, test_label, best_score, best_predicted_label, target_fam, best_roc, save_roc50):
    result_path = 'results/' + target_fam + '/result/'
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    writer = open(result_path + 'result.txt', 'w')
    writer.write('protein\ttrue label\tscore\tprediction\n')
    for j in range(len(test_names)):
        result_file = open(result_path + test_names[j] + '.txt', 'w')
        result_file.write('protein\ttrue label\tscore\tprediction\n')
        result_file.write(test_names[j] + '\t' + str(test_label[j]) + '\t' + str(best_score[j][0]) + '\t' + str(
            best_predicted_label[j][0]) + '\n')
        writer.write(test_names[j] + '\t' + str(test_label[j]) + '\t' + str(best_score[j][0]) + '\t' + str(
            best_predicted_label[j][0]) + '\n')
    writer.write('roc:\t' + str(best_roc) + '\troc50:\t' + str(save_roc50) + '\n')


def TrainingModel(args):
    target_fam = args.family_index
    nb_epochs = Parameters.nb_of_epochs

    ROOT = 'results/' + target_fam + '/'
    if not os.path.exists(ROOT):
        os.mkdir(ROOT)
    prepare = PrepareData.prepareData()
    train, train_lable, test, test_label, test_names = prepare.generateInputData(args)
    best_roc = -1
    save_roc50 = -1
    # construct the neural network
    model = get_graph()
    print model.summary()

    model_path =  'models/'+target_fam+'/'
    result_path = 'results/'+target_fam+'/'
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    # save the structure of the neural network
    model_json = model.to_json()
    with open(model_path + 'model' + ".json", "w") as json_file:
        json_file.write(model_json)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for epoch in range(nb_epochs):
        print 'epoch: ', epoch
        loss = model.fit(train, train_lable, epochs=1, shuffle=True)
        x, acc = model.evaluate(test, test_label)
        score = model.predict(test)
        predicted_label = model.predict_classes(test)

        roc = get_roc(test_label, score, score.shape[0])
        roc50 = get_roc(test_label, score, 50)
        print 'roc: ', roc
        # select the epoch with the best performance
        if roc >= best_roc:
            best_roc = roc
            save_roc50 = roc50
            save_epoch = epoch
            best_score = score
            model.save_weights(model_path + str(save_epoch) + ".h5")
            best_predicted_label = predicted_label

    print 'target_famlily:', target_fam
    print 'Performance of ProDec-BLSTM: roc:', best_roc, 'roc50:', save_roc50

    writeResults(test_names, test_label, best_score, best_predicted_label, target_fam, best_roc, save_roc50)
    # save the predictions


def performPredictions(args):
    model_path = args.model_dir
    weights_path = args.weights_dir
    prepare = PrepareData.prepareData()
    test, test_label, test_names = prepare.generateTestingSamples(args)

    # load model file
    jason_file = open(model_path, 'r')
    loaded_jason_file = jason_file.read()
    jason_file.close()
    model = model_from_json(loaded_jason_file)

    # load weights file
    model.load_weights(weights_path)
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

    # perform predictions
    x, best_acc = model.evaluate(test, test_label)
    score = model.predict(test)
    predicted_label = model.predict_classes(test)

    roc = get_roc(test_label, score, score.shape[0])
    roc50 = get_roc(test_label, score, 50)

    print 'target_famlily:', args.family_index
    print 'Performance of ProDec-BLSTM: roc:', roc, 'roc50:', roc50

    # save predictions to disk
    writeResults(test_names, test_label, score, predicted_label, args.family_index, roc, roc50)


def parseArguments(parser):
    '''

    parser the input argument 
    '''
    parser.add_argument('-family_index', type=str, help='family index')
    parser.add_argument('-train', type=bool, default=False, help='train a CNN-BLSTM-PSSM')
    parser.add_argument('-test', type=bool, default=False, help=' load the trained CNN-BLSTM-PSSM model')
    parser.add_argument('-model_dir', type=str,
                        help='the directory of the trained model json file of CNN-BLSTM-PSSM. If test is false, this argument can be empty.')
    parser.add_argument('-weights_dir', type=str,
                        help='the directory of the trained model weights file of CNN-BLSTM-PSSM. If test is false, this argument can be empty.')
    parser.add_argument('-pos_train_dir', type=str, help='the directory of positive training dataset')
    parser.add_argument('-neg_train_dir', type=str, help='the directory of negative training dataset')
    parser.add_argument('-pos_test_dir', type=str, help='the directory of positive testing dataset')
    parser.add_argument('-neg_test_dir', type=str, help='the directory of negative testing dataset')
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args = parseArguments(parser)

    target_fam = args.family_index
    if not os.path.exists('results/'+target_fam+'/'):
        os.mkdir('results/'+target_fam+'/')

    # train CNN-BLSTM-PSSM model and test on testing set
    if args.train == True:
        TrainingModel(args)
    elif args.test == True:
        performPredictions(args)