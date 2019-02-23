from main import *
import xgboost as xgb

def main():
    with open('data/rt-polarity.pos.txt', encoding='utf-8') as f:
        neg_texts = f.read().splitlines()   # f.read()读取整个文件, splitlines() 按行分割，形成list
    with open('data/rt-polarity.neg.txt', encoding='utf-8') as f:
        pos_texts = f.read().splitlines()

    list_texts = filter_text(neg_texts + pos_texts)

    counter = collections.Counter(list_texts)
    list_word_count = counter.most_common(1000)
    vocab = [x[0] for x in list_word_count]
    word2idx = {c: i for i, c in enumerate(vocab)}
    t_pos = sentences(pos_texts, word2idx, 1, vocab)
    t_neg = sentences(neg_texts, word2idx, 0, vocab)
    t_all = np.row_stack((t_pos, t_neg))
    train_data, test_data = split_data(t_all)
    data_train = xgb.DMatrix(train_data[:, :-1], train_data[:, -1])
    data_test = xgb.DMatrix(test_data[:, :-1], test_data[:, -1])
    param = {'max_depth': 4, 'eta': 0.99, 'gamma': 0.6, 'subsample': 0.95, 'lambda': 1, 'min_child_weight': 1,
             'max_bin': 390, 'alpha': 0.3, 'colsample_bytree': 0.7, 'objective': 'binary:logistic'}
    watchlist = [(data_train, 'train'), (data_test, 'test')]

    booster = xgb.train(param, data_train, num_boost_round=400, evals=watchlist, early_stopping_rounds=50)
    y_predicted = booster.predict(data_test)
    y = data_test.get_label()
    accuracy = sum(y == (y_predicted > 0.5))
    accuracy_rate = float(accuracy) / len(y_predicted)
    print('best_ntree_limit: ', booster.best_ntree_limit)
    print('正确率：{0:.3f}'.format(accuracy_rate))
    return

if __name__=="__main__":
    main()