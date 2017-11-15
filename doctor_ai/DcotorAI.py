import time

import numpy as np
import tensorflow as tf

from data_process import CreateDataset


class DoctorAI(object):
    def __init__(self,
                 n_disease,
                 n_drug,
                 n_visit,
                 n_disease_category,
                 n_drug_category,
                 n_embed=500,
                 n_rnn=(200, 200),
                 dropout_rate=0.2,
                 batch_size=100):
        self.n_disease = n_disease
        self.n_drug = n_drug
        self.n_visit = n_visit
        self.n_disease_category = n_disease_category
        self.n_drug_category = n_drug_category
        self.n_embed = n_embed
        self.n_rnn = n_rnn
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size

    def load_train_data(self, dataset_disease_path='',
                        label_disease_path='',
                        dataset_drug_path='',
                        label_drug_path='',
                        dataset_duration_path='',
                        label_duration_path=''):
        dataset_disease = np.load(dataset_disease_path)
        dataset_disease_train = dataset_disease['train']
        dataset_disease_valid = dataset_disease['valid']
        label_disease = np.load(label_disease_path)
        label_disease_train = label_disease['train']
        label_disease_valid = label_disease['valid']
        dataset_drug = np.load(dataset_drug_path)
        dataset_drug_train = dataset_drug['train']
        dataset_drug_valid = dataset_drug['valid']
        dataset_duration = np.load(dataset_duration_path)
        dataset_duration_train = dataset_duration['train']
        dataset_duration_valid = dataset_duration['valid']
        label_drug = np.load(label_drug_path)
        label_drug_train = label_drug['train']
        label_drug_valid = label_drug['valid']
        label_duration = np.load(label_duration_path)
        label_duration_train = label_duration['train']
        label_duration_valid = label_duration['valid']
        return dataset_disease_train, dataset_disease_valid, dataset_drug_train, dataset_drug_valid, \
               dataset_duration_train, dataset_duration_valid, label_drug_train, label_drug_valid, \
               label_disease_train, label_disease_valid, label_duration_train, label_duration_valid

    def load_test_data(self, dataset_disease_path='',
                       label_disease_path='',
                       dataset_drug_path='',
                       label_drug_path='',
                       dataset_duration_path='',
                       label_duration_path=''):
        dataset_disease = np.load(dataset_disease_path)
        dataset_disease_test = dataset_disease['test']
        label_disease = np.load(label_disease_path)
        label_disease_test = label_disease['test']
        dataset_drug = np.load(dataset_drug_path)
        dataset_drug_test = dataset_drug['test']
        dataset_duration = np.load(dataset_duration_path)
        dataset_duration_test = dataset_duration['test']
        label_drug = np.load(label_drug_path)
        label_drug_test = label_drug['test']
        label_duration = np.load(label_duration_path)
        label_duration_test = label_duration['test']
        return dataset_disease_test, dataset_drug_test, dataset_duration_test, label_drug_test, \
               label_disease_test, label_duration_test

    def precision_top(self, y_true, y_predict, rank=None):
        if rank is None:
            rank = [5, 10, 15]
        pre = list()
        for i in range(len(y_predict)):
            this_one = list()
            count = 0
            for j in y_true[i]:
                if j == 1:
                    count += 1
            if count:
                codes = np.argsort(y_true[i])
                tops = np.argsort(y_predict[i])
                for rk in rank:
                    if len(set(codes[len(codes) - count:]).intersection(set(tops[len(tops) - rk:]))) >= 1:
                        this_one.append(1)
                    else:
                        this_one.append(0)
                pre.append(this_one)
        return (np.array(pre)).mean(axis=0).tolist()

    def calculate_r_squared(self, true_vec, predict_vec):
        true_vec = np.reshape(true_vec,[-1])
        predict_vec = np.reshape(predict_vec,[-1])
        mean_duration = np.mean(true_vec)
        numerator = ((true_vec - predict_vec) ** 2).sum()
        denominator = ((true_vec - mean_duration) ** 2).sum()
        return 1.0 - (numerator / denominator)

    def gru_unit(self, hidden_size, dropout_rate):
        gru_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
        gru_cell = tf.nn.rnn_cell.DropoutWrapper(cell=gru_cell, input_keep_prob=1.0, output_keep_prob=1 - dropout_rate)
        return gru_cell

    def build_model(self, x_disease, x_drug, x_duration, disease_label, drug_label, duration_label):
        x_code = tf.concat([x_disease, x_drug], axis=2)
        x_code = tf.reshape(x_code, [-1, self.n_disease + self.n_drug])
        w_embed = tf.get_variable('w_embed',
                                  shape=[self.n_disease + self.n_drug, self.n_embed])
        b_embed = tf.get_variable('b_mebed', shape=[self.n_embed])
        hidden = tf.nn.tanh(tf.add(tf.matmul(x_code, w_embed), b_embed))
        hidden = tf.reshape(hidden, [-1, self.n_visit, self.n_embed])
        hidden = tf.concat([hidden, x_duration], axis=2)
        stacked_gru = tf.nn.rnn_cell.MultiRNNCell(
            [self.gru_unit(self.n_rnn[0], self.dropout_rate) for i in range(len(self.n_rnn))], state_is_tuple=True)
        outputs, states = tf.nn.dynamic_rnn(cell=stacked_gru, inputs=hidden, dtype=tf.float32)

        w_code = tf.get_variable('w_code',
                                 shape=[self.n_rnn[-1], self.n_disease_category + self.n_drug_category])
        b_code = tf.get_variable('b_code', shape=[self.n_disease_category + self.n_drug_category])
        code_label = tf.concat([disease_label, drug_label], axis=2)
        code_loss = tf.reduce_sum(
            tf.nn.softmax_cross_entropy_with_logits(logits=tf.add(tf.matmul(outputs[:, -1, :], w_code), b_code),
                                                    labels=code_label[:, -1, :]))
        for i in range(self.n_visit - 1):
            code_y = tf.add(tf.matmul(outputs[:, i, :], w_code), b_code)
            code_loss += tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(logits=code_y, labels=code_label[:, i, :]))
        code_y_last = tf.nn.softmax(tf.add(tf.matmul(outputs[:, -1, :], w_code), b_code))

        w_duration = tf.get_variable('w_duration',
                                     shape=[self.n_rnn[-1], 1])
        b_duration = tf.get_variable('b_duration', shape=[1])
        duration_loss = tf.reduce_sum(tf.pow(tf.nn.relu(tf.add(tf.matmul(outputs[:, -1, :], w_duration), b_duration))
                                             - duration_label[:, -1, :], 2))
        for i in range(self.n_visit - 1):
            duration_y = tf.nn.relu(tf.add(tf.matmul(outputs[:, i, :], w_duration), b_duration))
            duration_loss += tf.reduce_sum(tf.pow(duration_y - duration_label[:, i, :], 2))
        duration_y_last = tf.nn.relu(tf.add(tf.matmul(outputs[:, -1, :], w_duration), b_duration))
        return code_loss, duration_loss, code_y_last, duration_y_last

    def print2file(self, content, out_file):
        outfd = open(out_file, 'a')
        outfd.write(content + '\n')
        outfd.close()

    def train(self,
              dataset_disease_path='',
              label_disease_path='',
              dataset_drug_path='',
              label_drug_path='',
              dataset_duration_path='',
              label_duration_path='',
              model_path='',
              out_path='',
              n_epochs=20,
              save_max_keep=0,
              weights=(50, 1)):
        x_disease = tf.placeholder(tf.float32, [None, self.n_visit, self.n_disease])
        x_drug = tf.placeholder(tf.float32, [None, self.n_visit, self.n_drug])
        x_duration = tf.placeholder(tf.float32, [None, self.n_visit, 1])
        y_disease = tf.placeholder(tf.float32, [None, self.n_visit, self.n_disease_category])
        y_drug = tf.placeholder(tf.float32, [None, self.n_visit, self.n_drug_category])
        y_duration = tf.placeholder(tf.float32, [None, self.n_visit, 1])
        code_loss, duration_loss, code_y, duration_y = self.build_model(x_disease, x_drug, x_duration, y_disease,
                                                                        y_drug, y_duration)
        total_loss = weights[0] * code_loss + weights[1] * duration_loss
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.01)
        optimize = optimizer.minimize(loss=total_loss)

        init_op = tf.global_variables_initializer()
        dataset_disease_train, dataset_disease_valid, dataset_drug_train, dataset_drug_valid, \
        dataset_duration_train, dataset_duration_valid, label_drug_train, label_drug_valid, \
        label_disease_train, label_disease_valid, label_duration_train, label_duration_valid \
            = self.load_train_data(dataset_disease_path, label_disease_path, dataset_drug_path,
                                   label_drug_path, dataset_duration_path, label_duration_path)
        saver = tf.train.Saver(max_to_keep=save_max_keep)
        log_file = out_path + '.log'

        with tf.Session() as sess:
            # create a log writer. run 'tensorboard --logdir=./logs'
            # writer = tf.summary.FileWriter("./logs", sess.graph)  # for 1.0
            if model_path == '':
                sess.run(init_op)
            else:
                saver.restore(sess, model_path)
            n_train_batches = int(np.ceil(float(dataset_disease_train.shape[0])) / float(self.batch_size))
            n_valid_batches = int(np.ceil(float(dataset_disease_valid.shape[0])) / float(self.batch_size))

            # 训练集进行训练
            idx = np.arange(dataset_disease_train.shape[0])
            for epoch in range(n_epochs):
                loss_vec = []
                code_loss_vec = []
                duration_loss_vec = []
                print("epoch:", epoch + 1)
                for i in range(n_train_batches):
                    batch_idx = np.random.choice(idx, size=self.batch_size, replace=False)
                    batch_x_disease = dataset_disease_train[batch_idx]
                    batch_x_drug = dataset_drug_train[batch_idx]
                    batch_x_duration = dataset_duration_train[batch_idx]
                    batch_y_disease = label_disease_train[batch_idx]
                    batch_y_drug = label_drug_train[batch_idx]
                    batch_y_duration = label_duration_train[batch_idx]
                    _, loss, loss_code, loss_duration = sess.run(
                        [optimize, total_loss, code_loss, duration_loss],
                        feed_dict={x_disease: batch_x_disease, x_drug: batch_x_drug, x_duration: batch_x_duration,
                                   y_disease: batch_y_disease, y_drug: batch_y_drug, y_duration: batch_y_duration})
                    loss_vec.append(loss)
                    code_loss_vec.append(loss_code)
                    duration_loss_vec.append(loss_duration)
                print("loss:", np.mean(loss_vec), " code_loss:", np.mean(code_loss_vec), " duration_loss:",
                      np.mean(duration_loss_vec))

                # 验证集进行验证
                idx = np.arange(len(dataset_disease_valid))
                top_precision_vec = []
                r_squared_vec = []
                for i in range(n_valid_batches):
                    batch_idx = np.random.choice(idx, size=self.batch_size, replace=False)
                    batch_x_disease = dataset_disease_valid[batch_idx]
                    batch_x_drug = dataset_drug_valid[batch_idx]
                    batch_x_duration = dataset_duration_valid[batch_idx]
                    batch_y_disease = label_disease_valid[batch_idx]
                    batch_y_drug = label_drug_valid[batch_idx]
                    batch_y_duration = label_duration_valid[batch_idx]
                    predict_code, predict_duration = sess.run(
                        [code_y, duration_y],
                        feed_dict={x_disease: batch_x_disease, x_drug: batch_x_drug, x_duration: batch_x_duration,
                                   y_disease: batch_y_disease, y_drug: batch_y_drug, y_duration: batch_y_duration})
                    top_precision_vec.append(
                        self.precision_top(np.concatenate([batch_y_disease[:, -1, :], batch_y_drug[:, -1, :]], axis=1),
                                           predict_code))
                    r_squared_vec.append(self.calculate_r_squared(batch_y_duration[:, -1, :], predict_duration))
                print("valid_top_precision:", np.mean(top_precision_vec,axis=0), " valid_r_squared:", np.mean(r_squared_vec))
                # self.print2file(buffer, log_file)
                # save_path = saver.save(sess, out_path, global_step=epoch)
                # print(save_path)


def get_config():
    model_config = dict()
    # 模型参数
    model_config['n_disease'] = CreateDataset.disease_num
    model_config['n_drug'] = CreateDataset.drug_num
    model_config['n_visit'] = CreateDataset.visit_num
    model_config['n_disease_category'] = CreateDataset.disease_category_num
    model_config['n_drug_category'] = CreateDataset.drug_category_num
    model_config['n_embed'] = 600
    model_config['n_rnn'] = (300, 300)
    model_config['dropout_rate'] = 0.2
    model_config['use_duration'] = True
    model_config['predict_drug'] = True
    model_config['predict_duration'] = True
    # 数据集路径
    model_config['disease_file'] = '../dataset/dataset_disease.npz'
    model_config['disease_label'] = '../dataset/label_disease_categ.npz'
    model_config['drug_file'] = '../dataset/dataset_drug.npz'
    model_config['drug_label'] = '../dataset/label_drug_categ.npz'
    model_config['duration_file'] = '../dataset/dataset_duration.npz'
    model_config['duration_label'] = '../dataset/label_duration.npz'
    # 训练参数
    model_config['out_file'] = '../model_result/result'
    model_config['model_file'] = ''
    model_config['n_epoch'] = 20
    model_config['batch_size'] = 500
    model_config['save_max_keep'] = 1
    model_config['weights'] = (1, 0.0001)
    return model_config


if __name__ == '__main__':
    start = time.clock()
    config = get_config()
    med_model = DoctorAI(n_disease=config['n_disease'],
                         n_drug=config['n_drug'],
                         n_visit=config['n_visit'],
                         n_disease_category=config['n_disease_category'],
                         n_drug_category=config['n_drug_category'],
                         n_embed=config['n_embed'],
                         n_rnn=config['n_rnn'],
                         dropout_rate=config['dropout_rate'],
                         batch_size=config['batch_size'])
    med_model.train(dataset_disease_path=config['disease_file'],
                    label_disease_path=config['disease_label'],
                    dataset_drug_path=config['drug_file'],
                    label_drug_path=config['drug_label'],
                    dataset_duration_path=config['duration_file'],
                    label_duration_path=config['duration_label'],
                    model_path=config['model_file'],
                    out_path=config['out_file'],
                    n_epochs=config['n_epoch'],
                    save_max_keep=config['save_max_keep'],
                    weights=config['weights'])
    print(time.clock() - start)
