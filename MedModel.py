import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import l2_regularizer
import time

_TEST_RATIO = 0.2


class MedModel(object):
    def __init__(self,
                 n_disease,
                 n_drug,
                 n_visit,
                 n_disease_category,
                 n_embed=500,
                 n_rnn=(200, 200),
                 rnn_type='',
                 bn_decay=0.99,
                 l2_scale=0.001,
                 dropout_rate=0.2):
        self.n_disease = n_disease
        self.n_drug = n_drug
        self.n_visit = n_visit
        self.n_disease_category = n_disease_category
        self.n_embed = n_embed
        self.n_rnn = n_rnn
        self.rnn_type = rnn_type
        self.bn_decay = bn_decay
        self.l2_scale = l2_scale
        self.dropout_rate = dropout_rate

    def load_data(self, data_path):
        dataset = np.load(data_path)
        train_info, test_info = train_test_split(dataset['dataset_info'],
                                                 test_size=_TEST_RATIO, random_state=0)
        train_disease, test_disease = train_test_split(dataset['dataset_disease'],
                                                       test_size=_TEST_RATIO, random_state=0)
        train_drug, test_drug = train_test_split(dataset['dataset_drug'],
                                                 test_size=_TEST_RATIO, random_state=0)
        train_label_disease, test_label_disease = train_test_split(dataset['label_disease'],
                                                                   test_size=_TEST_RATIO, random_state=0)
        train_label_probability, test_label_probability = train_test_split(dataset['label_probability'],
                                                                           test_size=_TEST_RATIO, random_state=0)

        return train_info, test_info, train_disease, test_disease, train_drug, test_drug, \
               train_label_disease, test_label_disease, train_label_probability, test_label_probability

    def build_model(self, x_disease, x_drug, probability_label, disease_label):
        temp_x = tf.concat([x_disease, x_drug], axis=2)
        for i, rnn_dim in enumerate(self.n_rnn[:-1]):
            with tf.variable_scope('model_rnn' + str(i + 1), regularizer=l2_regularizer(self.l2_scale)):
                if self.rnn_type == 'gru':
                    rnn_cell = tf.contrib.rnn.GRUCell(num_units=rnn_dim)
                elif self.rnn_type == 'lstm':
                    rnn_cell = tf.contrib.rnn.LSTMCell(num_units=rnn_dim)
                else:
                    rnn_cell = tf.contrib.rnn.BaiscCell(num_units=rnn_dim)
                rnn_output = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=temp_x, dtype=tf.float32)[0]
                dropout_output = tf.nn.dropout(rnn_output, keep_prob=1 - self.dropout_rate)
                temp_x = dropout_output
        rnn_cell = tf.contrib.rnn.GRUCell(num_units=self.n_rnn[-1])
        rnn_output = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=temp_x, dtype=tf.float32)[0]
        dropout_output = tf.nn.dropout(rnn_output[:, -1, :], keep_prob=1 - self.dropout_rate)
        temp_x = dropout_output
        w_probability = tf.get_variable('w_probability', shape=[self.n_rnn[-1], 1])
        b_probability = tf.get_variable('b_probability', shape=[1])
        probability_y = tf.add(tf.matmul(temp_x, w_probability), b_probability)
        w_disease = tf.get_variable('w_disease', shape=[self.n_rnn[-1], self.n_disease_category])
        b_disease = tf.get_variable('b_disease', shape=[self.n_disease_category])
        disease_y = tf.add(tf.matmul(temp_x, w_disease), b_disease)
        probability_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=probability_y, labels=probability_label))
        disease_loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=disease_y, labels=disease_label))
        return probability_loss, disease_loss, disease_y, probability_y

    def print2file(self, content, out_file):
        outfd = open(out_file, 'a')
        outfd.write(content + '\n')
        outfd.close()

    def generate_data(self,
                      n_samples=100,
                      model_file='model',
                      batch_size=100,
                      out_file='out'):
        x_dummy = tf.placeholder('float', [None, self.input_dim])
        _, decode_variables = self.build_autoencoder(x_dummy)
        x_random = tf.placeholder('float', [None, self.random_dim])
        bn_train = tf.placeholder('bool')
        x_emb = self.build_generator_test(x_random, bn_train)
        temp_vec = x_emb
        i = 0
        for _ in self.decompress_dims[:-1]:
            temp_vec = self.ae_activation(
                tf.add(tf.matmul(temp_vec, decode_variables['aed_W_' + str(i)]), decode_variables['aed_b_' + str(i)]))
            i += 1

        if self.data_type == 'binary':
            x_reconst = tf.nn.sigmoid(
                tf.add(tf.matmul(temp_vec, decode_variables['aed_W_' + str(i)]), decode_variables['aed_b_' + str(i)]))
        else:
            x_reconst = tf.nn.relu(
                tf.add(tf.matmul(temp_vec, decode_variables['aed_W_' + str(i)]), decode_variables['aed_b_' + str(i)]))

        np.random.seed(12345)
        saver = tf.train.Saver()
        output_vec = []
        burn_in = 1000
        with tf.Session() as sess:
            saver.restore(sess, model_file)
            # print('burning in')
            # for i in range(burn_in):
            #     random_x = np.random.normal(size=(batch_size, self.random_dim))
            #     output = sess.run(x_reconst, feed_dict={x_random: random_x, bn_train: True})

            print('generating')
            n_batches = int(np.ceil(float(n_samples)) / float(batch_size))
            for i in range(n_batches):
                random_x = np.random.normal(size=(batch_size, self.random_dim))
                output = sess.run(x_reconst, feed_dict={x_random: random_x, bn_train: False})
                output_vec.extend(output)

        output_mat = np.array(output_vec)
        np.save(out_file, output_mat)
        output_mat = output_mat // 0.51
        output_mat = np.transpose(output_mat)
        np.savetxt('g_data.txt', output_mat)
        result = []
        for i in output_mat:
            result.append(np.sum(i) / n_samples)
        plt.scatter(np.arange(2, 101, 2) / 100, result)
        plt.plot([0, 1], [0, 1])
        plt.show()

    def calculate_auc(self, predict, real):
        auc = roc_auc_score(y_true=real, y_score=predict)
        return auc

    def calculate_accuracy(self, predict, real):
        total = len(predict)
        right = 0
        for i in range(total):
            if np.abs(predict[i] - real[i]) < 0.5:
                right += 1
        acc = float(right) / float(total)
        return acc

    def train(self,
              data_path='',
              model_path='',
              out_path='',
              n_epochs=500,
              batch_size=100,
              save_max_keep=0,
              weights=(50, 1)):
        x_disease = tf.placeholder(tf.float32, [None, self.n_visit, self.n_disease])
        x_drug = tf.placeholder(tf.float32, [None, self.n_visit, self.n_drug])
        y_probability = tf.placeholder(tf.float32, [None, 1])
        y_disease = tf.placeholder(tf.float32, [None, self.n_disease_category])

        probability_loss, disease_loss, disease_y, probability_y = self.build_model(x_disease, x_drug, y_probability,
                                                                                    y_disease)
        disease_y = tf.nn.softmax(disease_y)
        probability_y = tf.nn.sigmoid(probability_y)
        total_loss = probability_loss * weights[0] + disease_loss * weights[1]
        optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1)
        optimize = optimizer.minimize(loss=total_loss)

        init_op = tf.global_variables_initializer()

        train_info, test_info, train_disease, test_disease, train_drug, test_drug, train_label_disease, \
        test_label_disease, train_label_probability, test_label_probability = self.load_data(data_path)
        saver = tf.train.Saver(max_to_keep=save_max_keep)
        log_file = out_path + '.log'

        with tf.Session() as sess:
            # create a log writer. run 'tensorboard --logdir=./logs'
            writer = tf.summary.FileWriter("./logs", sess.graph)  # for 1.0
            if model_path == '':
                sess.run(init_op)
            else:
                saver.restore(sess, model_path)
            n_train_batches = int(np.ceil(float(train_disease.shape[0])) / float(batch_size))
            n_test_batches = int(np.ceil(float(test_disease.shape[0])) / float(batch_size))

            # 训练生成器和判别器
            idx = np.arange(train_disease.shape[0])
            for epoch in range(n_epochs):
                loss_vec = []
                probability_loss_vec = []
                disease_loss_vec = []
                for i in range(n_train_batches):
                    batch_idx = np.random.choice(idx, size=batch_size, replace=False)
                    batch_x_disease = train_disease[batch_idx]
                    batch_x_drug = train_drug[batch_idx]
                    batch_y_probability = train_label_probability[batch_idx]
                    batch_y_disease = train_label_disease[batch_idx]
                    _, loss, loss_probability, loss_disease = sess.run(
                        [optimize, total_loss, probability_loss, disease_loss],
                        feed_dict={x_disease: batch_x_disease, x_drug: batch_x_drug,
                                   y_probability: batch_y_probability,
                                   y_disease: batch_y_disease})
                    loss_vec.append(loss)
                    probability_loss_vec.append(loss_probability)
                    disease_loss_vec.append(loss_disease)

                # 验证集进行判别器验证
                idx = np.arange(len(test_disease))
                test_acc_vec = []
                test_auc_vec = []
                for i in range(n_test_batches):
                    batch_idx = np.random.choice(idx, size=batch_size, replace=False)
                    batch_x_disease = test_disease[batch_idx]
                    batch_x_drug = test_drug[batch_idx]
                    batch_y_probability = test_label_probability[batch_idx]
                    batch_y_disease = test_label_disease[batch_idx]
                    predict_disease, predict_probability = sess.run([disease_y, probability_y],
                                                                    feed_dict={x_disease: batch_x_disease,
                                                                               x_drug: batch_x_drug})
                    test_acc = self.calculate_accuracy(predict_probability, batch_y_probability)
                    test_auc = self.calculate_auc(predict_probability, batch_y_probability)
                    test_acc_vec.append(test_acc)
                    test_auc_vec.append(test_auc)
                buffer = 'Epoch:%d, loss:%f, probability_loss:%f, disease_loss:%f, test_accuracy:%f, test_AUC:%f' % (
                    epoch, np.mean(loss_vec), np.mean(probability_loss_vec), np.mean(disease_loss_vec),
                    np.mean(test_acc_vec), np.mean(test_auc_vec))
                print(buffer)
                self.print2file(buffer, log_file)
                save_path = saver.save(sess, out_path, global_step=epoch)
        print(save_path)


def get_config():
    model_config = dict()
    model_config['n_disease'] = 1236
    model_config['n_drug'] = 1096
    model_config['n_visit'] = 40
    model_config['n_disease_category'] = 588
    model_config['n_embed'] = 1000
    model_config['n_rnn'] = (500, 500)
    model_config['rnn_type'] = 'gru'
    model_config['data_file'] = './dataset/dataset_3month.npz'
    model_config['out_file'] = './model_result/result'
    model_config['model_file'] = ''
    model_config['n_epoch'] = 20
    model_config['batch_size'] = 100
    model_config['save_max_keep'] = 1
    return model_config


if __name__ == '__main__':
    start = time.clock()
    config = get_config()

    med_model = MedModel(n_disease=config['n_disease'],
                         n_drug=config['n_drug'],
                         n_visit=config['n_visit'],
                         n_disease_category=config['n_disease_category'],
                         n_embed=config['n_embed'],
                         n_rnn=config['n_rnn'],
                         rnn_type=config['rnn_type'])
    med_model.train(data_path=config['data_file'],
                    model_path=config['model_file'],
                    out_path=config['out_file'],
                    n_epochs=config['n_epoch'],
                    batch_size=config['batch_size'],
                    save_max_keep=config['save_max_keep'],
                    weights=(3, 1))
    print(time.clock() - start)
