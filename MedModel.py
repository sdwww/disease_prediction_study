import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import l2_regularizer

_VALIDATION_RATIO = 0.1


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
        data = np.load(data_path)['arr_0']
        print(data.shape)
        trainX, validX = train_test_split(data, test_size=_VALIDATION_RATIO, random_state=0)
        return trainX, validX

    def build_model(self, x_disease, x_drug, probability_label, disease_label):
        temp_x = tf.concat([x_disease, x_drug], axis=2)
        with tf.variable_scope('model', regularizer=l2_regularizer(self.l2_scale)):
            for i, rnn_dim in enumerate(self.n_rnn):
                # W = tf.get_variable('W_' + str(i), shape=[self.n_disease + self.n_drug, rnn_dim])
                gru_cell = tf.contrib.rnn.GRUCell(num_units=rnn_dim)
                rnn_output = tf.nn.dynamic_rnn(cell=gru_cell, inputs=temp_x, dtype=tf.float32, time_major=False)[0]
                dropout_output = tf.nn.dropout(rnn_output, keep_prob=1 - self.dropout_rate)
                temp_x = dropout_output
        w_probability = tf.get_variable('w_probability', shape=[self.n_rnn[-1], 1])
        b_probability = tf.get_variable('b_probability', shape=[1])
        probability_y = tf.add(tf.matmul(temp_x, w_probability), b_probability)
        w_disease = tf.get_variable('w_disease', shape=[self.n_rnn[-1], self.n_disease_category])
        b_disease = tf.get_variable('b_disease', shape=[self.n_disease_category])
        disease_y = tf.add(tf.matmul(temp_x, w_disease), b_disease)
        probability_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=probability_y, labels=probability_label)
        disease_loss = tf.nn.softmax_cross_entropy_with_logits(logits=disease_y, labels=disease_label)
        return probability_loss, disease_loss

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

    def calculate_disc_auc(self, preds_real, preds_fake):
        preds = np.concatenate([preds_real, preds_fake], axis=0)
        labels = np.concatenate([np.ones((len(preds_real))), np.zeros((len(preds_fake)))], axis=0)
        auc = roc_auc_score(labels, preds)
        return auc

    def calculate_disc_accuracy(self, preds_real, preds_fake):
        total = len(preds_real) + len(preds_fake)
        hit = 0
        for pred in preds_real:
            if pred > 0.5: hit += 1
        for pred in preds_fake:
            if pred < 0.5: hit += 1
        acc = float(hit) / float(total)
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

        probability_loss, disease_loss = self.build_model(x_disease, x_drug, y_probability, y_disease)

        t_vars = tf.trainable_variables()
        model_vars = [var for var in t_vars if 'model' in var.name]

        loss = probability_loss * weights[0] + disease_loss * weights[1]
        optimize = tf.train.RMSPropOptimizer.minimize(loss, var_list=model_vars)

        init_op = tf.global_variables_initializer()

        train_x, valid_x = self.load_data(data_path)
        n_batches = int(np.ceil(float(train_x.shape[0]) / float(batch_size)))
        saver = tf.train.Saver(max_to_keep=save_max_keep)
        log_file = out_path + '.log'

        with tf.Session() as sess:
            # create a log writer. run 'tensorboard --logdir=./logs'
            writer = tf.summary.FileWriter("./logs", sess.graph)  # for 1.0
            if model_path == '':
                sess.run(init_op)
            else:
                saver.restore(sess, model_path)
            n_train_batches = int(np.ceil(float(train_x.shape[0])) / float(batch_size))
            n_valid_batches = int(np.ceil(float(valid_x.shape[0])) / float(batch_size))

            # 训练生成器和判别器
            idx = np.arange(train_x.shape[0])
            for epoch in range(n_epochs):
                loss_vec = []
                for i in range(n_batches):
                    batch_idx = np.random.choice(idx, size=batch_size, replace=False)
                    batch_x = train_x[batch_idx]
                    _, loss = sess.run([optimize, loss],
                                            feed_dict={x_disease: batch_x, x_drug: random_x, keep_prob: 1.0,
                                                       bn_train: False})
                    loss_vec.append(loss)

                # 验证集进行判别器验证
                idx = np.arange(len(valid_x))
                n_valid_batches = int(np.ceil(float(len(valid_x)) / float(batch_size)))
                valid_acc_vec = []
                valid_auc_vec = []
                for i in range(n_valid_batches):
                    batch_idx = np.random.choice(idx, size=batch_size, replace=False)
                    batch_x = valid_x[batch_idx]
                    random_x = np.random.normal(size=(batch_size, self.random_dim))
                    preds_real, preds_fake, = sess.run([y_hat_real, y_hat_fake],
                                                       feed_dict={x_raw: batch_x, x_random: random_x, keep_prob: 1.0,
                                                                  bn_train: False})
                    valid_acc = self.calculate_disc_accuracy(preds_real, preds_fake)
                    valid_auc = self.calculate_disc_auc(preds_real, preds_fake)
                    valid_acc_vec.append(valid_acc)
                    valid_auc_vec.append(valid_auc)
                buf = 'Epoch:%d, d_loss:%f, g_loss:%f, accuracy:%f, AUC:%f' % (
                    epoch, np.mean(d_loss_vec), np.mean(g_loss_vec), np.mean(valid_acc_vec), np.mean(valid_auc_vec))
                print(buf)
                self.print2file(buf, log_file)
                save_path = saver.save(sess, out_path, global_step=epoch)
        print(save_path)


def get_config():
    model_config = dict()
    model_config['n_disease'] = 1236
    model_config['n_drug'] = 1096
    model_config['n_visit'] = 40
    model_config['n_disease_category'] = 588
    model_config['n_embed'] = 500
    model_config['n_rnn'] = (200, 200)
    model_config['rnn_type'] = 'gru'
    model_config['data_file'] = './data_npz/dataset_jbbm_test.npz'
    model_config['out_file'] = './medGAN_result/result'
    model_config['model_file'] = ''
    model_config['n_epoch'] = 200
    model_config['batch_size'] = 100
    model_config['save_max_keep'] = 10
    return model_config


if __name__ == '__main__':
    config = get_config()

    med_model = MedModel(n_disease=config['n_disease'],
                         n_drug=config['n_drug'],
                         n_visit=config['n_visit'],
                         n_disease_category=config['n_disease_category'],
                         n_embed=config['n_embed'],
                         n_rnn=config['n_rnn'],
                         rnn_type=config['rnn_type'])
    train_x, valid_x = med_model.load_data(config['data_file'])
    # med_model.train(data_path=config['data_file'],
    #                 model_path=config['model_file'],
    #                 out_path=config['out_file'],
    #                 n_epochs=config['n_epoch'],
    #                 batch_size=config['batch_size'],
    #                 save_max_keep=config['save_max_keep'],
    #                 weights=(50, 1))
