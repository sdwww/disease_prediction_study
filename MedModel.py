import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tensorflow.contrib.layers import batch_norm
from tensorflow.contrib.layers import l2_regularizer

_VALIDATION_RATIO = 0.1


class MedModel(object):
    def __init__(self,
                 n_disease,
                 n_drug,
                 n_embed=500,
                 n_rnn=(200, 200),
                 rnn_type='',
                 n_disease_category=100,
                 bn_decay=0.99,
                 l2_scale=0.001):

        self.input_dim = n_disease
        self.n_drug = n_drug
        self.n_embed = n_embed
        self.n_disease_category = n_disease_category
        self.bn_decay = bn_decay
        self.l2_scale = l2_scale

    def load_data(self, data_path):
        data = np.load(data_path)['arr_0']

        if self.data_type == 'binary':
            data = np.clip(data, 0, 1)

        trainX, validX = train_test_split(data, test_size=_VALIDATION_RATIO, random_state=0)
        return trainX, validX

    def build_autoencoder(self, x_input):
        decode_variables = {}
        with tf.variable_scope('autoencoder', regularizer=l2_regularizer(self.l2_scale)):
            temp_vec = x_input
            temp_dim = self.input_dim
            i = 0
            for compress_dim in self.compress_dims:
                W = tf.get_variable('aee_W_' + str(i), shape=[temp_dim, compress_dim])
                b = tf.get_variable('aee_b_' + str(i), shape=[compress_dim])
                temp_vec = self.ae_activation(tf.add(tf.matmul(temp_vec, W), b))
                temp_dim = compress_dim
                i += 1

            i = 0
            for decompress_dim in self.decompress_dims[:-1]:
                W = tf.get_variable('aed_W_' + str(i), shape=[temp_dim, decompress_dim])
                b = tf.get_variable('aed_b_' + str(i), shape=[decompress_dim])
                temp_vec = self.ae_activation(tf.add(tf.matmul(temp_vec, W), b))
                temp_dim = decompress_dim
                decode_variables['aed_W_' + str(i)] = W
                decode_variables['aed_b_' + str(i)] = b
                i += 1
            W = tf.get_variable('aed_W_' + str(i), shape=[temp_dim, self.decompress_dims[-1]])
            b = tf.get_variable('aed_b_' + str(i), shape=[self.decompress_dims[-1]])
            decode_variables['aed_W_' + str(i)] = W
            decode_variables['aed_b_' + str(i)] = b

            if self.data_type == 'binary':
                x_reconst = tf.nn.sigmoid(tf.add(tf.matmul(temp_vec, W), b))
                loss = tf.reduce_mean(-tf.reduce_sum(
                    x_input * tf.log(x_reconst + 1e-12) + (1. - x_input) * tf.log(1. - x_reconst + 1e-12), 1), 0)
            else:
                x_reconst = tf.nn.relu(tf.add(tf.matmul(temp_vec, W), b))
                loss = tf.reduce_mean((x_input - x_reconst) ** 2)
        return loss, decode_variables

    def build_generator(self, x_input, bn_train):
        temp_vec = x_input
        temp_dim = self.random_dim
        with tf.variable_scope('generator', regularizer=l2_regularizer(self.l2_scale)):
            for i, gen_dim in enumerate(self.generator_dims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[temp_dim, gen_dim])
                h = tf.matmul(temp_vec, W)
                h2 = batch_norm(h, decay=self.bn_decay, scale=True, is_training=bn_train, updates_collections=None)
                h3 = self.generator_activation(h2)
                temp_vec = h3 + temp_vec
                temp_dim = gen_dim
            W = tf.get_variable('W' + str(i), shape=[temp_dim, self.generator_dims[-1]])
            h = tf.matmul(temp_vec, W)
            h2 = batch_norm(h, decay=self.bn_decay, scale=True, is_training=bn_train, updates_collections=None)

            if self.data_type == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + temp_vec
        return output

    def build_generator_test(self, x_input, bn_train):
        temp_vec = x_input
        temp_dim = self.random_dim
        with tf.variable_scope('generator', regularizer=l2_regularizer(self.l2_scale)):
            for i, genDim in enumerate(self.generator_dims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[temp_dim, genDim])
                h = tf.matmul(temp_vec, W)
                h2 = batch_norm(h, decay=self.bn_decay, scale=True, is_training=bn_train, updates_collections=None,
                                trainable=False)
                h3 = self.generator_activation(h2)
                temp_vec = h3 + temp_vec
                temp_dim = genDim
            W = tf.get_variable('W' + str(i), shape=[temp_dim, self.generator_dims[-1]])
            h = tf.matmul(temp_vec, W)
            h2 = batch_norm(h, decay=self.bn_decay, scale=True, is_training=bn_train, updates_collections=None,
                            trainable=False)

            if self.data_type == 'binary':
                h3 = tf.nn.tanh(h2)
            else:
                h3 = tf.nn.relu(h2)

            output = h3 + temp_vec
        return output

    def get_discriminator_results(self, x_input, keep_rate, reuse=False):
        batch_size = tf.shape(x_input)[0]
        input_mean = tf.reshape(tf.tile(
            tf.reduce_mean(x_input, 0), [batch_size]), (batch_size, self.input_dim))
        temp_vec = tf.concat([x_input, input_mean], axis=1)
        temp_dim = self.input_dim * 2
        with tf.variable_scope('discriminator', reuse=reuse, regularizer=l2_regularizer(self.l2_scale)):
            for i, disc_dim in enumerate(self.discriminator_dims[:-1]):
                W = tf.get_variable('W_' + str(i), shape=[temp_dim, disc_dim])
                b = tf.get_variable('b_' + str(i), shape=[disc_dim])
                h = self.discriminator_activation(tf.add(tf.matmul(temp_vec, W), b))
                h = tf.nn.dropout(h, keep_rate)
                temp_vec = h
                temp_dim = disc_dim
            W = tf.get_variable('W', shape=[temp_dim, 1])
            b = tf.get_variable('b', shape=[1])
            y_hat = tf.squeeze(tf.nn.sigmoid(tf.add(tf.matmul(temp_vec, W), b)))
        return y_hat

    def build_discriminator(self, x_real, x_fake, keepRate, decodeVariables, bn_train):
        # Discriminate for real samples
        y_hat_real = self.get_discriminator_results(x_real, keepRate, reuse=False)
        # Decompress, then discriminate for real samples
        temp_vec = x_fake
        i = 0
        for _ in self.decompress_dims[:-1]:
            temp_vec = self.ae_activation(
                tf.add(tf.matmul(temp_vec, decodeVariables['aed_W_' + str(i)]), decodeVariables['aed_b_' + str(i)]))
            i += 1

        if self.data_type == 'binary':
            x_decoded = tf.nn.sigmoid(
                tf.add(tf.matmul(temp_vec, decodeVariables['aed_W_' + str(i)]), decodeVariables['aed_b_' + str(i)]))
        else:
            x_decoded = tf.nn.relu(
                tf.add(tf.matmul(temp_vec, decodeVariables['aed_W_' + str(i)]), decodeVariables['aed_b_' + str(i)]))

        y_hat_fake = self.get_discriminator_results(x_decoded, keepRate, reuse=True)

        loss_d = -tf.reduce_mean(tf.log(y_hat_real + 1e-12)) - tf.reduce_mean(tf.log(1. - y_hat_fake + 1e-12))
        loss_g = -tf.reduce_mean(tf.log(y_hat_fake + 1e-12))

        return loss_d, loss_g, y_hat_real, y_hat_fake

    def print2file(self, buf, out_file):
        outfd = open(out_file, 'a')
        outfd.write(buf + '\n')
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
              discriminator_train_period=2,
              generator_train_period=1,
              pretrain_batch_size=100,
              batch_size=100,
              pretrain_epochs=100,
              save_max_keep=0):
        x_raw = tf.placeholder(tf.float32, [None, self.input_dim])
        x_random = tf.placeholder(tf.float32, [None, self.random_dim])
        keep_prob = tf.placeholder(tf.float32, [None, self.])
        bn_train = tf.placeholder('bool')

        loss_ae, decode_variables = self.build_autoencoder(x_raw)
        x_fake = self.build_generator(x_random, bn_train)
        loss_d, loss_g, y_hat_real, y_hat_fake = self.build_discriminator(x_raw, x_fake, keep_prob, decode_variables,
                                                                          bn_train)
        train_x, valid_x = self.load_data(data_path)

        t_vars = tf.trainable_variables()
        ae_vars = [var for var in t_vars if 'autoencoder' in var.name]
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]

        all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        optimize_ae = tf.train.AdamOptimizer().minimize(loss_ae + sum(all_regs), var_list=ae_vars)
        optimize_d = tf.train.AdamOptimizer().minimize(loss_d + sum(all_regs), var_list=d_vars)
        optimize_g = tf.train.AdamOptimizer().minimize(loss_g + sum(all_regs),
                                                       var_list=g_vars.append(decode_variables.values()))

        init_op = tf.global_variables_initializer()

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
            n_train_batches = int(np.ceil(float(train_x.shape[0])) / float(pretrain_batch_size))
            n_valid_batches = int(np.ceil(float(valid_x.shape[0])) / float(pretrain_batch_size))

            # 训练自动编码器
            if model_path == '':
                for epoch in range(pretrain_epochs):
                    # 乱序排列
                    idx = np.random.permutation(train_x.shape[0])
                    train_loss_vec = []
                    for i in range(n_train_batches):
                        batch_x = train_x[idx[i * pretrain_batch_size:(i + 1) * pretrain_batch_size]]
                        _, loss = sess.run([optimize_ae, loss_ae], feed_dict={x_raw: batch_x})
                        train_loss_vec.append(loss)
                    idx = np.random.permutation(valid_x.shape[0])
                    valid_loss_vec = []
                    for i in range(n_valid_batches):
                        batch_x = valid_x[idx[i * pretrain_batch_size:(i + 1) * pretrain_batch_size]]
                        loss = sess.run(loss_ae, feed_dict={x_raw: batch_x})
                        valid_loss_vec.append(loss)
                    buf = 'Pretrain_Epoch:%d, trainLoss:%f, validLoss:%f ' \
                          % (epoch, np.mean(train_loss_vec), np.mean(valid_loss_vec))
                    print(buf)
                    self.print2file(buf, log_file)

            # 训练生成器和判别器
            idx = np.arange(train_x.shape[0])
            for epoch in range(n_epochs):
                d_loss_vec = []
                g_loss_vec = []
                for i in range(n_batches):
                    for _ in range(discriminator_train_period):
                        batch_idx = np.random.choice(idx, size=batch_size, replace=False)
                        batch_x = train_x[batch_idx]
                        random_x = np.random.normal(size=(batch_size, self.random_dim))
                        _, disc_loss = sess.run([optimize_d, loss_d],
                                                feed_dict={x_raw: batch_x, x_random: random_x, keep_prob: 1.0,
                                                           bn_train: False})
                        d_loss_vec.append(disc_loss)
                    for _ in range(generator_train_period):
                        random_x = np.random.normal(size=(batch_size, self.random_dim))
                        _, generator_loss = sess.run([optimize_g, loss_g],
                                                     feed_dict={x_random: random_x, keep_prob: 1.0,
                                                                bn_train: True})
                        g_loss_vec.append(generator_loss)

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
    model_config['n_disease_category']=588
    model_config['n_embed'] = 500
    model_config['n_rnn']=(200,200)
    model_config['batch_norm_decay'] = 0.99
    model_config['L2'] = 0.001
    model_config['data_file'] = './random_data.npz'
    model_config['out_file'] = './medGAN_result/result'
    model_config['model_file'] = ''
    model_config['n_epoch'] = 200
    model_config['batch_size'] = 100
    model_config['save_max_keep'] = 10
    return model_config


if __name__ == '__main__':
    config = get_config()

    mg = MedModel(n_disease=config['n_disease'],
                  n_drug=config['n_drug'],
                  n_disease_category=config['n_disease_category'],
                  n_embed=config['n_embed'],
                  n_rnn=config['n_rnn'],
                  rnn_type='')

    mg.train(data_path=config['data_file'],
             model_path=config['model_file'],
             out_path=config['out_file'],
             pretrain_epochs=config['n_pretrain_epoch'],
             n_epochs=config['n_epoch'],
             discriminator_train_period=config['n_discriminator_update'],
             generator_train_period=config['n_generator_update'],
             pretrain_batch_size=config['pretrain_batch_size'],
             batch_size=config['batch_size'],
             save_max_keep=config['save_max_keep'])
