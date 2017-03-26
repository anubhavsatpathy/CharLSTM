import numpy as np
import urllib
import os
import tensorflow as tf

file_url = 'https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt'
file_name = 'shakes_corpus.txt'

if not os.path.exists(file_name):
    urllib.urlretrieve(file_url, file_name)

with open(file_name,'r') as f:
    raw_data = f.read()

ids_chars = {}
chars_ids = {}

for idx,ch in enumerate(set(raw_data)):
    ids_chars[idx] = ch

#print ids_words

for idx,ch in enumerate(set(raw_data)):
    chars_ids[ch] = idx

print chars_ids[' ']
print len(chars_ids)
corp_seq = []

for i in raw_data:
    corp_seq.append(i)

#corp_seq = np.array(corp_seq)
#print np.shape(corp_seq)
print corp_seq[0:10]

X = []
Y = []
seq_len = 200

for i in range(len(corp_seq)):
    j = i+seq_len
    if j<len(corp_seq):
        Y.append(chars_ids[corp_seq[j]])
        k=0
        temp = []
        while k<seq_len:
            temp.append(chars_ids[corp_seq[i+k]])
            k = k+1
        X.append(temp)


X = np.array(X)
Y = np.array(Y)
print np.shape(Y)
print np.shape(X)
print X[0:5]
print Y[0]



batch_size = 200
num_steps = seq_len
state_size = 100
num_classes = len(chars_ids)

x = tf.placeholder(tf.int32,[None,num_steps])
embedings = tf.get_variable("embedding_matrix",[num_classes,state_size])
rnn_inputs = tf.nn.embedding_lookup(embedings,x)
x_one_hot = tf.one_hot(x,len(chars_ids), dtype=tf.float32)
y = tf.placeholder(tf.int32,shape = [None] )
y_onehot = tf.one_hot(y,len(chars_ids),dtype=tf.float32)
#rnn_inputs = tf.unpack(x_one_hot,axis=1) # This aint required for dynamic rnn

cell = tf.nn.rnn_cell.LSTMCell(num_units=state_size)
cell = tf.nn.rnn_cell.MultiRNNCell([cell]*3)


initial_state = cell.zero_state(batch_size,dtype=tf.float32)

val,state = tf.nn.dynamic_rnn(cell,x_one_hot, dtype=tf.float32,initial_state=initial_state)



val = tf.transpose(val,[1,0,2])
last = tf.gather(val,tf.shape(val)[0]-1)
W = tf.Variable(tf.truncated_normal(shape=[state_size,num_classes]))
B = tf.Variable(tf.truncated_normal(shape=[num_classes]))
output = tf.nn.relu(tf.matmul(last,W) + B)
logits = tf.nn.softmax(output)
losses = tf.nn.softmax_cross_entropy_with_logits(output,y_onehot)
mean_loss = tf.reduce_mean(losses)
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(mean_loss)
correct_pred = tf.equal(tf.arg_max(logits,1), tf.arg_max(y_onehot,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
saver = tf.train.Saver()

def next_bath():
    i = np.random.randint(0,len(X)-batch_size)
    return X[i:i+batch_size],Y[i:i+batch_size]

num_epochs = 5


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print sess.run(tf.shape(val),feed_dict={x:X[0:200],y:Y[0:200]})
    print cell.output_size
    print cell.state_size
    print sess.run(tf.shape(x_one_hot),feed_dict={x:X[0:200],y:Y[0:200]})
    for i in range(num_epochs):
        for i in range(1114990):
            _, valshape = sess.run([train_step, tf.shape(val)], feed_dict={x: X[i:i + 200], y: Y[i:i + 200]})

            # print valshape
            acc = sess.run(accuracy, feed_dict={x: X[i:i + 200], y: Y[i:i + 200]})
            print i, " :", acc
        save_path = saver.save(sess, "model.ckpt")
        print save_path
    saver.restore(sess,"model.ckpt")
    print X[0:200]
    print "Now the Network\n"
    for i in range(200):
        Y = np.array(X[0:200])
        next = sess.run(logits,feed_dict={x:Y})
        next_id = sess.run(tf.arg_max(next,1))
        print ids_chars[next_id[199]]
        #X[199,199]= next_id[0]
        for j in range(200):
            if j+1 < len(X[199]):
                X[199,j] = X[199,j+1]
            X[199,199] = next_id[199]
























