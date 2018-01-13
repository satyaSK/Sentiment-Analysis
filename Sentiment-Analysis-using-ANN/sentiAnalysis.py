import tensorflow as tf 
import numpy as np 
from tqdm import tqdm
from sentimentData import get_data

# Get da data
positive_file_path = 'Data/positive.txt'
negative_file_path = 'Data/negative.txt'
x_train, y_train, x_test, y_test = get_data(positive_file_path, negative_file_path,test_size=0.1)


#Define Hyperparameters
learning_rate = 0.001
input_dim = len(x_train[0])
hidden_1= 1000
hidden_2= 1000
hidden_3= 500
classes = 2
batch_size = 100
epochs = 12


#Defining our placeholders
X = tf.placeholder(tf.float32,[None,input_dim],name='X')
Y = tf.placeholder(tf.float32,[None, classes],name='Y')


Weights = {'w1':tf.Variable(tf.truncated_normal([input_dim, hidden_1])),
           'w2':tf.Variable(tf.truncated_normal([hidden_1, hidden_2])),
           'w3':tf.Variable(tf.truncated_normal([hidden_2, hidden_3])),
           'w4':tf.Variable(tf.truncated_normal([hidden_3, classes]))}
 
Biases = {'b1': tf.Variable(tf.random_normal([hidden_1])),
          'b2': tf.Variable(tf.random_normal([hidden_2])),
          'b3': tf.Variable(tf.random_normal([hidden_3])),
          'b4': tf.Variable(tf.random_normal([classes]))}

#Define the model
def ANN(data):
    hidden_layer_1 = tf.nn.relu(tf.matmul(X, Weights['w1']) + Biases['b1'])
    hidden_layer_2 = tf.nn.relu(tf.matmul(hidden_layer_1, Weights['w2']) + Biases['b2'])
    hidden_layer_3 = tf.nn.relu(tf.matmul(hidden_layer_2, Weights['w3']) + Biases['b3'])
    output_layer_logits = tf.matmul(hidden_layer_3, Weights['w4']) + Biases['b4']
    return output_layer_logits

logits = ANN(X)
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y)
loss = tf.reduce_mean(entropy, name='loss')
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        total_error = 0
        i = 0
        while i < len(x_train):
            start = i 
            end = i + batch_size
            X_batch, Y_batch = np.array(x_train[start:end]), np.array(y_train[start:end])
            _,e = sess.run([optimizer, loss], feed_dict={X:X_batch,Y:Y_batch})
            i += batch_size
            

            total_error += e
        print("Epoch {0} - Error {1}".format(epoch+1,total_error))

    #Testing
    predictions = tf.nn.softmax(ANN(X))    
    on_point = tf.equal(tf.argmax(predictions,axis=1),tf.argmax(Y,axis=1))
    accuracy = tf.reduce_sum(tf.cast(on_point, tf.float32))
    
    overall_correct_preds = 0.0
    j = 0
    num_batches = int(len(x_test)/batch_size)
    print(num_batches)
    while j < len(x_test):
        start = j
        end = j + batch_size
        X_batch, Y_batch = np.array(x_test[start:end]), np.array(y_test[start:end])
        a = sess.run(accuracy, feed_dict={X:X_batch, Y:Y_batch})         
        overall_correct_preds += a
        j += batch_size
        print("Test Batch Accuracy ",a/batch_size)
    print("total accuracy: {0}".format(overall_correct_preds/(num_batches*batch_size)))