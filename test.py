#!/usr/bin/env python
# coding: utf-8

# # Definicion del numero de capas y neuronas 

# In[ ]:


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 - Multi layer perceptron, MLP.
 - Simple artificial neural networks with 2 hidden units and 1 output unit to
   learn XOR gate.

@author: nasir
"""

#%%

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


num_features = 2
num_iter = 40000
display_step = int(num_iter / 20)
learning_rate = 0.01

num_input = 2          # units in the input layer 28x28 images
num_hidden1 = 2        # units in the first hidden layer
num_output = 1         # units in the output, only one output 0 or 1


# # Training sample y modelo 

# In[ ]:


#%% mlp function

def multi_layer_perceptron_xor(x, weights, biases):

    hidden_layer1 = tf.add(tf.matmul(x, weights['w_h1']), biases['b_h1'])
    hidden_layer1 = tf.nn.sigmoid(hidden_layer1)

    out_layer = tf.add(tf.matmul(hidden_layer1, weights['w_out']), biases['b_out'])

    return out_layer

#%%
x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float32)  # 4x2, input
y = np.array([0, 1, 1, 0], np.float32)                      # 4, correct output, AND operation
y = np.reshape(y, [4,1])                                    # convert to 4x1

# trainum_inputg data and labels
X = tf.placeholder('float', [None, num_input])     # training data
Y = tf.placeholder('float', [None, num_output])    # labels

# weights and biases
weights = {
    'w_h1' : tf.Variable(tf.random_normal([num_input, num_hidden1])), # w1, from input layer to hidden layer 1
    'w_out': tf.Variable(tf.random_normal([num_hidden1, num_output])) # w2, from hidden layer 1 to output layer
}
biases = {
    'b_h1' : tf.Variable(tf.zeros([num_hidden1])),
    'b_out': tf.Variable(tf.zeros([num_output]))
}

model = multi_layer_perceptron_xor(X, weights, biases)


# # Definicion de la funcion de costo 

# In[ ]:


'''
- cost function and optimization
- sigmoid cross entropy -- single output
- softmax cross entropy -- multiple output, normalized
'''

# = - x * z + log(1 + exp(x))   donde   logits=x  y labels = z  
loss_func = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_func)


# # Tensorboard

# In[ ]:


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

cost_summary = tf.summary.scalar("cost",loss_func)
# Merge all summaries
all_summary = tf.summary.merge_all()
# Summary writer
writer = tf.summary.FileWriter("C:/Users/Alfredo/Downloads/ML2", sess.graph)


# # Proceso de aprendizaje y graficas

# In[ ]:



for k in range(num_iter):
    tmp_cost, _ = sess.run([loss_func, optimizer], feed_dict={X: x, Y: y})
    if k % display_step == 0:
        #print('output: ', sess.run(model, feed_dict={X:x}))
        print('loss= ' + "{:.5f}".format(tmp_cost))
        # Generate accuracy stats on test data 
        summary_results = sess.run(all_summary,feed_dict={X: x, Y: y})
        writer.add_summary(summary_results,k)
        # separates the input space  
        W = np.squeeze(sess.run(weights['w_h1']))   # 2x2
        b = np.squeeze(sess.run(biases['b_h1']))    # 2,
        #%%
        print('w[0,0]',W[0,0],'W[1,0]',W[1,0])
        print('W[0,1]',W[0,1],'W[1,1]',W[1,1])
        print('b[0]',b[0],'b[1]',b[1])
        # Now plot the fitted line. We need only two points to plot the line
        plot_x = np.array([np.min(x[:, 0] - 0.2), np.max(x[:, 1]+0.2)])
        plot_y =  -1 / W[1, 0] * (W[0, 0] * plot_x + b[0])
        plot_y = np.reshape(plot_y, [2, -1])
        plot_y = np.squeeze(plot_y)

        plot_y2 = -1 / W[1, 1] * (W[0, 1] * plot_x + b[1])
        plot_y2 = np.reshape(plot_y2, [2, -1])
        plot_y2 = np.squeeze(plot_y2)

        ytemp = np.array([0, 1, 1, 0], np.float32)  
        #plt.scatter(x[:, 0], x[:, 1], c=y, s=100, cmap='viridis')
        plt.scatter(x[:, 0], x[:, 1], c=ytemp, s=100, cmap='viridis')
        plt.plot(plot_x, plot_y, color='k', linewidth=2)    # line 1
        plt.plot(plot_x, plot_y2, color='k', linewidth=2)   # line 2
        plt.xlim([-0.2, 1.2]); plt.ylim([-0.2, 1.25]);
        plt.text(0.425, 1.05, 'XOR', fontsize=14)
        plt.xticks([0.0, 0.5, 1.0]); plt.yticks([0.0, 0.5, 1.0])
        plt.show()

#%%

sess.close()


# In[ ]:




