import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

x_data = np.linspace(1, 10, 10) + np.random.uniform(-1, 1, 10)
y_data = x_data * 2 + 10 + np.random.uniform(-1, 1, 10)
plt.plot(x_data, y_data, "*")
plt.show()
m = tf.Variable(np.random.random(1)[0])
b = tf.Variable(np.random.random(1)[0])

err = 0
for x , y in zip(x_data , y_data):
    y_ = m*x + b
    err += 0.5*(y_ - y)**2

opt = tf.train.GradientDescentOptimizer(0.005)
exe = opt.minimize(err)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    
    for j in range(1000):
        sess.run(exe)
        
    m_ = m.eval()
    b_ = b.eval()
    
print("Slope: " , m_)
print("Y_Intercept: ", b_)
