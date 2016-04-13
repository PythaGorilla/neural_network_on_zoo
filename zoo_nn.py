from mlp import MLP
import numpy as np
import pandas as pd

sample_size=101
data=pd.read_csv("zoo.data",header=None)
data=data.as_matrix(columns=None)
animal_names=data[0]
input=data[:sample_size,1:-1]
output=data[:sample_size,-1:]
samples = np.zeros(sample_size, dtype=[('input',  float, 16), ('output', float, 1)])
network = MLP(16,16,1)
#to do

for i in xrange(sample_size):
    a=input[i][-4]
    input[i][-4]=a/8.0
    output[i]=output[i]/7.0
    samples[i]=input[i],output[i]
# print samples
# print data

def learn(network,samples, epochs=2500, lrate=.1, momentum=0.1):
        # Train
        for i in range(epochs):
            n = np.random.randint(samples.size)
            network.propagate_forward( samples['input'][n] )
            network.propagate_backward( samples['output'][n], lrate, momentum )
        # Test
        for i in range(samples.size):
            o = network.propagate_forward( samples['input'][i] )
            print i, samples['input'][i], '%.2f' % (o[0]*7.0),
            print '(expected %.2f)' % (samples['output'][i]*7.0)
        print
learn(network,samples)