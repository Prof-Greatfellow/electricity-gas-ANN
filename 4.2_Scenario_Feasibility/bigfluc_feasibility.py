import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


df=pd.read_csv('archived_result/electricity_gas_cplex_largefluc_classification2.csv',index_col=0).reset_index()
No_of_nodes=20
No_of_buses=24
No_of_generators=10
No_of_inputs=df.shape[0]
No_of_training_samples=int(No_of_inputs*0.8)    # The rest of the samples are set as the test set.
input_list=['loadfactor'+str(i+1) for i in range(No_of_nodes)]+['load_factor_power'+str(i+1) for i in range(No_of_buses)]
output_list=['feasible?']
df=df[input_list+output_list]
# Divide up the training and testing datasets
train_dataset=df.loc[(No_of_inputs-No_of_training_samples):]
train_labels=train_dataset[output_list]
train_dataset=train_dataset[input_list]

train_stats=train_dataset.describe().transpose()
test_dataset=df.loc[:(No_of_inputs-No_of_training_samples)]
test_labels=test_dataset[output_list]
test_dataset=test_dataset[input_list]

# Normalize data
def normalize(x):
    return (x-train_stats['mean'])/train_stats['std']
normed_train_data=normalize(train_dataset)
normed_test_data=normalize(test_dataset)

# Build a model
model=tf.keras.models.Sequential([
        tf.keras.layers.Dense(6,input_shape=[No_of_nodes+No_of_buses]),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(6,activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(1,activation='sigmoid')
        ])


optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001)
Epochs=20000
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['binary_crossentropy'])

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history = model.fit(normed_train_data, train_labels, 
                    epochs=Epochs, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])

#history=model.fit(normed_train_data,train_labels,
#                  epochs=Epochs,validation_split=0.2,verbose=0,
#                  callbacks=[tfdocs.modeling.EpochDots()])

loss,binary_crossentropy=model.evaluate(normed_test_data,test_labels,verbose=1)

hist=pd.DataFrame(history.history)
hist['epoch']=history.epoch
plotter=tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric = "binary_crossentropy")
plt.ylabel('binary_crossentropy')
temp=np.round(model.predict(normed_test_data).flatten()-np.array(test_labels).flatten())
print((sum(abs(temp))/len(temp)))
#power_prediction=model.predict(normed_test_data)