import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import pandas as pd
import matplotlib.pyplot as plt

def compute_values(df,test_dataset,gen,time):
    time_scale=24
    No_of_inputs=df.shape[0]
    No_of_training_samples=int(No_of_inputs*1)    # The rest of the samples are set as the test set.
    input_list=['powerfactor'+str(i+1) for i in range(time_scale)]+['gasfactor'+str(i+1) for i in range(time_scale)]
    output_list=['power_generation'+str(gen+1)+'_'+str(time+1)]
    output_length=len(output_list)
    df=df[input_list+output_list]

# Divide up the training and testing datasets
    train_dataset=df.loc[(No_of_inputs-No_of_training_samples):]
    train_labels=train_dataset[output_list]
    train_dataset=train_dataset[input_list]
    
    train_stats=train_dataset.describe().transpose()
    #test_dataset=df.loc[:(No_of_inputs-No_of_training_samples)]
    #test_labels=test_dataset[output_list]
    #test_dataset=test_dataset[input_list]
    
    # Normalize data
    def normalize(x):
        return (x-train_stats['mean'])/train_stats['std']
    normed_train_data=normalize(train_dataset)
    normed_test_data=normalize(test_dataset)
    
    model=tf.keras.models.Sequential([
            tf.keras.layers.Dense(15,input_shape=[len(input_list)]),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(10,activation='relu'),
            tf.keras.layers.Dense(output_length)
        ])
    
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0002)
    Epochs=20000
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse'])
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    
    history = model.fit(normed_train_data, train_labels, 
                        epochs=Epochs, validation_split = 0.2, verbose=0, 
                        callbacks=[early_stop, tfdocs.modeling.EpochDots()])
    
    hist=pd.DataFrame(history.history)
    hist['epoch']=history.epoch
    plotter=tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mae")
    plt.ylabel('MAE [MPG]')
    return model.predict(normed_test_data)
    #power_prediction=model.predict(normed_test_data)



Number_of_generators=8
Number_of_timeslots=24
df=pd.read_csv('training_set.csv',index_col=0).reset_index()
test_dataset=pd.read_csv('test_set.csv',index_col=0).reset_index()
final_statistics=test_dataset.copy()
for gen in range(Number_of_generators):
    for tt in range(Number_of_timeslots):
        temp=compute_values(df,test_dataset,gen,tt)
        final_statistics['power_output_'+str(tt+1)+'_'+str(tt+1)]=pd.Series(temp.squeeze())