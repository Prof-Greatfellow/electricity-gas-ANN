import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import matplotlib.pyplot as plt

def get_pressure_results(node,whether_pressure=True):
    df=pd.read_csv('model_data_electricity_gas_piecewise_getpressure.csv',index_col=0).reset_index()
    No_of_sources=5
    No_of_generators=10
    No_of_buses=24
    No_of_nodes=20
    list_of_vars_input=['loadfactor'+str(i+1) for i in range(No_of_nodes)]+\
        ['load_factor_power'+str(i+1) for i in range(No_of_buses)]
    if whether_pressure:
        list_of_vars_output=['nodal_pressure'+str(node+1)]
    else:
        list_of_vars_output=['pipeline_flow'+str(node+1)]
    #df_piecewise_result=df_SOC[list_of_vars_new].rename(columns=lambda x:x+' output')
    No_of_inputs=df.shape[0]
    No_of_training_samples=int(No_of_inputs*0.8)    # The rest of the samples are set as the test set.
    #No_of_training_samples=int(No_of_inputs)
    input_length=len(list_of_vars_input)
    output_length=len(list_of_vars_output)

    train_dataset=df
    # Divide up the training and testing datasets
    #train_dataset=df.loc[(No_of_inputs-No_of_training_samples):]
    
    avg=np.average(train_dataset[list_of_vars_output])
    train_labels=train_dataset[list_of_vars_output]-avg
    train_dataset=train_dataset[list_of_vars_input]
    print('avg=',avg)
    test_dataset=df.loc[:(No_of_inputs-No_of_training_samples)]
    test_labels=test_dataset[list_of_vars_output]-avg
    test_dataset=test_dataset[list_of_vars_input]
    
    # Normalize data
    train_stats=train_dataset.describe().transpose()
    
    def normalize(x):
        return (x-train_stats['mean'])/train_stats['std']
        #return x
    normed_train_data=normalize(train_dataset)
    normed_test_data=normalize(test_dataset)

    #normed_test_data=normalize(pd.read_csv('pressure_testset_correct.csv',index_col=0).reset_index()[list_of_vars_input])
    if df.describe().loc['std'][list_of_vars_output].values[0]<0.3:
        print('\n 399 or 398:',No_of_inputs-No_of_training_samples+1)
        return np.zeros((No_of_inputs-No_of_training_samples+1,1))+avg
    # Build a model
    model=tf.keras.models.Sequential([
            tf.keras.layers.Dense(12,input_shape=[input_length]),
            tf.keras.layers.Dropout(0.15),
            tf.keras.layers.Dense(8,activation='relu'),
            tf.keras.layers.Dense(output_length)
            ])
    
    
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0005)
    Epochs=20000
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse'])
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    #normed_train_data = normed_train_data.repeat().shuffle(No_of_training_samples)
    history = model.fit(normed_train_data, train_labels, 
                        epochs=Epochs, validation_split = 0.2,verbose=0, 
                        #callbacks=[early_stop])
                        callbacks=[early_stop, tfdocs.modeling.EpochDots()])
    
    #loss,mae,mse=model.evaluate(normed_test_data,test_labels,verbose=1)
    
    hist=pd.DataFrame(history.history)
    hist['epoch']=history.epoch
    plotter=tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mae")
    plt.ylabel('MAE [MPG]')
    #print(np.mean(abs(model.predict(normed_test_data).flatten()-np.array(test_labels).flatten())))

    return model.predict(normed_test_data)+avg
statistics_pressure=pd.DataFrame()
for node in range(20):
    temp=get_pressure_results(node,whether_pressure=True)
    statistics_pressure['node '+str(node+1)]=pd.Series(temp.squeeze())
statistics_pressure.to_csv('templist_newversion20200831.csv')