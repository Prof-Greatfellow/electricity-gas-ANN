#%% import packages
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import matplotlib.pyplot as plt
import seaborn as sns

#%% load in PWL dataset
df_piecewise=pd.read_csv(generated_dataset.csv',index_col=0).reset_index()
No_of_sources=5
No_of_generators=10
list_of_vars_input=df_piecewise.columns.values[0:20+24]

def estimate_gen(gen_source_no,whether_gen):
    if whether_gen:
        list_of_vars_output=['power_generation'+str(gen_source_no+1)]
    else:
        list_of_vars_output=['source_generation'+str(gen_source_no+1)]
        
    #df_piecewise_result=df_SOC[list_of_vars_new].rename(columns=lambda x:x+' output')
    No_of_inputs=df_piecewise.shape[0]
    No_of_training_samples=int(No_of_inputs*0.8)    # The rest of the samples are set as the test set.
    df=df_piecewise
    
    input_length=len(list_of_vars_input)
    output_length=len(list_of_vars_output)
    
    # Divide up the training and testing datasets
    train_dataset=df
    #train_dataset=df.loc[(No_of_inputs-No_of_training_samples):]
    avgg=np.average(train_dataset[list_of_vars_output],axis=0)
    train_labels=train_dataset[list_of_vars_output]-avgg
    train_dataset=train_dataset[list_of_vars_input]
    
    test_dataset=pd.read_csv('testset_single_output.csv',index_col=None)
    
#    test_dataset=df.loc[:(No_of_inputs-No_of_training_samples)]
#    test_labels=test_dataset[list_of_vars_output]-avgg
#    test_dataset=test_dataset[list_of_vars_input]
    if gen_source_no==0 or gen_source_no==1 or gen_source_no==3 or gen_source_no==4 or gen_source_no==5 or gen_source_no==6 or gen_source_no==8 or gen_source_no==9:
        if whether_gen is True:
            return np.ones(3,)*avgg[0]
            #return np.mean(abs(np.array(test_labels))),avgg[0]
    # Normalize labels
    #label_stats=train_labels.describe().transpose()
    # Normalize data
    train_stats=train_dataset.describe().transpose()
    
    def normalize(x):
        return (x-train_stats['mean'])/train_stats['std']
    normed_train_data=normalize(train_dataset)
    normed_test_data=normalize(test_dataset)
    
    
    # Build a model
    model=tf.keras.models.Sequential([
            tf.keras.layers.Dense(6,input_shape=[input_length]),
            tf.keras.layers.Dense(4,activation='relu'),
            tf.keras.layers.Dense(output_length)
            ])
       
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0003)
    Epochs=20000
    model.compile(loss='mse',optimizer=optimizer,metrics=['mae','mse'])
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)
    #normed_train_data = normed_train_data.repeat().shuffle(No_of_training_samples)
    history = model.fit(normed_train_data, train_labels, 
                        epochs=Epochs, validation_split = 0.2,verbose=0, 
                        callbacks=[early_stop, tfdocs.modeling.EpochDots()])
    
    #hist=pd.DataFrame(history.history)
    #hist['epoch']=history.epoch
    #plotter=tfdocs.plots.HistoryPlotter(smoothing_std=2)
    #plotter.plot({'Basic': history}, metric = "mae")
    #plt.ylabel('MAE [MPG]')
    #print(np.mean(abs(model.predict(normed_test_data).flatten()-np.array(test_labels).flatten())))
    #return hist
    #return model.evaluate(normed_test_data(,test_labels,verbose=1)[1],avgg[0])
    return (model.predict(normed_test_data)+avgg).squeeze()

#mae_list=[]
#avgg_list=[]
#avgg_list=[i[0] for i in avgg_list]
#for gen in range(No_of_sources):
#    mae,avgg= estimate_gen(gen,False)
#    mae_list.append(mae)
#    avgg_list.append(avgg)
outputs=pd.DataFrame()

for gen in range(No_of_generators):
    estimates=estimate_gen(gen,True).squeeze()
    outputs['generator '+str(gen+1)]=pd.Series(estimates)
input("Press Any Key to continue...")


#%%Post-ANN data 
fig,ax=plt.subplots(figsize=(15,10))
df_statistics=pd.DataFrame(data=np.array([avgg_list,mae_list]).T,columns=['avg','mae']).reset_index()
df_statistics['mae']*=3
df_statistics['index']+=1
#df_statistics['mae']+=df_statistics['avg']
#df_new=pd.melt(df_statistics,id_vars=['index'],value_vars=['avg','mae'])
#sns.barplot(x='index',y='value',hue='variable',data=df_new,orient='v')
df_statistics.plot(kind = "bar",x="index", y = "avg", legend = False, yerr = "mae",ax=ax,edgecolor='black')
ax.tick_params(labelsize=24)
plt.xticks(rotation='horizontal')
ax.set_xlabel('Gas Source Index',fontsize=24)
#ax.set_xlabel('Generator Index',fontsize=24)
#ax.set_ylabel('Power Output (GW)',fontsize=24)
ax.set_ylabel('Gas Output (kcf)',fontsize=24)