'''Import libraries'''
import pandas as pd 
from sklearn.model_selection import train_test_split 
import keras
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
# from tensorflow.keras.models import save_model
import numpy as np
import keras.backend as K
import os
# import warnings
# warnings.filterwarnings("ignore")
# import logging
# logging.getLogger('tensorflow').disabled = True
import shutil
import sys


''' Main Function '''
def main():
    
    folder_path = 'pyModels/' 
    folder_name = folder_path + "weights/"
    driveLink = sys.argv[1]
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    
    landmark_df = pd.read_csv( folder_path + driveLink)
    
    landmark_df['class'] =  landmark_df['class'].astype('category')
    landmark_df_X = landmark_df.drop(['class'], axis = 1)
    landmark_df_Y = landmark_df[['class']]
    
    x_train, x_test, y_train, y_test = train_test_split(landmark_df_X,landmark_df_Y, test_size =0.20, random_state =17)
    
    # convert categorical to dummy variables (one hot encoded)
    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    y_train_new = to_categorical(encoded_Y)
    encoded_Y = encoder.transform(y_test)
    y_test_new = to_categorical(encoded_Y)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(36, activation="relu"))
    model.add(keras.layers.Dense(36, activation="relu"))
    model.add(keras.layers.Dense(7,activation="softmax"))
    
    # categorical_crossentropy
    model.compile(optimizer = 'adam',
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    model.fit(x_train, y_train_new, epochs = 100,validation_data=(x_test,y_test_new), verbose=0 )
    # save_model(model, folder_path + "test.h5")
    
    ''' predicting Test set '''
    # y_pred = model.predict(x_test)
    # y_pred_test = model.predict(x_test)
    # y_pred_train = model.predict(x_train)
    
    
    # model_name = '/home/vaibhav/workspace/mask_classifier/masknet_v9/2020-06-21_masknet_v9.h5'
    # model = load_model(model_name)
    # model.summary()
    
    '''Pasring the model architacture in the .npy files '''
    for index,layer in enumerate(model.layers):
    	name = (layer.__class__.__name__)
    	if name=='InputLayer' or name == 'Flatten' or name == 'permute' or name == 'SpatialDropout2D' or name == 'Reshape' or name == 'MaxPooling2D' or name == 'Activation' or name == 'Dropout':
    		continue
    	else:
    		indexed= layer.get_config()['name']
    		weight,bias=layer.get_weights()
    		# print(weight.shape)
    		if(len(weight.shape)==4):
    			weights=np.transpose(weight,(3,2,0,1))
    		if(len(weight.shape)==2):
    			weights=np.transpose(weight,(1,0))

    		
    		np.save(folder_name + indexed + "_w",weights)
    		np.save(folder_name + indexed + "_b",bias)
    K.clear_session()
    del model
    
    '''weight Zip file  '''
    shutil.make_archive("model", "zip", folder_name)
    
    print(folder_name + 'model.zip')

if __name__ == '__main__':
    main()




