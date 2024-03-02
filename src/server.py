import os
import json
from datetime import datetime
import time
import pandas as pd
import uvicorn
import numpy as np
import pickle
import joblib

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

app = FastAPI()
# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("./src/index.html", "r") as f:
    html = f.read()


class DataProcessor:
    def __init__(self):
        self.data_buffer = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = f"fall_front__mew_data_{timestamp}.csv"

    def add_data(self, data):
        self.data_buffer.append(data)

    def save_to_csv(self):
        df = pd.DataFrame.from_dict(self.data_buffer)
        self.data_buffer = []
        # Append the new row to the existing DataFrame
        df.to_csv(
            self.file_path,
            index=False,
            mode="a",
            header=not os.path.exists(self.file_path),
        )
        # print(f"DataFrame saved to {self.file_path}")

    def sliding(self, nwin):
        del self.data_buffer[:nwin]


data_processor = DataProcessor()
data_processor_2 = DataProcessor()


def load_model():
    # load model from pickle file
    with open('./model/RandomForest_3/scaler.pkl','rb') as sc_file:
        sc = pickle.load(sc_file)

    with open('./model/RandomForest_3/RF_classifier_model.pkl', 'rb') as model_file:  
        model = pickle.load(model_file)

    return sc, model

def rm_baseline(x, nwin):    
    N = len(x)
    idx = int((nwin-1)/2)
    x_norm = np.zeros(N)*np.nan
    
    # Remove baseline with mean
    for i in np.arange(N-nwin):
        x_win = x[i:(i+nwin)]
        x_mean = np.mean(x_win)
        x_norm[i+idx] = x[i+idx]-x_mean

    return x_norm


def apply_LR_acceleration_x(feat):
    
    with open('./model/LR_acceleration_x/LR_scaler.pkl','rb') as sc_file:
        LR_sc = pickle.load(sc_file)

    with open('./model/LR_acceleration_x/LR_classifier_model.pkl', 'rb') as model_file:  
        LR_classifier = pickle.load(model_file)

    feat2 = np.array(feat['acceleration_x'].fillna(0))
    feat2 = np.lib.stride_tricks.sliding_window_view(feat2, 21)
    feat2 =  LR_sc.transform(feat2)
    x_prob = LR_classifier.predict_proba(feat2)[:,1]

    x_prob_final = np.ones(feat.shape[0])*np.nan
    x_prob_final[10:-10] = x_prob

    return x_prob_final

def calFeat(df, nwin, nwin_smoothing):

    # Smoothing signal
    feat_1 = df.rolling(window=nwin_smoothing, center = True)['acceleration_x', 'acceleration_y', 'acceleration_z'].agg(["median"])
    feat_1.columns = [f"{col[0]}_{col[1]}" for col in feat_1.columns] 

    # # Remove baseline signal    
    dataset_rm = df.apply(lambda x: rm_baseline(x.values, nwin))

    # # Calculate magnitude using Euclidean norm
    dataset_rm['acc_magnitude'] = np.sqrt(dataset_rm['acceleration_x']**2 +
                                    dataset_rm['acceleration_y']**2 +
                                    dataset_rm['acceleration_z']**2)

    dataset_rm['gyr_magnitude'] = np.sqrt(dataset_rm['gyroscope_x']**2 +
                                    dataset_rm['gyroscope_y']**2 +
                                    dataset_rm['gyroscope_z']**2)
    
    # # Calculate time domain features -> do agg as long as data point >= nwin/2
    feat_2 = dataset_rm.rolling(window=nwin, min_periods=int(nwin/2), center = True).agg(["sum", "mean", "std", "max","min"])
    feat_2.columns = [f"{col[0]}_{col[1]}" for col in feat_2.columns]
    
    feat_2.loc[:,'acceleration_x_LR'] = apply_LR_acceleration_x(df)

    
    return pd.concat([df,feat_1,feat_2], axis=1)


def predict_label(sc=None, model=None, data=None):
    # you should modify this to return the label
    if model is not None:

        # selected_feature = ['acceleration_x_median', 'acceleration_y_median', 'acceleration_z_median', 'acceleration_x_std', 'acceleration_x_max', 'acceleration_y_std', 'acceleration_y_max', 'acceleration_z_mean', 'acceleration_z_std', 'acceleration_z_min', 'acceleration_z_max', 'gyroscope_x_std', 'gyroscope_y_std', 'gyroscope_y_min', 'gyroscope_y_max', 'gyroscope_z_std', 'gyroscope_z_min', 'gyroscope_z_max', 'acc_magnitude_sum', 'acc_magnitude_mean', 'acc_magnitude_std', 'acc_magnitude_min', 'acc_magnitude_max', 'gyr_magnitude_std', 'gyr_magnitude_min', 'gyr_magnitude_max']

        selected_feature = ['acceleration_x_median', 'acceleration_x_LR', 'acceleration_x_std',
       'acceleration_z_median', 'acceleration_z_min', 'gyr_magnitude_std',
       'gyroscope_z_std', 'acceleration_y', 'acceleration_y_median',
       'acc_magnitude_min', 'acceleration_y_std', 'acceleration_x_sum']
        
        

        feat = data[selected_feature].dropna()
        
        feat_scale = sc.transform(feat)
        y = model.predict(feat_scale)
        y = list(y)
        label = max(set(y), key=y.count)

        return label       
    else:
        return 0


class WebSocketManager:
    def __init__(self):
        self.active_connections = set()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        print("WebSocket connected")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print("WebSocket disconnected")

    async def broadcast_message(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                # Handle disconnect if needed
                self.disconnect(connection)


websocket_manager = WebSocketManager()
(sc, model) = load_model()



@app.get("/")
async def get():
    return HTMLResponse(html)

label = ''

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket_manager.connect(websocket)
    try:
        count_false = 0
        count_window = 0
        fall_trigger = 0
        fall_thres = 20

        while True:
            data = await websocket.receive_text()
            # print("hi")

            # Broadcast the incoming data to all connected clients
            json_data = json.loads(data)

            # use raw_data for prediction
            raw_data = list(json_data.values())

            data_processor_2.add_data(raw_data)  

            # Add time stamp to the last received data
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            data_processor.add_data(json_data)
            
            # this line save the recent 100 samples to the CSV file. you can change 100 if you want.
            if len(data_processor.data_buffer) >= 100:
                data_processor.save_to_csv()


            if len(data_processor_2.data_buffer) >= 30: # Collecting 30 data points for processing
                
                # convert data to dataframe
                df = pd.DataFrame(data_processor_2.data_buffer, columns=['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyroscope_x','gyroscope_y', 'gyroscope_z'])
                
                # Process data and calculate feature
                df_feat = calFeat(df,11,21)

                # Fall prediction
                global label
                label = predict_label(sc, model, df_feat)

                # Remove first 10 data point from data buffer
                data_processor_2.sliding(10)

                print(datetime.now().strftime("%H:%M:%S.%f"))
                print(label)

                
                if fall_trigger == 1:
                    # Try to count laying window after fall

                    if (label == 'falling') & (count_window == 0):
                        print('trigger fall detection')
                        
                    else:

                        count_window = count_window + 1

                        if label != 'laying':
                            count_false = count_false + 1                    
                                
                        if count_false >= 2:
                            # False fall detection -> reset counter
                                count_false = 0
                                count_window = 0
                                fall_trigger = 0

                        elif count_window == fall_thres:
                            # Fall detection activate
                            print('!!! ALERT !!!')
                            count_false = 0
                            count_window = 0

                    # print('win = ', count_window ,'/',fall_thres , 'false = ', count_false)

                else:
                    if label == 'falling':
                        fall_trigger = 1
                        print('trigger fall detection')
                
            """  
            In this line we use the model to predict the labels.
            Right now it only return 0.
            You need to modify the predict_label function to return the true label
            """
            # label = predict_label(model, raw_data)
            
            json_data["label"] = label

            # print the last data in the terminal
            # print(json_data)

            # broadcast the last data to webpage
            await websocket_manager.broadcast_message(json.dumps(json_data))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
