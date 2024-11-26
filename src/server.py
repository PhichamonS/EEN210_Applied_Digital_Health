import os
import json
from datetime import datetime
import time
import pandas as pd
import uvicorn
import numpy as np
import pickle
import joblib
import asyncio

from fastapi import FastAPI, WebSocket, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import WebSocketDisconnect
from starlette.middleware.cors import CORSMiddleware

from fhirclient import client
import fhirclient.models.patient as p

app = FastAPI()
# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FHIR server URL
url = "https://fhirsandbox.healthit.gov/open/r4/fhir/Patient?_format=json"

# Load HTML for the root endpoint
with open("./src/app.html", "r") as f:
    html = f.read()

# DataProcessor class and methods remain unchanged
class DataProcessor:
    def __init__(self):
        self.data_buffer = []

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = f"fall_front_data_{timestamp}.csv"

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


data_collection = DataProcessor()
data_processor = DataProcessor()



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



def fhir_resource_w19(patientId, fall_datetime, record_datetime):
    """
    Generates a FHIR Condition resource for a patient with ICD-10 code W19 (Unspecified fall).
    
    Parameters:
    - patientId (str): The unique identifier of the patient.
    
    Returns:
    - dict: A FHIR Condition resource for the specified patient ID and ICD-10 code W19.
    """
    condition_resource = {
        "resourceType": "Condition",
        "id": "example-W19",
        "clinicalStatus": {
            "coding": [
                {"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active", "display": "Active"}
            ]
        },
        "verificationStatus": {
            "coding": [
                {"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed", "display": "Confirmed"}
            ]
        },
        "category": [
            {
                "coding": [
                    {"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "encounter-diagnosis", "display": "Encounter Diagnosis"}
                ]
            }
        ],
        "severity": {
            "coding": [
                {"system": "http://snomed.info/sct", "code": "255604002", "display": "Mild"}
            ]
        },
        "code": {
            "coding": [
                {"system": "http://hl7.org/fhir/sid/icd-10", "code": "W19", "display": "Unspecified fall"}
            ]
        },
        "subject": {
            "reference": f"Patient:{patientId}",
        },
        "encounter": {
            "reference": "Encounter/example"
        },
        "onsetDateTime": fall_datetime,
        "recordedDate": record_datetime,
        "recorder": {
            "reference": "Practitioner",
            "display": "Dr. Margueritte"
        },
        "asserter": {
            "reference": "Practitioner",
            "display": "Dr. Margueritte"
        }
    }

    return condition_resource



label = ""
content = ""
persist_patient_id = ""


async def wait_for_patient_id():
    while persist_patient_id == "":  # Check if the patient ID is not set
        await asyncio.sleep(1)  # Wait for 1 second before checking again


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):

    # Wait for connection
    await websocket_manager.connect(websocket)

    # Wait for user input
    await wait_for_patient_id()
    await websocket_manager.broadcast_message(json.dumps({
                                                'action':'updateDiv',
                                                'content': 'Awaiting fall detection'}))
                                                
    try:
        count_false = 0
        count_window = 0
        fall_trigger = 0
        fall_thres = 20
        fall_datetime = ""

        while True:            

            data = await websocket.receive_text()

            # Broadcast the incoming data to all connected clients
            json_data = json.loads(data)

            # use raw_data for prediction
            raw_data = list(json_data.values())

            data_processor.add_data(raw_data)  

            # Add time stamp to the last received data
            json_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")            
            
            # This line save the recent 100 samples to the CSV file. you can change 100 if you want.
            # data_collection.add_data(json_data)
            # if len(data_collection.data_buffer) >= 100:
            #     data_collection.save_to_csv()

            if len(data_processor.data_buffer) >= 30: # Collecting 30 data points for processing
                
                # convert data to dataframe
                df = pd.DataFrame(data_processor.data_buffer, 
                                  columns=['acceleration_x', 'acceleration_y', 'acceleration_z', 'gyroscope_x','gyroscope_y', 'gyroscope_z'])
                
                # Process data and calculate feature
                df_feat = calFeat(df,11,21)

                # Fall prediction
                global label
                label = predict_label(sc, model, df_feat)

                datetime_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(datetime_str, ':', label)

                await websocket_manager.broadcast_message(json.dumps({'action' : "timeUpdate", 'content': f"{datetime_str[:-4]} | {label}"}))

                # Remove first 10 data point from data buffer
                data_processor.sliding(10)                
                
                if fall_trigger == 1:
                    # Try to count laying window after fall

                    if (label == 'falling') & (count_window == 0):
                        print('trigger fall detection')
                        
                    else:

                        count_window = count_window + 1

                        if label != 'laying':
                            count_false = count_false + 1                    
                                
                        if count_false >= 4:
                            # False fall detection -> reset counter
                                count_false = 0
                                count_window = 0
                                fall_trigger = 0
                                fall_datetime =""

                        elif count_window == fall_thres:
                            # Fall detection activate
                            print('alert')
                            alert_message = json.dumps({
                                    'action':'fallAlert',
                                    'content': fhir_resource_w19(persist_patient_id,fall_datetime, datetime.now().strftime("%Y%m%d %H:%M:%S"))
                                })
                            await websocket_manager.broadcast_message(alert_message)

                            count_false = 0
                            count_window = 0
                            fall_trigger = 0
                            fall_datetime =""

                    # print('win = ', count_window ,'/',fall_thres , 'false = ', count_false)

                else:
                    if label == 'falling':
                        fall_trigger = 1
                        fall_datetime = datetime.now().strftime("%Y%m%d %H:%M:%S")
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

persit_patient_id = ''

# Endpoint for fetching patient data from a FHIR server
@app.post("/fetch-patient")
async def fetch_patient_data(patientId: str = Form(...)):
    # Configuration for FHIR server
    settings = {
        'app_id': 'my_app',
        'api_base': url  # Use the provided FHIR server URL
    }
        
    # Initialize the FHIR client
    fhir_client = client.FHIRClient(settings=settings)

    try:
        # Retrieve the patient resource
        patient = p.Patient.read(patientId, fhir_client.server)
        global persist_patient_id
        persist_patient_id = patientId

        # Extract patient's name
        if patient.name:
            full_names = [' '.join(name.given + [name.family]) for name in patient.name]
            patient_name = ', '.join(full_names)
        else:
            patient_name = "Patient's name is not available"

        # Calculate age
        if patient.birthDate:
            birth_date = datetime.strptime(patient.birthDate.isostring, '%Y-%m-%d')
            today = datetime.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            patient_age = age
        else:
            patient_age = "Patient's birth date is not available"

        patient_data = {'name': patient_name, 'age': patient_age}

    except Exception as e:
        return {'error': str(e)}

    return JSONResponse(content=patient_data)

# Ensure your code for DataProcessor, load_model, and any other necessary functionality is properly integrated above
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
