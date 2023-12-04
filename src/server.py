from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import paho.mqtt.client as mqtt
import json
import pandas as pd
from datetime import datetime
import asyncio

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Fall detection</title>
         <style>
            /* Add some CSS for styling the scrollable area */
            #messages {
                overflow-y: scroll;
                height: 90vh;  /* Limit the height of the scrollable area */
                width: 60vw;  /* Limit the height of the scrollable area */
            }
        </style>
    </head>
    <body>
        <h1>Real-time data collection</h1>
        <button id="closeButton">Stängd</button>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages');
                var message = document.createElement('li');
                var content = document.createTextNode(event.data);
                message.appendChild(content);
       
                 // Parse the JSON data
                var data = JSON.parse(event.data);

                // Check if gyroscope_z is less than -22
                if (data.acceleration_x > 0.1) {
                    message.style.color = 'red'; // Set the text color to red
                }
                messages.appendChild(message);
                messages.scrollTop = messages.scrollHeight;
            };
                 close = function(){
                    ws.send("close");
            }
             var closeButton = document.getElementById('closeButton');
            closeButton.addEventListener('click', function() {
                // Call the close function when the button is clicked
                close();
            });
       
        </script>
    </body>
</html>
"""

class MQTTHandler:
    def __init__(self, broker_address, mqtt_topic):
        self.broker_address = broker_address
        self.mqtt_topic = mqtt_topic
        self.mqtt_data = None
        self.columns = ["timestamp", "acceleration_x", "acceleration_y", "acceleration_z", "gyroscope_x", "gyroscope_y", "gyroscope_z"]
        self.df = pd.DataFrame(columns=self.columns)

        self.mqtt_client = mqtt.Client("FastAPIMQTTListener1", clean_session=True)
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(self.broker_address)
        self.mqtt_client.subscribe(self.mqtt_topic)
        self.mqtt_client.loop_start()
    
    def unsubscribe_mqtt(self):
        self.mqtt_client.unsubscribe(self.mqtt_topic)
        self.mqtt_client.disconnect()

    def on_message(self, client, userdata, message):
        self.mqtt_data = message.payload.decode()    
        mqtt_data = json.loads(self.mqtt_data)
        print(mqtt_data)
        # Add timestamp to the data
        mqtt_data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_row = pd.DataFrame({
            "timestamp": [mqtt_data["timestamp"]],
            "acceleration_x": [mqtt_data["acceleration_x"]],
            "acceleration_y": [mqtt_data["acceleration_y"]],
            "acceleration_z": [mqtt_data["acceleration_z"]],
            "gyroscope_x": [mqtt_data["gyroscope_x"]],
            "gyroscope_y": [mqtt_data["gyroscope_y"]],
            "gyroscope_z": [mqtt_data["gyroscope_z"]]
        })

        # Append the new row to the existing DataFrame
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        # Save the DataFrame to a CSV file periodically
        if len(self.df["timestamp"]) % 100 == 0:  # Save every 1000 entries (adjust as needed)
            self.df['timestamp'] = self.df['timestamp'].astype(str)
            self.save_to_csv()


    def save_to_csv(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = f"fall_data_{timestamp}.csv"
        self.df.to_csv(file_path, index=False, sep=';')
        print(f"DataFrame saved to {file_path}")

class CloseEvent:
    def __init__(self, mqtt_handler):
        self.mqtt_handler = mqtt_handler

    async def __call__(self, websocket, _):
        self.mqtt_handler.unsubscribe_mqtt()

mqtt_handler = MQTTHandler("broker.hivemq.com", "ddwddwdd")
# mqtt_handler = MQTTHandler("broker.emqx.io", "ddwddwdd")

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        mqtt_data = mqtt_handler.mqtt_data
        if mqtt_data is not None:
            print(mqtt_data)
            # if mqtt_data.get("acceleration_x", 0) > 0.1:
            #     mqtt_data = "ALERT!"
            
            await websocket.send_text(json.dumps(mqtt_data))
        # close_event = await websocket.receive_text()
        # if close_event == "close":
        #     print("CLose")
        #     await websocket.close()
        #     break
        # await asyncio.sleep(0.002)
    

      

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
