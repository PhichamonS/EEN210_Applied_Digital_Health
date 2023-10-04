from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import paho.mqtt.client as mqtt
import json

app = FastAPI()

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Chat</title>
    </head>
    <body>
        <h1>Real-time data collection</h1>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8000/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""
class MQTTHandler:
    def __init__(self, broker_address, mqtt_topic):
        self.broker_address = broker_address
        self.mqtt_topic = mqtt_topic
        self.mqtt_data = None

        # Create an MQTT client and set the message callback
        self.mqtt_client = mqtt.Client("FastAPIMQTTListener")
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.connect(self.broker_address)
        self.mqtt_client.subscribe(self.mqtt_topic)

        # Start the MQTT loop
        self.mqtt_client.loop_start()

    def on_message(self, client, userdata, message):
        self.mqtt_data = message.payload.decode()

mqtt_handler = MQTTHandler("broker.hivemq.com", "EEN210!")

@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        mqtt_data = mqtt_handler.mqtt_data
        if mqtt_data is not None:
            mqtt_data = json.loads(mqtt_data)
            print(mqtt_data)
            print(type(mqtt_data))
            if mqtt_data["gyroscope_z"] < -22:
                mqtt_data = "LARM!"

            await websocket.send_text(json.dumps(mqtt_data))
        await asyncio.sleep(1)


if __name__ == "__main__":
    import uvicorn
    import asyncio

    uvicorn.run(app, host="0.0.0.0", port=8000)
