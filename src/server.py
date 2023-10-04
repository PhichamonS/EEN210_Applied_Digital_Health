from fastapi import FastAPI, Request
import paho.mqtt.client as mqtt
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")  # Create a "templates" folder in your project directory


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

mqtt_handler = MQTTHandler("broker.hivemq.com", "EEN210")

@app.get("/")
def read_root():
    return {"message": "FastAPI is running"}

# Without websockets, you would need to refresh the page each time
@app.get("/mqtt_data")
async def get_mqtt_data(request: Request):
    mqtt_data = mqtt_handler.mqtt_data
    if mqtt_data is not None:
        return templates.TemplateResponse("mqtt_data.html", {"request": request, "mqtt_data": mqtt_data})
    else:
        return templates.TemplateResponse("mqtt_data.html", {"request": request, "mqtt_data": "No MQTT data received yet"})

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
