#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>
#include <Adafruit_BusIO_Register.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <AsyncTCP.h>
#include <PubSubClient.h> // Include the PubSubClient library

MPU6050 mpu; // Define the sensor
const char* ssid; //definie the ssid
const char* password; //definie the password

const char* mqttServer = "broker.hivemq.com";
const int mqttPort = 1883; // Default MQTT port

WiFiClient espClient;
PubSubClient mqttClient(espClient);
bool mqttConnected = false; // Maintain MQTT connection state


// the following variables are unsigned longs because the time, measured in
// milliseconds, will quickly become a bigger number than can be stored in an int.
unsigned long lastTime = 0;
// Timer set to 10 minutes (600000)
//unsigned long timerDelay = 600000;
// Set timer to 5 seconds (5000)
unsigned long timerDelay = 5000;

void setup() {
  Serial.begin(9600);
  // initialize sensor
  Wire.begin();
  mpu.initialize(); // Initialize the MPU6050

  WiFi.begin(ssid, password);
  Serial.println("Connecting");
  while(WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());
 
  Serial.println("Timer set to 5 seconds (timerDelay variable), it will take 5 seconds before publishing the first reading.");

  // Initialize MQTT client
  mqttClient.setServer(mqttServer, mqttPort);
  mqttClient.setKeepAlive(100); // Keep-alive interval in seconds
}


void reconnectMQTT() {
  while (!mqttClient.connected()) {
    Serial.println("Attempting MQTT connection...");
    if (mqttClient.connect("ESP32Client")) {
      Serial.println("Connected to MQTT broker");
      mqttClient.subscribe("EEN210"); // Subscribe to MQTT topic(s)
      mqttConnected = true; // Set MQTT connection state to true
    } else {
      Serial.print("Failed to connect to MQTT broker, rc=");
      Serial.println(mqttClient.state());
      delay(5000);
    }
  }
}

void loop() {
  Serial.println("inside loop");

  if (WiFi.status() == WL_CONNECTED) {
    if (!mqttConnected) {
      reconnectMQTT();
    }

    char topic[] = "EEN210!"; // Replace with the MQTT topic you want to publish to

    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    // Get acceleration data
    mpu.getAcceleration(&ax, &ay, &az);
    // Get gyroscope data
    mpu.getRotation(&gx, &gy, &gz);

    // Calculate acceleration in m/s^2
    float accelX = ax / 16384.0; // 16384 LSB/g for +/- 2g range
    float accelY = ay / 16384.0;
    float accelZ = az / 16384.0;

    // Calculate gyroscope data in degrees per second
    float gyroX = gx / 131.0; // 131 LSB/deg/s for +/- 250 deg/s range
    float gyroY = gy / 131.0;
    float gyroZ = gz / 131.0;

    // Convert data to a JSON string
    String payload = "{\"acceleration_x\":" + String(accelX, 4) +
                    ",\"acceleration_y\":" + String(accelY, 4) +
                    ",\"acceleration_z\":" + String(accelZ, 4) +
                    ",\"gyroscope_x\":" + String(gyroX, 4) +
                    ",\"gyroscope_y\":" + String(gyroY, 4) +
                    ",\"gyroscope_z\":" + String(gyroZ, 4) + "}";

    // Publish the data to the MQTT topic
    if (mqttClient.publish(topic, payload.c_str())) {
      Serial.println("Message sent successfully");
    } else {
      Serial.println("Failed to send message");
      mqttConnected = false; // Set MQTT connection state to false on failure
    }
    // Handle MQTT client
    mqttClient.loop();
  } else {
    Serial.println("error wifi");
  }
  delay(100);
}
