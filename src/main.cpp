#include <Arduino.h>
#include <Wire.h>
#include <MPU6050.h>
#include <Adafruit_ADXL345_U.h>
#include <Adafruit_BusIO_Register.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <HTTPClient.h>
#include <AsyncTCP.h>
#include <PubSubClient.h> // Include the PubSubClient library

// const int batteryPin = 2;  // Analog pin to measure battery voltage

// void setup() {
//     Serial.begin(9600);
// }

// void loop() {
//     // Read battery voltage
//     int rawValue = analogRead(batteryPin);
//     Serial.println(rawValue);
//     float voltage = (rawValue / 4095.0) * 3.6;  // Assuming a voltage divider is used

//     // Check if voltage is below a threshold (adjust as needed)
//     if (voltage < 3.8) {
//         Serial.println("Running on battery power!");
//         Serial.println(voltage);
//     } else {
//         Serial.println("Running on external power or USB." );
//         Serial.println(voltage);
//     }

//     delay(5000);  // Delay for 5 seconds before checking again
// }





MPU6050 mpu; // Define the sensor
const char* ssid = "iPhone (8)";
const char* password = "NYluft1993";

const char* mqttServer = "broker.hivemq.com";
// const char* mqttServer = "mqtt.eclipse.org";
const int mqttPort = 1883; // Default MQTT port
// const int mqttPort = 8883; // Default MQTT port

WiFiClient espClient;
PubSubClient mqttClient(espClient);
bool mqttConnected = false; // Maintain MQTT connection state

// the following variables are unsigned longs because the time, measured in
// milliseconds, will quickly become a bigger number than can be stored in an int.
unsigned long lastTime = 0;


void setup() {

  // 9600 bits per second, communication betwen the IMU and your computer
  Serial.begin(9600);;
  // initialize sensor
  Wire.begin();
  // Initialize the MPU6050
  mpu.initialize(); 

  WiFi.begin(ssid, password);
  Serial.println("Connecting");
  while(WiFi.status() != WL_CONNECTED) {
    // delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());
  
  // Initialize MQTT client
  mqttClient.setServer(mqttServer, mqttPort);
  mqttClient.setKeepAlive(100); // Keep-alive interval in seconds
}


void reconnectMQTT() {
  while (!mqttClient.connected()) {
    Serial.println("Attempting MQTT connection...");
    if (mqttClient.connect("ESP32Client")) {
      Serial.println("Connected to MQTT broker");
      mqttClient.subscribe("seth1993"); // Subscribe to MQTT topic(s)
      mqttConnected = true; // Set MQTT connection state to true
    } else {
      Serial.print("Failed to connect to MQTT broker, rc=");
      Serial.println(mqttClient.state());
      // delay(5000);
    }
  }
}

void loop() {
  Serial.println("inside loop");

  if (WiFi.status() == WL_CONNECTED) {
    while (!mqttConnected) {
      reconnectMQTT();
    }

    char topic[] = "ddwddwdd"; // Replace with the MQTT topic you want to publish to

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
                    
    Serial.println(payload);
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
  // delay(100);
}