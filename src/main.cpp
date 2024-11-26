#include <Wire.h>
#include <MPU6050.h>
#include <WiFi.h>
#include <WebSocketsClient.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BusIO_Register.h>

// Public internet, allow port in firewall
// Replace with your network credentials
const char *ssid = "Aless";
const char *password = "12345678";

// Replace with your WebSocket server address
const char *webSocketServer = "192.168.200.79";

const int webSocketPort = 8000;
const char *webSocketPath = "/";
MPU6050 mpu; // Define the sensor
WebSocketsClient client;
// SocketIOClient socketIO;

bool wifiConnected = false;

void setup()
{
  Serial.begin(9600);
  Wire.begin();
  mpu.initialize();

 Serial.println("Adafruit MPU6050 test!");

  Adafruit_MPU6050 mpu;
  // Try to initialize!
  if (!mpu.begin()) {
    Serial.println("Failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("Accelerometer range set to: ");
  switch (mpu.getAccelerometerRange()) {
  case MPU6050_RANGE_2_G:
    Serial.println("+-2G");
    break;
  case MPU6050_RANGE_4_G:
    Serial.println("+-4G");
    break;
  case MPU6050_RANGE_8_G:
    Serial.println("+-8G");
    break;
  case MPU6050_RANGE_16_G:
    Serial.println("+-16G");
    break;
  }
  mpu.setGyroRange(MPU6050_RANGE_2000_DEG);
  Serial.print("Gyro range set to: ");
  switch (mpu.getGyroRange()) {
  case MPU6050_RANGE_250_DEG:
    Serial.println("+- 250 deg/s");
    break;
  case MPU6050_RANGE_500_DEG:
    Serial.println("+- 500 deg/s");
    break;
  case MPU6050_RANGE_1000_DEG:
    Serial.println("+- 1000 deg/s");
    break;
  case MPU6050_RANGE_2000_DEG:
    Serial.println("+- 2000 deg/s");
    break;
  }

  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);
  Serial.print("Filter bandwidth set to: ");
  switch (mpu.getFilterBandwidth()) {
  case MPU6050_BAND_260_HZ:
    Serial.println("260 Hz");
    break;
  case MPU6050_BAND_184_HZ:
    Serial.println("184 Hz");
    break;
  case MPU6050_BAND_94_HZ:
    Serial.println("94 Hz");
    break;
  case MPU6050_BAND_44_HZ:
    Serial.println("44 Hz");
    break;
  case MPU6050_BAND_21_HZ:
    Serial.println("21 Hz");
    break;
  case MPU6050_BAND_10_HZ:
    Serial.println("10 Hz");
    break;
  case MPU6050_BAND_5_HZ:
    Serial.println("5 Hz");
    break;
  }

  Serial.println("");
  delay(100);

  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());
  client.begin(webSocketServer, webSocketPort, "/ws");
  Serial.println(client.isConnected());
  wifiConnected = true;
}

void loop()
{
  if (wifiConnected)
  {
    client.loop();

    // Get sensor data
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    mpu.getAcceleration(&ax, &ay, &az);
    mpu.getRotation(&gx, &gy, &gz);

    // Convert data to a JSON string
    String payload = "{\"acceleration_x\":" + String(ax) +
                     ",\"acceleration_y\":" + String(ay) +
                     ",\"acceleration_z\":" + String(az) +
                     ",\"gyroscope_x\":" + String(gx) +
                     ",\"gyroscope_y\":" + String(gy) +
                     ",\"gyroscope_z\":" + String(gz) + 
                     "}";

    Serial.println("Skiikar....");
    // server address, port and URL
    // Send data via WebSocket
    client.sendTXT(payload);
    client.loop();

    delay(50); // Adjust delay as needed
  }
}
