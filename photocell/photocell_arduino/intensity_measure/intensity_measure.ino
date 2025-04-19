// This program make arduino uno return the light intensity data and the timestamp in milliseconds

#include <TimerOne.h>
#include <Math.h>

#define PHOTOTRANS_PIN A0

// GLOBAL INITIALIZATION ===================================================
// Timer Interrupt variables ----------
int read_freq = 500; 
int current_val;
unsigned long time;
char buf[32];

// MAIN ====================================================================
void setup() {
  // Serial initialization for monitoring
  Serial.begin(115200);

  // Timer interrupt initialization
  Timer1.initialize(round(1000000.0/read_freq));
  Timer1.attachInterrupt(timerIsr);
}

void loop() {
}

// TIMER INTERRUPT =========================================================
void timerIsr() {
    // Read data from the light sensor
    current_val = analogRead(PHOTOTRANS_PIN);
    // time = millis();
    time = micros();
    ltoa(time, buf, 10);

    // Serial.println("timestamp :\t" + buf + " \tData :\t" + String(current_val));
    Serial.print(buf);
    Serial.println(','+String(current_val));
}