import RPi.GPIO as GPIO
import time

# Initialise
BUTTON_PIN = 26
GPIO.setmode(GPIO.BCM)

def button_callback(channel):
    print("Button pressed")

GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)

GPIO.add_event_detect(
    BUTTON_PIN, GPIO.RISING, 
    callback = button_callback, bouncetime = 50
)

try:
    while True:
        time.sleep(0.01)
except KeyboardInterrupt:
    GPIO.cleanup()
