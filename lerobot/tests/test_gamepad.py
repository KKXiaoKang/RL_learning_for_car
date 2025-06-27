#!/usr/bin/env python
import pygame
import time

def test_gamepad():
    pygame.init()
    pygame.joystick.init()
    
    if pygame.joystick.get_count() == 0:
        print("No gamepad detected!")
        return
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    
    print(f"Connected to: {joystick.get_name()}")
    print(f"Number of buttons: {joystick.get_numbuttons()}")
    print(f"Number of axes: {joystick.get_numaxes()}")
    
    print("\n🎮 Press buttons and move joysticks to see their values:")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"Button {event.button} pressed")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"Button {event.button} released")
                elif event.type == pygame.JOYAXISMOTION:
                    if abs(event.value) > 0.1:  # 只在有显著移动时打印
                        print(f"Axis {event.axis}: {event.value:.2f}")
            
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting...")
    
    pygame.quit()

if __name__ == "__main__":
    test_gamepad()
