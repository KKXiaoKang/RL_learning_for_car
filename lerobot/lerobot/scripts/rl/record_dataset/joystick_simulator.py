#!/usr/bin/env python3
"""
Meta VR Joystick Simulator

This tool simulates the Quest3Controller by publishing joystick data to /quest_joystick_data topic.
It provides a menu interface to control intervention and execute the Bézier trajectory tool.
"""

import os
import sys
import time
import threading
import subprocess
from typing import Optional
import signal

# ROS imports
try:
    import rospy
    from noitom_hi5_hand_udp_python.msg import JoySticks
    from std_srvs.srv import Trigger, TriggerResponse
    ROS_AVAILABLE = True
except ImportError:
    print("Warning: ROS packages not available. ROS functionality will be disabled.")
    ROS_AVAILABLE = False


class JoystickSimulator:
    """
    Simulates Quest3 VR joystick data for testing intervention functionality.
    """
    
    def __init__(self):
        """Initialize the joystick simulator."""
        self.enable_ros = ROS_AVAILABLE
        self.publisher = None
        self.service_server = None
        self.publishing = False
        self.intervention_active = False
        self.bezier_process = None
        self.running = True
        
        # Joystick data defaults
        self.left_grip = 0.0
        self.right_grip = 0.0
        self.left_first_button_pressed = False
        self.left_second_button_pressed = False
        
        # Setup ROS
        if self.enable_ros:
            self._setup_ros()
    
    def _setup_ros(self):
        """Setup ROS node and publisher."""
        try:
            # Initialize ROS node
            rospy.init_node("joystick_simulator", anonymous=True)
            
            # Create publisher for joystick data
            self.publisher = rospy.Publisher(
                '/quest_joystick_data', 
                JoySticks, 
                queue_size=10
            )
            
            # Create service server for auto record tool
            self.service_server = rospy.Service(
                '/robot_control/start_record_tool',
                Trigger,
                self.handle_start_record_tool
            )
            
            # Wait for connections
            time.sleep(1.0)
            
            print("ROS joystick simulator initialized successfully")
            print("  - Publisher: /quest_joystick_data")
            print("  - Service: /robot_control/start_record_tool")
            
        except Exception as e:
            print(f"Failed to setup ROS: {e}")
            self.enable_ros = False
    
    def create_joystick_message(self) -> 'JoySticks':
        """
        Create a JoySticks message with current state.
        
        Returns:
            JoySticks message
        """
        if not self.enable_ros:
            return None
        
        msg = JoySticks()
        
        # Set joystick axis values (default to 0.0)
        msg.left_x = 0.0
        msg.left_y = 0.0
        msg.left_trigger = 0.0
        msg.right_x = 0.0
        msg.right_y = 0.0
        msg.right_trigger = 0.0
        
        # Set grip values
        msg.left_grip = self.left_grip
        msg.right_grip = self.right_grip
        
        # Set button states
        msg.left_first_button_pressed = self.left_first_button_pressed
        msg.left_second_button_pressed = self.left_second_button_pressed
        msg.left_first_button_touched = False
        msg.left_second_button_touched = False
        msg.right_first_button_pressed = False
        msg.right_second_button_pressed = False
        msg.right_first_button_touched = False
        msg.right_second_button_touched = False
        
        return msg
    
    def handle_start_record_tool(self, req):
        """
        Handle the start record tool service request.
        
        Args:
            req: Trigger service request (empty)
            
        Returns:
            TriggerResponse: Service response with success status and message
        """
        try:
            # Log the service call
            rospy.loginfo("Received start_record_tool service request")
            
            # Start the Bezier tool in a non-blocking way
            self.start_bezier_tool()
            
            # Create response
            response = TriggerResponse()
            response.success = True
            response.message = "Bezier trajectory tool started successfully"
            
            rospy.loginfo("start_record_tool service completed successfully")
            return response
            
        except Exception as e:
            rospy.logerr(f"Error in start_record_tool service: {e}")
            
            # Create error response
            response = TriggerResponse()
            response.success = False
            response.message = f"Failed to start Bezier tool: {str(e)}"
            
            return response
    
    def publish_joystick_data(self):
        """Publish joystick data continuously."""
        if not self.enable_ros:
            print("Warning: ROS is not enabled. Cannot publish joystick data.")
            return
        
        rate = rospy.Rate(30)  # 30 Hz publishing rate
        
        print("Started publishing joystick data at 30 Hz...")
        
        try:
            while not rospy.is_shutdown() and self.publishing:
                msg = self.create_joystick_message()
                if msg:
                    self.publisher.publish(msg)
                rate.sleep()
        except rospy.ROSInterruptException:
            print("ROS interrupted")
        except Exception as e:
            print(f"Error publishing joystick data: {e}")
    
    def start_publishing(self):
        """Start publishing joystick data in a separate thread."""
        if self.publishing:
            print("Already publishing joystick data")
            return
        
        self.publishing = True
        self.publish_thread = threading.Thread(target=self.publish_joystick_data)
        self.publish_thread.daemon = True
        self.publish_thread.start()
        print("✓ Started publishing joystick data")
    
    def stop_publishing(self):
        """Stop publishing joystick data."""
        if not self.publishing:
            print("Not currently publishing")
            return
        
        self.publishing = False
        if hasattr(self, 'publish_thread'):
            self.publish_thread.join(timeout=1.0)
        print("✓ Stopped publishing joystick data")
    
    def set_intervention(self, active: bool):
        """
        Set intervention state.
        
        Args:
            active: Whether intervention should be active
        """
        self.intervention_active = active
        if active:
            self.right_grip = 1.0  # Trigger intervention
            print("✓ Intervention ACTIVATED (right_grip = 1.0)")
        else:
            self.right_grip = 0.0  # Disable intervention
            print("✓ Intervention DEACTIVATED (right_grip = 0.0)")
    
    def trigger_success(self):
        """Trigger episode success."""
        self.left_second_button_pressed = True
        print("✓ Episode SUCCESS triggered (left_second_button)")
        time.sleep(0.1)  # Brief press
        self.left_second_button_pressed = False
    
    def trigger_failure(self):
        """Trigger episode failure."""
        self.left_first_button_pressed = True
        print("✓ Episode FAILURE triggered (left_first_button)")
        time.sleep(0.1)  # Brief press
        self.left_first_button_pressed = False
    
    def start_bezier_tool(self):
        """Start the Bézier trajectory tool."""
        if self.bezier_process and self.bezier_process.poll() is None:
            print("Bézier tool is already running")
            return
        
        # Get the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        bezier_script = os.path.join(script_dir, "robotic_bezier_action_record_tool.py")
        
        if not os.path.exists(bezier_script):
            print(f"Error: Bézier script not found at {bezier_script}")
            return
        
        try:
            cmd = [
                sys.executable, 
                bezier_script, 
                "--mode", "play_actions", 
                "--rate", "15.0", 
                "--debug"
            ]
            
            self.bezier_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            print("✓ Started Bézier trajectory tool")
            print(f"  Command: {' '.join(cmd)}")
            print(f"  PID: {self.bezier_process.pid}")
            
        except Exception as e:
            print(f"Error starting Bézier tool: {e}")
            self.bezier_process = None
    
    def stop_bezier_tool(self):
        """Stop the Bézier trajectory tool."""
        if not self.bezier_process or self.bezier_process.poll() is not None:
            print("Bézier tool is not running")
            return
        
        try:
            # Send SIGTERM to the process group
            os.killpg(os.getpgid(self.bezier_process.pid), signal.SIGTERM)
            
            # Wait for process to terminate
            try:
                self.bezier_process.wait(timeout=5.0)
                print("✓ Bézier tool stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if it doesn't stop gracefully
                os.killpg(os.getpgid(self.bezier_process.pid), signal.SIGKILL)
                self.bezier_process.wait()
                print("✓ Bézier tool force stopped")
                
        except Exception as e:
            print(f"Error stopping Bézier tool: {e}")
        finally:
            self.bezier_process = None
    
    def get_bezier_status(self) -> str:
        """Get the status of the Bézier tool."""
        if not self.bezier_process:
            return "Not started"
        elif self.bezier_process.poll() is None:
            return f"Running (PID: {self.bezier_process.pid})"
        else:
            return f"Stopped (exit code: {self.bezier_process.returncode})"
    
    def print_status(self):
        """Print current simulator status."""
        print("\n" + "="*60)
        print("JOYSTICK SIMULATOR STATUS")
        print("="*60)
        print(f"ROS Available: {'✓' if self.enable_ros else '✗'}")
        print(f"Publishing: {'✓' if self.publishing else '✗'}")
        print(f"Service Server: {'✓' if self.service_server else '✗'}")
        print(f"Intervention: {'ACTIVE' if self.intervention_active else 'INACTIVE'}")
        print(f"Left Grip: {self.left_grip:.1f}")
        print(f"Right Grip: {self.right_grip:.1f}")
        print(f"Bézier Tool: {self.get_bezier_status()}")
        print("="*60)
        if self.service_server:
            print("Service: /robot_control/start_record_tool (ACTIVE)")
        print("="*60)
    
    def print_menu(self):
        """Print the main menu."""
        print("\n" + "="*60)
        print("META VR JOYSTICK SIMULATOR - MAIN MENU")
        print("="*60)
        print("1. Start/Stop Publishing Joystick Data")
        print("2. Toggle Intervention (Right Grip)")
        print("3. Start Bézier Trajectory Tool")
        print("4. Stop Bézier Trajectory Tool")
        print("5. Trigger Episode Success")
        print("6. Trigger Episode Failure")
        print("7. Show Status")
        print("8. Test Record Tool Service")
        print("9. Quick Start (Publishing + Bézier Tool)")
        print("10. Quick Stop (Stop All)")
        print("0. Exit")
        print("="*60)
    
    def quick_start(self):
        """Quick start: Enable publishing and start Bézier tool."""
        print("\n🚀 QUICK START: Initializing simulation...")
        
        # Start publishing
        if not self.publishing:
            self.start_publishing()
            time.sleep(0.5)
        
        # Start Bézier tool
        self.start_bezier_tool()
        time.sleep(1.0)
        
        print("✓ Quick start completed!")
        print("  - Joystick data publishing: ACTIVE")
        print("  - Bézier trajectory tool: RUNNING")
        print("  - Use option 2 to toggle intervention when ready")
    
    def quick_stop(self):
        """Quick stop: Stop everything."""
        print("\n🛑 QUICK STOP: Shutting down simulation...")
        
        # Disable intervention first
        if self.intervention_active:
            self.set_intervention(False)
        
        # Stop Bézier tool
        self.stop_bezier_tool()
        time.sleep(0.5)
        
        # Stop publishing
        self.stop_publishing()
        
        print("✓ Quick stop completed!")
        print("  - All processes stopped")
    
    def test_record_tool_service(self):
        """Test the record tool service by calling it locally."""
        if not self.enable_ros:
            print("❌ ROS not available - cannot test service")
            return
        
        try:
            print("\n🧪 Testing record tool service...")
            
            # Wait for the service to be available
            rospy.wait_for_service('/robot_control/start_record_tool', timeout=2.0)
            
            # Create service proxy
            start_record_tool = rospy.ServiceProxy('/robot_control/start_record_tool', Trigger)
            
            # Call the service
            response = start_record_tool()
            
            if response.success:
                print(f"✅ Service call successful: {response.message}")
            else:
                print(f"❌ Service call failed: {response.message}")
                
        except rospy.ServiceException as e:
            print(f"❌ Service call failed: {e}")
        except rospy.ROSException as e:
            print(f"❌ ROS error: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
    
    def run_interactive_menu(self):
        """Run the interactive menu."""
        print("Meta VR Joystick Simulator")
        print("Simulates Quest3 controller for testing intervention functionality")
        print("Provides ROS service: /robot_control/start_record_tool")
        
        if not self.enable_ros:
            print("\n⚠️  WARNING: ROS not available! Limited functionality.")
        
        try:
            while self.running:
                self.print_menu()
                
                try:
                    choice = input("\nEnter your choice (0-10): ").strip()
                    
                    if choice == '1':
                        if self.publishing:
                            self.stop_publishing()
                        else:
                            self.start_publishing()
                    
                    elif choice == '2':
                        self.set_intervention(not self.intervention_active)
                    
                    elif choice == '3':
                        self.start_bezier_tool()
                    
                    elif choice == '4':
                        self.stop_bezier_tool()
                    
                    elif choice == '5':
                        self.trigger_success()
                    
                    elif choice == '6':
                        self.trigger_failure()
                    
                    elif choice == '7':
                        self.print_status()
                    
                    elif choice == '8':
                        self.test_record_tool_service()
                    
                    elif choice == '9':
                        self.quick_start()
                    
                    elif choice == '10':
                        self.quick_stop()
                    
                    elif choice == '0':
                        print("\n👋 Exiting simulator...")
                        self.running = False
                        break
                    
                    else:
                        print("❌ Invalid choice. Please try again.")
                
                except KeyboardInterrupt:
                    print("\n\n⚠️  Keyboard interrupt detected")
                    break
                except Exception as e:
                    print(f"❌ Error: {e}")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        print("\n🧹 Cleaning up...")
        
        # Stop intervention
        self.set_intervention(False)
        
        # Stop Bézier tool
        self.stop_bezier_tool()
        
        # Stop publishing
        self.stop_publishing()
        
        print("✓ Cleanup completed")


def main():
    """Main function."""
    print("Meta VR Joystick Simulator")
    print("=" * 40)
    
    if not ROS_AVAILABLE:
        print("Error: ROS dependencies not available")
        print("Please ensure ROS and noitom_hi5_hand_udp_python package are installed")
        return 1
    
    try:
        simulator = JoystickSimulator()
        simulator.run_interactive_menu()
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
