#!/usr/bin/env python3
"""
Simple MP4 Frame Publisher Tool (No cv_bridge dependency)

This tool extracts specific frames from an MP4 video file and publishes them to ROS topics
without using cv_bridge, making it compatible with NumPy 2.x.

Usage:
    python mp4_frame_publisher_simple.py --video episode_000003.mp4 --info
    python mp4_frame_publisher_simple.py --video episode_000003.mp4 --frame 100
"""

import argparse
import cv2
import numpy as np
import sys
import os
import time
import threading

# ROS imports with fallback
try:
    import rospy
    from sensor_msgs.msg import Image, CameraInfo
    ROS_AVAILABLE = True
except ImportError:
    print("⚠️  ROS packages not available. Running in offline mode.")
    ROS_AVAILABLE = False
    rospy = None
    Image = None
    CameraInfo = None

def cv2_to_ros_image(cv_image, encoding="rgb8"):
    """
    Convert OpenCV image to ROS Image message without using cv_bridge.
    Compatible with NumPy 2.x.
    
    Args:
        cv_image: OpenCV image (numpy array)
        encoding: Image encoding (default: "rgb8")
        
    Returns:
        sensor_msgs/Image message
    """
    if not ROS_AVAILABLE:
        raise RuntimeError("ROS not available")
    
    ros_image = Image()
    
    if len(cv_image.shape) == 3:
        height, width, channels = cv_image.shape
    else:
        height, width = cv_image.shape
        channels = 1
    
    ros_image.height = height
    ros_image.width = width
    ros_image.encoding = encoding
    ros_image.is_bigendian = 0
    ros_image.step = width * channels
    ros_image.data = cv_image.flatten().tobytes()
    
    return ros_image

class SimpleMP4FramePublisher:
    def __init__(self, video_path):
        """
        Initialize the Simple MP4 Frame Publisher
        
        Args:
            video_path (str): Path to the MP4 video file
        """
        self.video_path = video_path
        
        # Check if video file exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Try multiple video backends for better compatibility
        backends = [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER, cv2.CAP_ANY]
        self.cap = None
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(video_path, backend)
                if self.cap.isOpened():
                    # Test if we can actually read a frame
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        print(f"✅ Successfully opened video with backend: {backend}")
                        # Reset to beginning
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break
                    else:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                print(f"⚠️  Backend {backend} failed: {e}")
                if self.cap:
                    self.cap.release()
                    self.cap = None
        
        if self.cap is None or not self.cap.isOpened():
            raise ValueError(f"Cannot open video file with any backend: {video_path}")
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        # Additional validation for problematic videos
        if self.total_frames <= 0:
            print("⚠️  Warning: Could not determine frame count, attempting manual counting...")
            self.total_frames = self._count_frames_manually()
            
        print(f"📊 Video loaded - Backend: OpenCV, Frames: {self.total_frames}, Size: {self.width}x{self.height}")
        
        # Initialize ROS if available
        self.ros_available = False
        if ROS_AVAILABLE:
            try:
                rospy.init_node('simple_mp4_frame_publisher', anonymous=True)
                self.image_pub = rospy.Publisher('/camera/eval/image_raw', Image, queue_size=1)
                self.ros_available = True
                print("✅ ROS initialized successfully")
            except Exception as e:
                print(f"⚠️  ROS initialization failed: {e}")
                print("Will run in offline mode")
        
        # Camera info subscription for header synchronization
        self.latest_camera_info = None
        self.camera_info_sub = None
        if self.ros_available:
            try:
                self.camera_info_sub = rospy.Subscriber('/camera/rgb/camera_info', CameraInfo, self._camera_info_callback)
                print("✅ Subscribed to /camera/rgb/camera_info for header synchronization")
            except Exception as e:
                print(f"⚠️  Failed to subscribe to camera_info: {e}")
        
        # Background publishing control
        self.publishing_thread = None
        self.publishing_active = False
        self.publish_rate = None
    
    def _count_frames_manually(self):
        """Manually count frames by reading through the video"""
        frame_count = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        while True:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            frame_count += 1
            
            # Avoid infinite loops
            if frame_count > 10000:
                print("⚠️  Frame counting exceeded 10000, stopping")
                break
        
        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"📊 Manual frame count: {frame_count}")
        return frame_count
    
    def _camera_info_callback(self, msg):
        """
        Callback function for camera_info subscription
        
        Args:
            msg (CameraInfo): Camera info message from /camera/rgb/camera_info
        """
        self.latest_camera_info = msg
        # Optional: Log when we first receive camera info
        if not hasattr(self, '_camera_info_received'):
            self._camera_info_received = True
            print(f"📷 Received camera_info: frame_id={msg.header.frame_id}, "
                  f"resolution={msg.width}x{msg.height}")
    
    def print_video_info(self):
        """Print video information in a formatted way"""
        print("\n" + "="*60)
        print("📹 VIDEO INFORMATION")
        print("="*60)
        print(f"📁 File: {self.video_path}")
        print(f"🎬 Total Frames: {self.total_frames:,}")
        print(f"⏱️  Frame Rate: {self.fps:.2f} FPS")
        print(f"⏰ Duration: {self.duration:.2f} seconds")
        print(f"📐 Resolution: {self.width}x{self.height}")
        print(f"🔢 Frame Range: 0 to {self.total_frames-1}")
        print(f"🚀 ROS Publishing: {'✅ Available' if self.ros_available else '❌ Disabled'}")
        
        # Camera info status
        if self.ros_available:
            camera_info_status = "✅ Connected" if self.latest_camera_info is not None else "⏳ Waiting"
            print(f"📷 Camera Info Sub: {camera_info_status}")
            if self.latest_camera_info is not None:
                print(f"   └─ Frame ID: {self.latest_camera_info.header.frame_id}")
                print(f"   └─ Camera Resolution: {self.latest_camera_info.width}x{self.latest_camera_info.height}")
        
        print("="*60)
    
    def extract_frame(self, frame_number):
        """
        Extract a specific frame from the video
        
        Args:
            frame_number (int): Frame number to extract (0-based)
            
        Returns:
            numpy.ndarray: Frame image as BGR array, or None if failed
        """
        if frame_number < 0 or frame_number >= self.total_frames:
            print(f"❌ Frame number {frame_number} is out of range [0, {self.total_frames-1}]")
            return None
        
        # Set frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        # Read frame
        ret, frame = self.cap.read()
        if not ret:
            print(f"❌ Failed to read frame {frame_number}")
            return None
        
        print(f"✅ Successfully extracted frame {frame_number}")
        print(f"   Frame shape: {frame.shape} (H x W x C)")
        print(f"   Data type: {frame.dtype}")
        print(f"   Value range: [{frame.min()}, {frame.max()}]")
        return frame
    
    def publish_frame(self, frame, frame_number):
        """
        Publish frame to ROS topic with header synchronized from camera_info
        
        Args:
            frame (numpy.ndarray): Frame image as BGR array
            frame_number (int): Frame number for logging
            
        Returns:
            bool: True if published successfully, False otherwise
        """
        if not self.ros_available:
            print("⚠️  ROS not available, cannot publish frame")
            return False
        
        try:
            # Convert BGR to RGB (ROS standard)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to ROS Image message using our custom function
            ros_image = cv2_to_ros_image(frame_rgb, "rgb8")
            
            # Use camera_info header if available, otherwise use current time
            if self.latest_camera_info is not None:
                # Copy header completely from camera_info (including timestamp)
                ros_image.header = self.latest_camera_info.header
                header_source = "camera_info"
            else:
                # Fallback to default header
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = f"video_frame_{frame_number}"
                header_source = "default"
            
            # Publish
            self.image_pub.publish(ros_image)
            
            print(f"📡 Published frame {frame_number} to /camera/eval/image_raw")
            print(f"   Timestamp: {ros_image.header.stamp}")
            print(f"   Frame ID: {ros_image.header.frame_id}")
            print(f"   Header source: {header_source}")
            print(f"   Image size: {ros_image.width}x{ros_image.height}")
            print(f"   Encoding: {ros_image.encoding}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to publish frame {frame_number}: {e}")
            return False
    
    def extract_and_publish_frame(self, frame_number):
        """
        Extract and publish a specific frame
        
        Args:
            frame_number (int): Frame number to extract and publish
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Extract frame
        frame = self.extract_frame(frame_number)
        if frame is None:
            return False
        
        # Publish frame if ROS is available
        if self.ros_available:
            return self.publish_frame(frame, frame_number)
        else:
            print(f"✅ Frame {frame_number} extracted successfully")
            print(f"💡 To publish to ROS, ensure ROS is running and try again")
            return True
    
    def save_frame_as_image(self, frame_number, output_path=None):
        """
        Extract and save a frame as an image file
        
        Args:
            frame_number (int): Frame number to extract
            output_path (str): Output file path (optional)
            
        Returns:
            str: Path to saved image file, or None if failed
        """
        frame = self.extract_frame(frame_number)
        if frame is None:
            return None
        
        if output_path is None:
            output_path = f"frame_{frame_number:06d}.jpg"
        
        try:
            cv2.imwrite(output_path, frame)
            print(f"💾 Saved frame {frame_number} to: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Failed to save frame {frame_number}: {e}")
            return None
    
    def start_continuous_publishing(self, frame_number, frequency_hz=1.0):
        """
        Start publishing a specific frame continuously at given frequency
        
        Args:
            frame_number (int): Frame number to publish continuously
            frequency_hz (float): Publishing frequency in Hz (default: 1.0)
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.ros_available:
            print("⚠️  ROS not available, cannot start continuous publishing")
            return False
        
        if self.publishing_active:
            print("⚠️  Continuous publishing already active. Stop it first.")
            return False
        
        # Validate frame number
        if frame_number < 0 or frame_number >= self.total_frames:
            print(f"❌ Frame number {frame_number} is out of range [0, {self.total_frames-1}]")
            return False
        
        # Extract the frame once
        frame = self.extract_frame(frame_number)
        if frame is None:
            return False
        
        # Convert to RGB once (since we'll reuse it)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store publishing parameters
        self.current_frame = frame_rgb
        self.current_frame_number = frame_number
        self.publish_rate = rospy.Rate(frequency_hz)
        self.publishing_active = True
        
        # Start publishing thread
        self.publishing_thread = threading.Thread(
            target=self._continuous_publishing_loop,
            daemon=True
        )
        self.publishing_thread.start()
        
        print(f"🚀 Started continuous publishing of frame {frame_number} at {frequency_hz} Hz")
        print(f"📡 Publishing to: /camera/eval/image_raw")
        print(f"⏹️  Use stop_continuous_publishing() or Ctrl+C to stop")
        return True
    
    def _continuous_publishing_loop(self):
        """Background thread loop for continuous publishing"""
        publish_count = 0
        start_time = time.time()
        
        try:
            while self.publishing_active and not rospy.is_shutdown():
                try:
                    # Create ROS Image message
                    ros_image = cv2_to_ros_image(self.current_frame, "rgb8")
                    
                    # Use camera_info header if available, otherwise use current time
                    if self.latest_camera_info is not None:
                        # Copy header completely from camera_info (including timestamp)
                        ros_image.header = self.latest_camera_info.header
                    else:
                        # Fallback to default header
                        ros_image.header.stamp = rospy.Time.now()
                        ros_image.header.frame_id = f"video_frame_{self.current_frame_number}_continuous"
                    
                    # Publish
                    self.image_pub.publish(ros_image)
                    publish_count += 1
                    
                    # Log status periodically
                    if publish_count % 10 == 1:
                        elapsed_time = time.time() - start_time
                        actual_rate = publish_count / elapsed_time if elapsed_time > 0 else 0
                        print(f"📊 Published {publish_count} frames, "
                              f"actual rate: {actual_rate:.2f} Hz, "
                              f"frame: {self.current_frame_number}")
                    
                    # Sleep according to rate
                    self.publish_rate.sleep()
                    
                except Exception as e:
                    print(f"❌ Error in publishing loop: {e}")
                    break
                    
        except KeyboardInterrupt:
            print("\n⏹️  Publishing interrupted by user")
        finally:
            self.publishing_active = False
            print(f"🏁 Continuous publishing stopped. Total published: {publish_count} frames")
    
    def stop_continuous_publishing(self):
        """Stop continuous publishing"""
        if not self.publishing_active:
            print("⚠️  No continuous publishing is active")
            return False
        
        print("⏹️  Stopping continuous publishing...")
        self.publishing_active = False
        
        # Wait for thread to finish
        if self.publishing_thread and self.publishing_thread.is_alive():
            self.publishing_thread.join(timeout=2.0)
        
        print("✅ Continuous publishing stopped")
        return True
    
    def is_publishing(self):
        """Check if continuous publishing is active"""
        return self.publishing_active
    
    def get_publishing_status(self):
        """Get current publishing status information"""
        if not self.publishing_active:
            return {
                'active': False,
                'frame_number': None,
                'frequency_hz': None
            }
        
        return {
            'active': True,
            'frame_number': self.current_frame_number,
            'frequency_hz': self.publish_rate._freq if hasattr(self.publish_rate, '_freq') else None,
            'thread_alive': self.publishing_thread.is_alive() if self.publishing_thread else False
        }
    
    def __del__(self):
        """Cleanup resources"""
        # Stop publishing thread
        if hasattr(self, 'publishing_active'):
            self.stop_continuous_publishing()
        
        # Unsubscribe from camera_info
        if hasattr(self, 'camera_info_sub') and self.camera_info_sub is not None:
            try:
                self.camera_info_sub.unregister()
            except:
                pass
        
        # Release video capture
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()

def run_interactive_mode(publisher):
    """
    Run interactive mode for continuous publishing control
    
    Args:
        publisher (SimpleMP4FramePublisher): The publisher instance
    """
    print("\n🎮 INTERACTIVE CONTINUOUS PUBLISHING MODE")
    print("="*60)
    print("Commands:")
    print("  start <frame> [freq]  - Start continuous publishing (default freq: 1.0 Hz)")
    print("  stop                  - Stop continuous publishing")
    print("  status                - Show publishing status")
    print("  change <freq>         - Change frequency (if publishing)")
    print("  info                  - Show video information")
    print("  quit/exit             - Exit interactive mode")
    print("="*60)
    
    while True:
        try:
            # Show current status in prompt
            status_indicator = "🔴" if not publisher.is_publishing() else "🟢"
            user_input = input(f"\n{status_indicator} Enter command: ").strip().lower()
            
            if not user_input:
                continue
                
            parts = user_input.split()
            command = parts[0]
            
            if command in ['quit', 'exit', 'q']:
                if publisher.is_publishing():
                    print("⏹️  Stopping publishing before exit...")
                    publisher.stop_continuous_publishing()
                print("👋 Exiting interactive mode")
                break
                
            elif command == 'start':
                if len(parts) < 2:
                    print("❌ Usage: start <frame_number> [frequency]")
                    continue
                    
                try:
                    frame_num = int(parts[1])
                    freq = float(parts[2]) if len(parts) > 2 else 1.0
                    
                    if publisher.is_publishing():
                        print("⏹️  Stopping current publishing...")
                        publisher.stop_continuous_publishing()
                        time.sleep(0.5)
                    
                    success = publisher.start_continuous_publishing(frame_num, freq)
                    if not success:
                        print(f"❌ Failed to start publishing frame {frame_num}")
                        
                except ValueError:
                    print("❌ Invalid frame number or frequency")
                    
            elif command == 'stop':
                if publisher.is_publishing():
                    publisher.stop_continuous_publishing()
                else:
                    print("⚠️  No publishing is currently active")
                    
            elif command == 'status':
                status = publisher.get_publishing_status()
                print(f"\n📊 PUBLISHING STATUS:")
                print(f"   Active: {'✅ Yes' if status['active'] else '❌ No'}")
                if status['active']:
                    print(f"   Frame: {status['frame_number']}")
                    print(f"   Frequency: {status.get('frequency_hz', 'Unknown')} Hz")
                    print(f"   Thread: {'✅ Alive' if status.get('thread_alive') else '❌ Dead'}")
                    
            elif command == 'change':
                if len(parts) < 2:
                    print("❌ Usage: change <frequency>")
                    continue
                    
                try:
                    new_freq = float(parts[1])
                    if publisher.is_publishing():
                        current_frame = publisher.current_frame_number
                        print(f"🔄 Changing frequency to {new_freq} Hz...")
                        publisher.stop_continuous_publishing()
                        time.sleep(0.5)
                        publisher.start_continuous_publishing(current_frame, new_freq)
                    else:
                        print("⚠️  No publishing is currently active")
                except ValueError:
                    print("❌ Invalid frequency value")
                    
            elif command == 'info':
                publisher.print_video_info()
                
            else:
                print(f"❌ Unknown command: {command}")
                print("💡 Available commands: start, stop, status, change, info, quit")
                
        except KeyboardInterrupt:
            print("\n⏹️  Interrupted. Stopping publishing...")
            if publisher.is_publishing():
                publisher.stop_continuous_publishing()
            print("👋 Exiting interactive mode")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Simple MP4 Frame Publisher (NumPy 2.x compatible)')
    parser.add_argument('--video', '-v', required=True, help='Path to MP4 video file')
    parser.add_argument('--frame', '-f', type=int, help='Specific frame number to extract and publish')
    parser.add_argument('--info', '-i', action='store_true', help='Show video information only')
    parser.add_argument('--save', '-s', action='store_true', help='Save frame as image file instead of publishing')
    parser.add_argument('--output', '-o', help='Output image file path (for --save option)')
    parser.add_argument('--continuous', '-c', action='store_true', help='Publish frame continuously in background')
    parser.add_argument('--frequency', '--freq', type=float, default=1.0, help='Publishing frequency in Hz (default: 1.0)')
    parser.add_argument('--interactive', action='store_true', help='Run in interactive mode for continuous publishing control')
    
    args = parser.parse_args()
    
    try:
        # Create publisher
        publisher = SimpleMP4FramePublisher(args.video)
        
        if args.info:
            # Show video info only
            publisher.print_video_info()
        elif args.interactive:
            # Interactive mode for continuous publishing control
            publisher.print_video_info()
            run_interactive_mode(publisher)
        elif args.frame is not None:
            # Extract and publish/save specific frame
            publisher.print_video_info()
            print(f"\n🎯 Processing frame {args.frame}...")
            
            if args.save:
                # Save frame as image
                saved_path = publisher.save_frame_as_image(args.frame, args.output)
                if saved_path:
                    print(f"✅ Frame {args.frame} saved successfully")
                else:
                    print(f"❌ Failed to save frame {args.frame}")
                    sys.exit(1)
            elif args.continuous:
                # Start continuous publishing
                success = publisher.start_continuous_publishing(args.frame, args.frequency)
                if success:
                    try:
                        print("📡 Continuous publishing started. Press Ctrl+C to stop...")
                        # Keep the main thread alive
                        while publisher.is_publishing() and not rospy.is_shutdown():
                            time.sleep(1.0)
                    except KeyboardInterrupt:
                        print("\n⏹️  Stopping continuous publishing...")
                    finally:
                        publisher.stop_continuous_publishing()
                else:
                    print(f"❌ Failed to start continuous publishing")
                    sys.exit(1)
            else:
                # Publish frame to ROS once
                success = publisher.extract_and_publish_frame(args.frame)
                if success:
                    print(f"✅ Frame {args.frame} processed successfully")
                    
                    # Keep node alive for a moment to ensure publishing
                    if publisher.ros_available:
                        print("⏳ Keeping node alive for 2 seconds to ensure publishing...")
                        rospy.sleep(2)
                else:
                    print(f"❌ Failed to process frame {args.frame}")
                    sys.exit(1)
        else:
            # Default: show info and usage examples
            publisher.print_video_info()
            print(f"\n💡 USAGE EXAMPLES:")
            print(f"   Show info:         python {os.path.basename(sys.argv[0])} --video {args.video} --info")
            print(f"   Publish once:      python {os.path.basename(sys.argv[0])} --video {args.video} --frame 40")
            print(f"   Continuous 1Hz:    python {os.path.basename(sys.argv[0])} --video {args.video} --frame 40 --continuous")
            print(f"   Continuous 10Hz:   python {os.path.basename(sys.argv[0])} --video {args.video} --frame 40 --continuous --frequency 10")
            print(f"   Interactive mode:  python {os.path.basename(sys.argv[0])} --video {args.video} --interactive")
            print(f"   Save frame:        python {os.path.basename(sys.argv[0])} --video {args.video} --frame 40 --save")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
