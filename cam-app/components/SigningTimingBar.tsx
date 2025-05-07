import React, { useEffect, useState, useRef } from 'react';
import { View, StyleSheet, Animated, Text } from 'react-native';

// Props interface for the SigningTimingBar component
interface SigningTimingBarProps {
  isRecording: boolean;
  totalDuration?: number; // Total duration in ms (default: 5000ms)
  preparationTime?: number; // Time to prepare in ms (default: 1000ms)
  onRecordingPhaseChange?: (phase: 'idle' | 'prepare' | 'record' | 'complete') => void;
}

// Timing bar component to guide users during sign language recording
const SigningTimingBar: React.FC<SigningTimingBarProps> = ({ 
  isRecording, 
  totalDuration = 5000, // Total duration in ms (5 seconds)
  preparationTime = 1000, // Time to prepare in ms (1 second)
  onRecordingPhaseChange = () => {}, // Callback when phases change
}) => {
  const [phase, setPhase] = useState<'idle' | 'prepare' | 'record' | 'complete'>('idle');
  const progressAnim = useRef(new Animated.Value(0)).current;
  const flashAnim = useRef(new Animated.Value(0)).current;
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  // Reset and start animation when recording starts
  useEffect(() => {
    // Only initialize once when isRecording changes from false to true
    if (isRecording && phase === 'idle') {
      console.log("[DEBUG] Initializing timing bar");
      
      // Reset animation
      progressAnim.setValue(0);
      setPhase('prepare');
      onRecordingPhaseChange('prepare');
      
      // Start the animation
      Animated.timing(progressAnim, {
        toValue: 1,
        duration: totalDuration,
        useNativeDriver: false
      }).start(({ finished }) => {
        if (finished) {
          console.log("[DEBUG] Animation completed");
          setPhase('complete');
          onRecordingPhaseChange('complete');
        }
      });
      
      // Set timer to change from preparation to recording phase
      timerRef.current = setTimeout(() => {
        console.log("[DEBUG] Switching to record phase");
        setPhase('record');
        onRecordingPhaseChange('record');
        
        // Create flash animation when switching to record phase
        flashAnim.setValue(0.7); // Start with some opacity
        Animated.sequence([
          Animated.timing(flashAnim, {
            toValue: 0.2,  // Fade out
            duration: 300,
            useNativeDriver: false
          }),
          Animated.timing(flashAnim, {
            toValue: 0.7,  // Fade in again
            duration: 300,
            useNativeDriver: false
          }),
          Animated.timing(flashAnim, {
            toValue: 0,    // Fade out completely
            duration: 500,
            useNativeDriver: false
          })
        ]).start();
      }, preparationTime);
    } else if (!isRecording && phase !== 'idle') {
      console.log("[DEBUG] Resetting timing bar");
      // Stop animation if recording stops
      progressAnim.stopAnimation();
      Animated.timing(progressAnim).stop();
      setPhase('idle');
      
      // Clear the timer
      if (timerRef.current) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    }
    
    // Cleanup
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, [isRecording, phase, totalDuration, preparationTime, onRecordingPhaseChange]);

  // Calculate the width of the progress indicator
  const progressWidth = progressAnim.interpolate({
    inputRange: [0, 1],
    outputRange: ['0%', '100%']
  });
  
  // Determine the bar color based on the current position
  const barColor = progressAnim.interpolate({
    inputRange: [0, preparationTime/totalDuration, preparationTime/totalDuration + 0.01, 1],
    outputRange: ['#ff3b30', '#ff3b30', '#34c759', '#34c759']
  });
  
  // Text color should match the current phase
  const textColor = phase === 'prepare' ? '#ff3b30' : 
                    phase === 'record' ? '#34c759' : 
                    phase === 'complete' ? '#0096ff' : 'white';

  // Don't render if not recording
  if (!isRecording) return null;

  return (
    <View style={styles.container}>
      {/* Sign Now Flash Overlay - only visible during record phase */}
      {phase === 'record' && (
        <Animated.View 
          style={[
            styles.signNowFlash,
            { opacity: flashAnim }
          ]}
        />
      )}
      
      {/* Phase indicator text */}
      <Text style={[styles.phaseText, {color: textColor, fontSize: 24}]}>
        {phase === 'prepare' ? 'Get ready...' : 
         phase === 'record' ? 'SIGN NOW!' : 
         phase === 'complete' ? 'Processing...' : 'Recording...'}
      </Text>
      
      {/* Background bar */}
      <View style={styles.track}>
        {/* Red preparation zone */}
        <View style={[
          styles.zone, 
          styles.prepareZone, 
          { width: `${(preparationTime/totalDuration) * 100}%` }
        ]} />
        
        {/* Green recording zone */}
        <View style={[
          styles.zone, 
          styles.recordZone, 
          { width: `${((totalDuration-preparationTime)/totalDuration) * 100}%`, left: `${(preparationTime/totalDuration) * 100}%` }
        ]} />
        
        {/* Animated progress indicator */}
        <Animated.View 
          style={[
            styles.progressIndicator,
            { left: progressWidth }
          ]} 
        />
        
        {/* Animated progress fill */}
        <Animated.View 
          style={[
            styles.progress, 
            { 
              width: progressWidth,
              backgroundColor: barColor 
            }
          ]} 
        />
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    position: 'absolute',
    top: 140,
    left: 0,
    right: 0,
    alignItems: 'center',
    paddingHorizontal: 20,
    zIndex: 10, // Ensure it appears above other elements
  },
  signNowFlash: {
    position: 'absolute',
    top: -140, // Extend to top of screen
    left: -20,
    right: -20,
    bottom: -400, // Extend to bottom of screen
    backgroundColor: 'rgba(52, 199, 89, 0.3)', // Semi-transparent green
    borderWidth: 4,
    borderColor: '#34c759',
    zIndex: 5,
  },
  phaseText: {
    color: 'white',
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 8,
    textShadowColor: 'rgba(0, 0, 0, 0.75)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 3,
  },
  track: {
    height: 20,
    width: '100%',
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 10,
    overflow: 'hidden',
    flexDirection: 'row',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  zone: {
    height: '100%',
    position: 'absolute',
    opacity: 0.4,
  },
  prepareZone: {
    backgroundColor: '#ff3b30', // Red for preparation
    left: 0,
  },
  recordZone: {
    backgroundColor: '#34c759', // Green for recording
  },
  progress: {
    height: '100%',
    position: 'absolute',
    left: 0,
    backgroundColor: '#ff3b30', // Red by default, will be animated
    opacity: 0.8,
  },
  progressIndicator: {
    position: 'absolute',
    height: 30,
    width: 4,
    backgroundColor: 'white',
    top: -5,
    transform: [{ translateX: -2 }], // Center the indicator
    borderRadius: 2,
    zIndex: 10, // Ensure it's above other elements
  }
});

export default SigningTimingBar;