v3 Hardware
ESP32-CAM: camera only. It serves an MJPEG stream over WiFi, with no servo, mic, or PIR attached. This avoids the GPIO crunch and the servo/camera timer conflict.
Arduino Nano 33 IoT: PIR input and servo output. It talks to the laptop over WiFi using MQTT (Mosquitto on the laptop). It also sends a heartbeat. The servo gets its own 5V supply with a shared ground, and the PIR’s output goes to a Nano GPIO.
Laptop: runs the entire ML pipeline, the state machine, the mic (bark detection and recording owner commands), and audio playback to the Bluetooth speaker.
Cloud: frontier model for NUDGE and verification.
PIR + Servo ── Nano 33 IoT ──(MQTT/WiFi)──┐
                                          ├── Laptop (state machine, ML, mic, speaker) ── Cloud LLM
ESP32-CAM ──(MJPEG/WiFi)──────────────────┘
v3 State machine

The clock drives the mealtime window, and PIR picks the branch. A session starts at VERIFY and has an attempt cap and a total timeout.

IDLE: outside meal windows, do nothing. Inside a window with no motion, go to NUDGE every 10 minutes. Inside a window with PIR motion, start a session and go to VERIFY.
NUDGE: frontier call, with canned time-based fallbacks on failure. Return to IDLE.
VERIFY: detector confirms a dog. If the detector is unsure, one frontier check. Dog present goes to COMMAND, otherwise IDLE.
COMMAND: pick “sit” or “down” at random, avoiding the same command as last round when possible. The laptop plays the recording, which prevents repeat-command loops.
WATCH: detector, posture classifier, temporal gate (about 80% of frames over about 2s), then one frontier check before rewarding. If the cloud fails, fall back to local-only with a stricter threshold. Reward requires target posture, no barking, and the daily cap not reached. On success go to REWARD. On timeout go to VERIFY.
REWARD: the laptop sends a dispense command to the Nano. The number of servo turns scales with the confidence score. Then a cooldown of a few seconds, then back to VERIFY.
ABORT (new, now defined): triggered by the dog leaving the frame for more than N seconds, the attempt cap being hit, the session timeout, or the daily cap being reached. It always goes to IDLE with a session cooldown.

Safety: a daily treat cap and a low-food/jam check. The simplest version is an alert after N dispenses without a hopper-level change, or a manual “refilled” button.

v3 Pipeline (laptop)
A YOLO-nano-class detector filters for the COCO “dog” class on frames.
A posture classifier on the dog crop labels sit, down, stand, or other.
The temporal gate smooths per-frame labels.
The frontier LLM gets 1-3 key frames as a final verifier.
The bark check runs on the laptop mic.
Evaluation plan
Per-class precision and recall for the posture classifier, with a confusion matrix, tuned for precision on the reward decision.
Per-stage latency (frame to detector to classifier to gate to LLM to servo).
Robustness across lighting: morning, midday, evening, and lights on or off.
An ablation comparing local-only against local plus the frontier verifier.
Cloud-failure behavior and false-reward rate over a fixed test session.
What to do first

I’d start with data collection, because it is the biggest risk. Mount the ESP32-CAM where it will actually go, record short clips of your dog doing sit, down, stand, and walking around, and check whether sit and down are visually separable from that angle and distance. If they are, the rest is straightforward. If not, move the camera now, before any software exists.

A good build order after that is: (1) ESP32 MJPEG stream plus a laptop viewer, (2) labeling and training the posture classifier, (3) the detector-to-classifier-to-gate pipeline run offline on recorded clips, (4) the Nano MQTT and servo, (5) the state machine, (6) the frontier verifier and NUDGE.
