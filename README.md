# Home Service Robot Project

## Table of Contents
- [Project Goal and Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-improvements)
- [Acknowledgments](#acknowledgments)
---

## Project Goal and Overview

The **Home Service Robot** is a groundbreaking project aimed at assisting the elderly and individuals with limited mobility by autonomously performing daily tasks. This robot integrates cutting-edge **multi-vision models**, **natural language processing (NLP)**, and **language grounding techniques** to ensure seamless human-robot interaction in home environments.

**Key components include:**
- **Multi-Vision Models** for object recognition and detection
- **Natural Language Understanding (NLU)** for command interpretation
- **PDDL-based Planning** for decision-making and task execution
- **MoveIt with 6DOF** for robotic arm manipulation

---

## System Architecture

- **Human-Robot Interaction**: Communicate with the robot using natural language through a messaging interface.
- **Integrated Multi-Vision Models**: Combines VQA, DOPE, and DINO models for rich environmental understanding.
- **Language Grounding**: Utilizes NLU and language models to convert language into actionable tasks.
- **Autonomous Navigation**: Leverages LiDAR and the ROS navigation stack for safe movement.
- **Object Manipulation**: Uses MoveIt and a 5DOF robotic arm for complex object handling.

![System Architecture](.png)

---
## Directory Structure
```
Home_Service_robot/
├── src/  # Source code directory
│   ├── command_parsing.py  # Command classification and parsing
│   ├── object_detection.py  # Object detection with GroundingDINO
│   ├── pose_estimation.py  # Pose estimation with DOPE
│   ├── navigation.py       # Navigation logic and control
│   ├── teleop.py           # Teleoperation code for movement
│   └── requirements.txt     # List of dependencies
├── models/                   # Pre-trained models and weights
├── data/                     # Datasets (e.g., HOPE)
└── README.md                 # Project README
```
---


