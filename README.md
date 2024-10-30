# Home Service Robot Project

## Table of Contents
- [Project Goal and Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Future Improvements](#future-Improvements)
- [Acknowledgments](#acknowledgments)
---

## Project Goal and Overview

The **Home Service Robot** is a groundbreaking project aimed at assisting the elderly and individuals with limited mobility by autonomously performing daily tasks. This robot integrates cutting-edge **multi-vision models**, **natural language processing (NLP)**, and **language grounding techniques** to ensure seamless human-robot interaction in home environments.

**Key components include:**
- **Multi-Vision Models** for Enviroment recognition and detection
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

![System Architecture](./System-arch.png)

---
## Directory Structure
```
Home_Service_robot/
├── src/  # Source code directory
│   ├── Classification   # Classification package
│   ├── detection        # GroundingDINO package
│   ├── dope             # Deep Pose estimation package
│   ├── navigation       # Navigation package
│   ├── VQA              # Visual question answering package
│   ├── LLM              # Large Language Model package
│   ├── planning         # PDDLStream package
│   ├── manipulation                # Manipulation package
│   ├── Porject_environment.yml     # List of project dependencies for conda enviroment
│   └── DOPE.yml                    # List of Dope package dependencies for conda enviroment                 
└── README.md                 # Project README
```
---
## Installation

### 1. Pre-installation Requirements

Ensure the following software is installed on your system:
- **ROS (Robot Operating System)**: Required for robot control and navigation.
  - Installation instructions: [ROS Installation Guide](http://wiki.ros.org/ROS/Installation)
- **Python 3.x**: The project requires Python 3 for compatibility with various models and libraries.
- **Anaconda**: Recommended for managing Python environments.
  - Installation instructions: [Anaconda Installation Guide](https://docs.anaconda.com/anaconda/install/)

### 2. Clone the Repository

Start by cloning this repository to your local machine:
```bash
git clone https://github.com/yourusername/Home_Service_robot.git
cd Home_Service_robot
catkin make 
```
### 3. Set Up the Conda Environments

The project uses multiple Conda environments to manage dependencies for different modules.
```bash
conda env create -f Project_environment.yml #This file includes dependencies for NLP, navigation, and general utilities.
conda env create -f dope.yml # This environment specifically handles the DOPE (Deep Object Pose Estimation) model dependencies.
```
### 4.  Download robot packages and AI Models

- **Turtlebot package**: Required for robot control and navigation.
  - Installation instructions: [Turtlebot Installation Guide](https://github.com/turtlebot)
  - Place the model in main directory: ``` src/ ```
- **DOPE Model**: Required for Object Pose Estimation.
  - Installation instructions: [DOPE Installation Guide](https://github.com/NVlabs/Deep_Object_Pose)
  - Place the model in ``` src/dope/ ```
- **VQA Model**: Required for Visual Question Answering.
  - Installation instructions: [VQA Installation Guide](https://github.com/dandelin/ViLT)
  - Place the model in ``` src/VQA/ ```
- **GroundingDINO Model**: Required for Object Detection.
  - Installation instructions: [GroundingDINO Installation Guide](https://github.com/IDEA-Research/GroundingDINO)
  - Place the model in ``` src/detection/ ```
- **LLM Model**: Required for human robot language interaction.
  - Installation instructions: [LLM Installation Guide](https://github.com/tincans-ai/gazelle)
  - Place the model in ``` src/LLM/ ```
- **PPDLStream Model**: Required for decision-making and task execution .
  - Installation instructions: [LLM Installation Guide](https://github.com/caelan/pddlstream)
  - Place the package in ``` src/planning/ ```
-  **Finally**: in the main directory : ```catkin make ```

---
### 6.  Future Improvements

 - **Decision Trees for Improved Planning**: We plan to incorporate decision trees to enhance task-planning capabilities, allowing the       robot to make smarter choices when performing tasks.
 - **Additional Multi-Modal Models**: Adding other vision and language models for improved perception and interaction.

---
### 7.  Acknowledgments


  - This project is inspired by the work of [**Sebastian Castro**](https://github.com/sea-bass). His contributions and insightful blog       post, [2020 Review: Service Robotics – MIT CSAIL](https://roboticseabass.com/2020/12/30/2020-review-service-robotics-mit-csail/),       provided invaluable inspiration and guidance for our team.
  - We extend our gratitude for his dedication to advancing the field of robotics, which has greatly influenced our approach and the         development of this project.

 
    
    
