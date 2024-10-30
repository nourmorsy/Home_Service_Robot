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
```
### 3. Set Up the Conda Environments

The project uses multiple Conda environments to manage dependencies for different modules.
```bash
conda env create -f Project_environment.yml #This file includes dependencies for NLP, navigation, and general utilities.
conda env create -f dope.yml # This environment specifically handles the DOPE (Deep Object Pose Estimation) model dependencies.
```
### 4.  Download robot packages and AI Models


