
# CCAM-DI: Reinforcement Learning-Based Autonomous Intersection Control

This repository implements a MARL-based autonomous intersection control system using PPO for learning vehicle policies and the SUMO traffic simulation. It evaluates robustness against data integrity attacks where malicious vehicles spoof position data. Experiments analyze training vs test-time perturbations and their impact on collision avoidance.

## 🚗 Overview

Autonomous intersection management is a key challenge in intelligent transportation systems. This project proposes a reinforcement learning approach where multiple vehicles (agents) learn to coordinate their actions to safely and efficiently cross intersections.

The core contribution of this work is the introduction of methods to improve training efficiency and policy performance by filtering irrelevant observations and actions during the learning process. These improvements significantly accelerate convergence and enhance the learned behavior in terms of collision avoidance and traffic throughput.

This repository also serves as the foundation for an extended study evaluating the robustness of the learned policies against data integrity attacks, where malicious vehicles manipulate their state information.

## 📂 Repository Structure

- `env/` – Simulation environment and interaction logic with the traffic simulator  
- `functions/` – Includes functions for adversarial attacks
- `saved_model/` – Trained models for evaluation and reproducibility  
- `config.py` – Configuration parameters for training and experiments  
- `run.py` – Main script to train and evaluate the RL agents  
- `requirements.txt` – Python dependencies  

## ⚙️ Methodology

The system is based on:
- Multi-Agent Reinforcement Learning (MARL)
- Policy optimization (e.g., PPO-style learning)
- SUMO for traffic simulation

Key innovations include:
- Filtering irrelevant observations to reduce state-space complexity  
- Filtering invalid or redundant actions to improve learning stability  
- Improved training efficiency and policy quality  

## 🧪 Experiments

The repository supports:
- Training RL agents for autonomous intersection control  
- Evaluating trained policies in simulated traffic scenarios  
- Testing under normal and adversarial conditions  

The extended study introduces adversarial settings where:
- Malicious vehicles inject incorrect position data  
- The system's robustness to such data integrity attacks is evaluated  

## 📊 Associated Publications

This codebase is used in the following works:

1. **Performance Improvement Study**  
   Árpád Huszák, Takahito Yoshizawa, Alireza Aghabagherloo, Dave Singelée, Bart Preneel  
   *On Performance Improvement of Reinforcement Learning for Collision Avoidance in Autonomous Intersections*  
   IEEE Access, 2025  

2. **Robustness Study (Extended Work)**  
   *Evaluating the Robustness of RL-based Autonomous Intersection Control to Data Integrity Attacks*  

The second paper extends the first by analyzing the susceptibility of the learned policies to adversarial manipulation and evaluating system resilience.

## ▶️ Usage

### 1. Install dependencies

pip install -r requirements.txt
