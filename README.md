# QA-Attack

**QA-Attack: Exposing Vulnerabilities in Question Answering Models Using Hybrid Ranking Fusion**

## Overview

QA-Attack is a novel adversarial attack framework designed specifically to target and expose vulnerabilities in Question Answering (QA) models. By leveraging a Hybrid Ranking Fusion (HRF) algorithm that combines attention-based and removal-based ranking methods, QA-Attack identifies the most critical tokens within context passages and questions, allowing for precise and effective adversarial examples that mislead QA models while maintaining semantic coherence.

## Features

- **Hybrid Ranking Fusion (HRF) Algorithm**: Combines Attention-based Ranking (ABR) and Removal-based Ranking (RBR) to identify the most vulnerable tokens in the input.
- **Versatile Attack**: Targets both "Yes/No" questions and "Wh-questions", demonstrating the robustness of the attack across different question types.
- **Black-box Approach**: No need for direct access to the target model’s architecture or parameters, making the attack applicable to a wide range of QA models.

## Installation

### Prerequisites

- Python 3.6 or higher
- Required Python packages (listed in `requirements.txt`)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/UTSJiyaoLi/QA-Attack.git
   cd QA-Attack
