# Pacman AI (LINQS-Pacai)

A modernized and enhanced version of the classic Pacman educational project. This repository represents a complete overhaul designed for modern Python environments and robust Artificial Intelligence experimentation.

## Project Overview

The `pacai` project serves as a comprehensive framework for teaching and testing AI concepts, including search, multi-agent systems, and reinforcement learning. Unlike the original version, this project is fully organized into modular Python packages for easier extension and integration into modern development workflows.

### Core Features

* **Modern Python Core:** Fully updated to support **Python >= 3.8**.
* **Enhanced Visualization:**
* **New Graphics Engine:** Features a flexible UI framework that replaces original legacy systems.
* **GIF Generation:** Capabilities to record any Pacman or Capture the Flag game and save it directly as an animated GIF.
* **Custom Sprites:** Support for custom spritesheet paths to change the visual theme of the game.


* **Professional Tooling:**
* **Integrated Logging:** Built-in support for detailed debug output or clean console sessions.
* **Automated Testing:** Includes a dedicated test runner and a suite of unit tests to ensure agent reliability.


* **Flexible Simulation:**
* **Headless Mode:** Run simulations without graphics (`--null-graphics`) for rapid training and data collection.
* **Text Mode:** View game state transitions in a lightweight text-only format (`--text-graphics`).
* **Game Replays:** Record game moves to files and replay them later to analyze specific agent decisions.



## Project Structure

* `pacai.bin`: The entry points for all executables, including Pacman, Capture, Gridworld, and Crawler.
* `pacai.core`: The fundamental game engine, including state management, physics, and layout loading.
* `pacai.student`: The primary workspace for AI development, such as search agents and Q-learning implementations.
* `pacai.ui`: Management for various display frontends including GUI, Text, and Null interfaces.
* `pacai.util`: Shared utilities for logging, data structures (Priority Queues, Stacks, Queues), and probability.

## Getting Started

### Prerequisites

* **Python:** 3.8 or higher.
* **Dependencies:** Requirements include `Pillow`, `packaging`, and `setuptools`.
* **Graphics:** `Tk` must be installed on your operating system for GUI support.

### Installation

Install the necessary Python dependencies from the root of the repository:

```bash
pip3 install --user -r requirements.txt

```

### Running the Project

Launch the standard Pacman game from the root directory:

```bash
python3 -m pacai.bin.pacman

```

### Advanced Commands

**Generate a GIF of a game:**

```bash
python3 -m pacai.bin.pacman --gif my_game.gif

```

**Run training with suppressed output:**

```bash
python3 -m pacai.bin.pacman --null-graphics --num-training 100

```

**Run the automated test suite:**

```bash
python3 run_tests.py

```
