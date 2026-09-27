# 3D DNA-Histone Binding Simulation
A 3D interactive Python simulation modeling the binding dynamics and structural wrapping of DNA around histone proteins using Matplotlib and NumPy.

## Overview
This tool visualizes the physical interaction between a helical DNA strand and free-floating histone proteins in 3D space. As free histones undergo random movement, they bind to the DNA strand upon entering a specified collision radius. Upon binding, the DNA strand physically wraps around the bound histone core.

The application offers two distinct modes:

Full Interactive Mode: Includes real-time animation control buttons (Start, Pause, Reset) and dynamic sliders to adjust movement speed and binding radii on the fly.

Quick Animation Mode: A lightweight, non-interactive visualizer for rapid testing and demonstration.

## Features
3D Helical DNA Rendering: Generates a dynamic 3D helical backbone structure.

Random Brownian Motion: Unbound (free) histones move continuously through 3D space with wall-bouncing physics.

Proximity-Based Binding: Detects collisions between DNA and histones based on a customizable Euclidean distance threshold.

Histone Wrapping Effect: Dynamically deforms the DNA helix around bound histones according to target rotation parameters.

### Interactive UI Controls: Built-in Matplotlib widgets including:

Start / Pause / Reset simulation state controls.

Speed Slider: Modify histone movement speed during runtime.

Radius Slider: Adjust binding/collision sensitivity dynamically.
