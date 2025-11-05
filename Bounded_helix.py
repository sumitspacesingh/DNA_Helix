import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib.widgets import Button, Slider, RadioButtons, TextBox
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from matplotlib import patches

class AdvancedHistoneDNAModel:
    def __init__(self):
        self.setup_parameters()
        self.setup_simulation()
        
    def setup_parameters(self):
        """Get comprehensive parameters from user input"""
        print("Advanced DNA-Histone Binding Simulation")
        print("=" * 50)
        
        # Boundary parameters
        print("\n--- Simulation Boundary ---")
        self.boundary_x = float(input("Boundary X size (10-100) [30]: ") or 30)
        self.boundary_y = float(input("Boundary Y size (10-100) [30]: ") or 30)
        self.boundary_z = float(input("Boundary Z size (10-100) [40]: ") or 40)
        
        # DNA parameters
        print("\n--- DNA Configuration ---")
        self.dna_length = int(input("DNA length (50-500) [100]: ") or 100)
        
        print("DNA Structure Options:")
        print("1. Straight Line")
        print("2. Spiral Helix")
        print("3. Random Walk")
        print("4. Circular")
        dna_choice = input("Choose DNA structure [2]: ") or "2"
        
        dna_options = {
            "1": "straight",
            "2": "spiral", 
            "3": "random",
            "4": "circular"
        }
        self.dna_structure = dna_options.get(dna_choice, "spiral")
        
        # DNA movement
        move_choice = input("Should DNA move randomly? (y/n) [n]: ") or "n"
        self.dna_movement = move_choice.lower() in ['y', 'yes', '1']
        self.dna_speed = 0.1 if self.dna_movement else 0.0
        
        # Histone parameters
        print("\n--- Histone Parameters ---")
        self.num_histones = int(input("Number of histones [8]: ") or 8)
        self.rotations_per_histone = float(input("Rotations per histone (1.0-3.0) [1.7]: ") or 1.7)
        self.histone_speed = float(input("Histone movement speed (0.1-2.0) [0.8]: ") or 0.8)
        self.binding_radius = float(input("Binding collision radius (0.5-3.0) [1.8]: ") or 1.8)
        
        # Simulation parameters
        print("\n--- Simulation Parameters ---")
        self.simulation_steps = int(input("Simulation steps [300]: ") or 300)
        
        # Initialize components
        self.histones = []
        self.bound_histones = []
        self.setup_initial_histones()
        self.setup_dna()
        
    def setup_dna(self):
        """Generate DNA based on user choice"""
        if self.dna_structure == "straight":
            self.dna_points = self._generate_straight_dna()
        elif self.dna_structure == "spiral":
            self.dna_points = self._generate_spiral_dna()
        elif self.dna_structure == "random":
            self.dna_points = self._generate_random_dna()
        elif self.dna_structure == "circular":
            self.dna_points = self._generate_circular_dna()
        
        # Center DNA in boundary
        self.center_dna_in_boundary()
        
    def _generate_straight_dna(self):
        """Generate straight DNA"""
        points = []
        for i in range(self.dna_length):
            x = 0
            y = 0
            z = i * (self.boundary_z * 0.8 / self.dna_length)
            points.append([x, y, z])
        return np.array(points)
    
    def _generate_spiral_dna(self):
        """Generate spiral DNA"""
        points = []
        max_radius = min(self.boundary_x, self.boundary_y) * 0.3
        for i in range(self.dna_length):
            t = i * 0.3
            x = np.cos(t) * max_radius
            y = np.sin(t) * max_radius
            z = i * (self.boundary_z * 0.8 / self.dna_length)
            points.append([x, y, z])
        return np.array(points)
    
    def _generate_random_dna(self):
        """Generate random walk DNA"""
        points = []
        current_pos = np.array([0.0, 0.0, 0.0])
        
        for i in range(self.dna_length):
            points.append(current_pos.copy())
            # Random step with constraints
            step = np.random.uniform(-0.5, 0.5, 3)
            current_pos += step
            
            # Keep within bounds
            current_pos[0] = np.clip(current_pos[0], -self.boundary_x*0.4, self.boundary_x*0.4)
            current_pos[1] = np.clip(current_pos[1], -self.boundary_y*0.4, self.boundary_y*0.4)
            current_pos[2] = np.clip(current_pos[2], 0, self.boundary_z*0.8)
            
        return np.array(points)
    
    def _generate_circular_dna(self):
        """Generate circular/toroidal DNA"""
        points = []
        radius = min(self.boundary_x, self.boundary_y) * 0.3
        for i in range(self.dna_length):
            angle = i * 2 * np.pi / self.dna_length
            x = np.cos(angle) * radius
            y = np.sin(angle) * radius
            z = self.boundary_z * 0.4  # Constant height
            points.append([x, y, z])
        return np.array(points)
    
    def center_dna_in_boundary(self):
        """Center DNA within the boundary"""
        center_x = 0
        center_y = 0
        center_z = self.boundary_z * 0.1
        
        self.dna_points += np.array([center_x, center_y, center_z])
    
    def setup_initial_histones(self):
        """Place histones randomly within bounded area"""
        for i in range(self.num_histones):
            x = random.uniform(-self.boundary_x/2, self.boundary_x/2)
            y = random.uniform(-self.boundary_y/2, self.boundary_y/2)
            z = random.uniform(0, self.boundary_z)
            
            self.histones.append({
                'position': np.array([x, y, z]),
                'bound': False,
                'bound_position': None,
                'id': i,
                'color': 'orange'
            })
    
    def setup_simulation(self):
        """Initialize simulation visualization"""
        self.fig = plt.figure(figsize=(16, 12))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize plot elements
        self.dna_line, = self.ax.plot([], [], [], 'b-', alpha=0.7, linewidth=3, label='DNA')
        self.free_histones_scatter = self.ax.scatter([], [], [], c='orange', s=120, alpha=0.8, label='Free Histones')
        self.bound_histones_scatter = self.ax.scatter([], [], [], c='red', s=180, alpha=0.9, label='Bound Histones')
        
        self.setup_controls()
        self.setup_boundary_visualization()
        self.setup_plot_limits()
        
    def setup_boundary_visualization(self):
        """Add visual boundary box"""
        # Create transparent boundary box
        x = [-self.boundary_x/2, self.boundary_x/2]
        y = [-self.boundary_y/2, self.boundary_y/2]
        z = [0, self.boundary_z]
        
        # Plot bounding box edges
        for i in range(2):
            for j in range(2):
                self.ax.plot([x[0], x[1]], [y[i], y[i]], [z[j], z[j]], 'k--', alpha=0.3, linewidth=0.5)
                self.ax.plot([x[i], x[i]], [y[0], y[1]], [z[j], z[j]], 'k--', alpha=0.3, linewidth=0.5)
                self.ax.plot([x[i], x[i]], [y[j], y[j]], [z[0], z[1]], 'k--', alpha=0.3, linewidth=0.5)
        
    def setup_plot_limits(self):
        """Set 3D plot limits based on boundary"""
        self.ax.set_xlim([-self.boundary_x/2, self.boundary_x/2])
        self.ax.set_ylim([-self.boundary_y/2, self.boundary_y/2])
        self.ax.set_zlim([0, self.boundary_z])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend(loc='upper left')
        
    def setup_controls(self):
        """Add comprehensive interactive controls"""
        # Control buttons
        ax_start = plt.axes([0.75, 0.02, 0.08, 0.04])
        ax_reset = plt.axes([0.84, 0.02, 0.08, 0.04])
        ax_pause = plt.axes([0.66, 0.02, 0.08, 0.04])
        
        self.btn_start = Button(ax_start, 'Start')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_pause = Button(ax_pause, 'Pause')
        
        self.btn_start.on_clicked(self.start_animation)
        self.btn_reset.on_clicked(self.reset_animation)
        self.btn_pause.on_clicked(self.pause_animation)
        
        # Sliders
        ax_speed = plt.axes([0.15, 0.02, 0.2, 0.02])
        ax_radius = plt.axes([0.15, 0.05, 0.2, 0.02])
        ax_dna_speed = plt.axes([0.15, 0.08, 0.2, 0.02])
        
        self.slider_speed = Slider(ax_speed, 'Histone Speed', 0.1, 2.0, valinit=self.histone_speed)
        self.slider_radius = Slider(ax_radius, 'Bind Radius', 0.5, 3.0, valinit=self.binding_radius)
        self.slider_dna_speed = Slider(ax_dna_speed, 'DNA Speed', 0.0, 0.5, valinit=self.dna_speed)
        
        self.slider_speed.on_changed(self.update_speed)
        self.slider_radius.on_changed(self.update_radius)
        self.slider_dna_speed.on_changed(self.update_dna_speed)
        
        # DNA structure radio buttons
        ax_dna_type = plt.axes([0.02, 0.7, 0.12, 0.2])
        self.radio_dna = RadioButtons(ax_dna_type, 
                                    ['Straight', 'Spiral', 'Random', 'Circular'],
                                    active=['straight', 'spiral', 'random', 'circular'].index(self.dna_structure))
        self.radio_dna.on_clicked(self.change_dna_structure)
        
        self.animation_running = False
        self.animation_paused = False
        self.current_step = 0
        
    def update_speed(self, val):
        self.histone_speed = val
        
    def update_radius(self, val):
        self.binding_radius = val
        
    def update_dna_speed(self, val):
        self.dna_speed = val
        self.dna_movement = val > 0
        
    def change_dna_structure(self, label):
        """Change DNA structure dynamically"""
        structure_map = {
            'Straight': 'straight',
            'Spiral': 'spiral', 
            'Random': 'random',
            'Circular': 'circular'
        }
        self.dna_structure = structure_map[label]
        self.setup_dna()
        if self.animation_running and not self.animation_paused:
            self.update_plot(self.current_step)
        
    def enforce_boundary_constraints(self, position):
        """Keep position within boundary with bounce effect"""
        new_pos = position.copy()
        
        # X boundary
        if new_pos[0] < -self.boundary_x/2:
            new_pos[0] = -self.boundary_x/2
        elif new_pos[0] > self.boundary_x/2:
            new_pos[0] = self.boundary_x/2
            
        # Y boundary  
        if new_pos[1] < -self.boundary_y/2:
            new_pos[1] = -self.boundary_y/2
        elif new_pos[1] > self.boundary_y/2:
            new_pos[1] = self.boundary_y/2
            
        # Z boundary
        if new_pos[2] < 0:
            new_pos[2] = 0
        elif new_pos[2] > self.boundary_z:
            new_pos[2] = self.boundary_z
            
        return new_pos

    def move_dna(self):
        """Move DNA randomly if enabled"""
        if self.dna_movement and self.dna_speed > 0:
            # Move entire DNA
            move = np.random.uniform(-1, 1, 3) * self.dna_speed
            new_dna_points = self.dna_points + move
            
            # Check if new positions are within bounds
            min_pos = np.min(new_dna_points, axis=0)
            max_pos = np.max(new_dna_points, axis=0)
            
            # Only apply move if DNA stays within bounds
            if (min_pos[0] >= -self.boundary_x/2 and max_pos[0] <= self.boundary_x/2 and
                min_pos[1] >= -self.boundary_y/2 and max_pos[1] <= self.boundary_y/2 and
                min_pos[2] >= 0 and max_pos[2] <= self.boundary_z):
                self.dna_points = new_dna_points

    def move_histones(self):
        """Move unbound histones with boundary constraints"""
        for histone in self.histones:
            if not histone['bound']:
                # Random movement
                move = np.random.uniform(-1, 1, 3) * self.histone_speed
                new_position = histone['position'] + move
                
                # Apply boundary constraints
                histone['position'] = self.enforce_boundary_constraints(new_position)

    def check_collisions(self):
        """Check for histone-DNA collisions"""
        for histone in self.histones:
            if not histone['bound']:
                nearest_idx, distance = self._find_nearest_dna_point(histone['position'])
                
                if distance < self.binding_radius:
                    # Bind histone to DNA
                    histone['bound'] = True
                    histone['bound_position'] = nearest_idx
                    histone['color'] = 'red'
                    self.bound_histones.append(histone)
                    print(f"Step {self.current_step}: Histone {histone['id']} bound!")

    def _find_nearest_dna_point(self, histone_pos):
        """Find nearest DNA point to histone"""
        distances = np.linalg.norm(self.dna_points - histone_pos, axis=1)
        nearest_idx = np.argmin(distances)
        return nearest_idx, distances[nearest_idx]

    def _apply_histone_wrapping(self, points, histone_pos, rotations=1.7):
        """Apply DNA wrapping around histones"""
        wrapped_points = points.copy()
        wrap_radius = 1.5
        wrap_length = 15
        
        start_idx = max(0, histone_pos - wrap_length//2)
        end_idx = min(len(points), histone_pos + wrap_length//2)
        
        for i in range(start_idx, end_idx):
            rel_pos = (i - start_idx) / (end_idx - start_idx)
            angle = rel_pos * 2 * np.pi * rotations
            
            center = points[histone_pos]
            
            x = center[0] + wrap_radius * np.cos(angle)
            y = center[1] + wrap_radius * np.sin(angle)
            z = points[i][2]
            
            wrapped_points[i] = [x, y, z]
            
        return wrapped_points

    def get_wrapped_dna(self):
        """Get DNA with histone wrapping applied"""
        wrapped_points = self.dna_points.copy()
        
        for histone in self.bound_histones:
            if histone['bound_position'] is not None:
                wrapped_points = self._apply_histone_wrapping(
                    wrapped_points, histone['bound_position'], self.rotations_per_histone
                )
            
        return wrapped_points

    def update_plot(self, frame):
        """Update animation frame"""
        if not self.animation_running or self.animation_paused:
            return self.dna_line, self.free_histones_scatter, self.bound_histones_scatter
            
        self.current_step = frame
        
        # Update positions
        self.move_dna()
        self.move_histones()
        self.check_collisions()
        
        # Get wrapped DNA
        wrapped_dna = self.get_wrapped_dna()
        
        # Update DNA plot
        self.dna_line.set_data(wrapped_dna[:, 0], wrapped_dna[:, 1])
        self.dna_line.set_3d_properties(wrapped_dna[:, 2])
        
        # Update histones
        free_positions = []
        bound_positions = []
        
        for histone in self.histones:
            if histone['bound'] and histone['bound_position'] is not None:
                bound_pos = wrapped_dna[histone['bound_position']]
                bound_positions.append(bound_pos)
            elif not histone['bound']:
                free_positions.append(histone['position'])
        
        # Update scatter plots
        if free_positions:
            free_positions = np.array(free_positions)
            self.free_histones_scatter._offsets3d = (free_positions[:, 0], free_positions[:, 1], free_positions[:, 2])
        else:
            self.free_histones_scatter._offsets3d = ([], [], [])
            
        if bound_positions:
            bound_positions = np.array(bound_positions)
            self.bound_histones_scatter._offsets3d = (bound_positions[:, 0], bound_positions[:, 1], bound_positions[:, 2])
        else:
            self.bound_histones_scatter._offsets3d = ([], [], [])
        
        # Update title with comprehensive info
        bound_count = len(self.bound_histones)
        dna_movement_status = "ON" if self.dna_movement else "OFF"
        
        self.ax.set_title(
            f'Advanced DNA-Histone Simulation\n'
            f'Step: {frame}/{self.simulation_steps} | '
            f'Bound: {bound_count}/{self.num_histones} | '
            f'DNA: {self.dna_structure.title()} | '
            f'DNA Move: {dna_movement_status}\n'
            f'Boundary: {self.boundary_x}x{self.boundary_y}x{self.boundary_z}'
        )
        
        return self.dna_line, self.free_histones_scatter, self.bound_histones_scatter

    def start_animation(self, event=None):
        """Start animation"""
        if not hasattr(self, 'anim'):
            self.anim = animation.FuncAnimation(
                self.fig, self.update_plot, frames=self.simulation_steps,
                interval=50, blit=False, repeat=False
            )
        
        self.animation_running = True
        self.animation_paused = False
        print("Simulation started!")

    def pause_animation(self, event=None):
        """Pause/resume animation"""
        self.animation_paused = not self.animation_paused
        status = "paused" if self.animation_paused else "resumed"
        print(f"Simulation {status}")

    def reset_animation(self, event=None):
        """Reset simulation"""
        self.animation_running = False
        self.animation_paused = False
        self.current_step = 0
        
        # Reinitialize
        self.histones = []
        self.bound_histones = []
        self.setup_initial_histones()
        self.setup_dna()
        
        self.update_plot(0)
        self.fig.canvas.draw_idle()
        print("Simulation reset!")

    def show_initial_state(self):
        """Show initial setup"""
        self.update_plot(0)
        plt.show()

    def print_initial_stats(self):
        """Print simulation parameters"""
        print(f"\n=== Simulation Configuration ===")
        print(f"Boundary: {self.boundary_x} x {self.boundary_y} x {self.boundary_z}")
        print(f"DNA: {self.dna_structure} structure, {self.dna_length} points")
        print(f"DNA Movement: {'ON' if self.dna_movement else 'OFF'}")
        print(f"Histones: {self.num_histones}, Speed: {self.histone_speed}")
        print(f"Binding Radius: {self.binding_radius}")
        print(f"Simulation Steps: {self.simulation_steps}")
        print(f"\nControls:")
        print(f"- Use sliders to adjust parameters in real-time")
        print(f"- Change DNA structure with radio buttons")
        print(f"- Click Start/Pause/Reset to control simulation")
        print(f"\nClose the plot window to exit.")

# Main execution
if __name__ == "__main__":
    print("Advanced DNA-Histone Binding Simulation")
    print("=" * 50)
    
    model = AdvancedHistoneDNAModel()
    model.print_initial_stats()
    model.show_initial_state()
