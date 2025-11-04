import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from matplotlib.widgets import Button, Slider
import matplotlib.animation as animation

class InteractiveHistoneDNAModel:
    def __init__(self):
        self.setup_parameters()
        self.setup_simulation()
        
    def setup_parameters(self):
        """Get parameters from user input"""
        print("DNA-Histone Binding Simulation")
        print("=" * 50)
        
        self.dna_length = int(input("Enter DNA length (50-500) [100]: ") or 100)
        self.num_histones = int(input("Enter number of histones [5]: ") or 5)
        self.rotations_per_histone = float(input("Enter rotations per histone (1.0-3.0) [1.7]: ") or 1.7)
        self.simulation_steps = int(input("Enter simulation steps [200]: ") or 200)
        self.histone_speed = float(input("Enter histone movement speed (0.1-2.0) [0.5]: ") or 0.5)
        self.binding_radius = float(input("Enter binding collision radius (0.5-3.0) [1.5]: ") or 1.5)
        
        # Initialize histones
        self.histones = []
        self.bound_histones = []
        self.setup_initial_histones()
        
    def setup_initial_histones(self):
        """Place histones randomly in 3D space around DNA"""
        for i in range(self.num_histones):
            # Start histones at random positions around the DNA
            x = random.uniform(-15, 15)
            y = random.uniform(-15, 15)
            z = random.uniform(-5, self.dna_length * 0.2 + 5)
            self.histones.append({
                'position': np.array([x, y, z]),
                'bound': False,
                'bound_position': None,
                'id': i,
                'color': 'orange'
            })
    
    def setup_simulation(self):
        """Initialize DNA structure and plot"""
        self.dna_points = self._generate_dna_backbone()
        self.fig = plt.figure(figsize=(15, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize plot elements
        self.dna_line, = self.ax.plot([], [], [], 'b-', alpha=0.6, linewidth=3, label='DNA')
        self.backbone_line, = self.ax.plot([], [], [], 'gray', alpha=0.3, linewidth=1, label='DNA Backbone')
        self.free_histones_scatter = self.ax.scatter([], [], [], c='orange', s=100, alpha=0.8, label='Free Histone')
        self.bound_histones_scatter = self.ax.scatter([], [], [], c='red', s=200, alpha=0.9, label='Bound Histone')
        
        self.setup_controls()
        self.setup_plot_limits()
        
    def setup_plot_limits(self):
        """Set consistent 3D plot limits"""
        max_range = max(self.dna_length * 0.1, 25)
        self.ax.set_xlim([-max_range, max_range])
        self.ax.set_ylim([-max_range, max_range])
        self.ax.set_zlim([-10, self.dna_length * 0.2 + 10])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.legend(loc='upper left')
        
    def setup_controls(self):
        """Add interactive controls to the plot"""
        # Add control buttons
        ax_start = plt.axes([0.7, 0.02, 0.1, 0.04])
        ax_reset = plt.axes([0.81, 0.02, 0.1, 0.04])
        ax_pause = plt.axes([0.59, 0.02, 0.1, 0.04])
        
        self.btn_start = Button(ax_start, 'Start')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_pause = Button(ax_pause, 'Pause')
        
        self.btn_start.on_clicked(self.start_animation)
        self.btn_reset.on_clicked(self.reset_animation)
        self.btn_pause.on_clicked(self.pause_animation)
        
        # Add sliders for real-time adjustment
        ax_speed = plt.axes([0.1, 0.02, 0.35, 0.02])
        ax_radius = plt.axes([0.1, 0.05, 0.35, 0.02])
        
        self.slider_speed = Slider(ax_speed, 'Speed', 0.1, 2.0, valinit=self.histone_speed)
        self.slider_radius = Slider(ax_radius, 'Radius', 0.5, 3.0, valinit=self.binding_radius)
        
        self.slider_speed.on_changed(self.update_speed)
        self.slider_radius.on_changed(self.update_radius)
        
        self.animation_running = False
        self.animation_paused = False
        self.current_step = 0
        
    def update_speed(self, val):
        self.histone_speed = val
        
    def update_radius(self, val):
        self.binding_radius = val

    def _generate_dna_backbone(self):
        """Generate a simple helical DNA backbone"""
        points = []
        for i in range(self.dna_length):
            t = i * 0.3
            x = np.cos(t) * 2
            y = np.sin(t) * 2
            z = i * 0.15
            points.append([x, y, z])
        return np.array(points)

    def _find_nearest_dna_point(self, histone_pos):
        """Find the nearest DNA point to a histone"""
        distances = np.linalg.norm(self.dna_points - histone_pos, axis=1)
        nearest_idx = np.argmin(distances)
        return nearest_idx, distances[nearest_idx]

    def move_histones(self):
        """Move unbound histones randomly - VISIBLE MOVEMENT"""
        for histone in self.histones:
            if not histone['bound']:
                # More pronounced random movement
                move = np.random.uniform(-1, 1, 3) * self.histone_speed * 2  # Increased movement
                histone['position'] += move
                
                # Keep histones in visible area with some bouncing effect
                for i in range(3):
                    if abs(histone['position'][i]) > 20:
                        histone['position'][i] = np.sign(histone['position'][i]) * 20
                        # Simple bounce effect
                        move[i] *= -0.5

    def check_collisions(self):
        """Check for collisions between histones and DNA"""
        for histone in self.histones:
            if not histone['bound']:
                nearest_idx, distance = self._find_nearest_dna_point(histone['position'])
                
                if distance < self.binding_radius:
                    # Bind histone to DNA
                    histone['bound'] = True
                    histone['bound_position'] = nearest_idx
                    histone['color'] = 'red'
                    self.bound_histones.append(histone)
                    print(f"Step {self.current_step}: Histone {histone['id']} bound at position {nearest_idx}")

    def _apply_histone_wrapping(self, points, histone_pos, rotations=1.7):
        """Apply histone wrapping effect around a specific position"""
        wrapped_points = points.copy()
        wrap_radius = 2.0
        wrap_length = 20
        
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
        """Apply histone wrapping to all bound histones"""
        wrapped_points = self.dna_points.copy()
        
        for histone in self.bound_histones:
            if histone['bound_position'] is not None:
                wrapped_points = self._apply_histone_wrapping(
                    wrapped_points, histone['bound_position'], self.rotations_per_histone
                )
            
        return wrapped_points

    def update_plot(self, frame):
        """Update function for animation"""
        if not self.animation_running or self.animation_paused:
            return self.dna_line, self.backbone_line, self.free_histones_scatter, self.bound_histones_scatter
            
        self.current_step = frame
        
        # Update simulation state
        self.move_histones()
        self.check_collisions()
        
        # Get current DNA state
        wrapped_dna = self.get_wrapped_dna()
        
        # Update DNA plot
        self.dna_line.set_data(wrapped_dna[:, 0], wrapped_dna[:, 1])
        self.dna_line.set_3d_properties(wrapped_dna[:, 2])
        
        self.backbone_line.set_data(self.dna_points[:, 0], self.dna_points[:, 1])
        self.backbone_line.set_3d_properties(self.dna_points[:, 2])
        
        # Separate free and bound histones for plotting
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
        
        # Update title
        bound_count = len(self.bound_histones)
        self.ax.set_title(
            f'DNA-Histone Binding Simulation\n'
            f'Step: {frame}/{self.simulation_steps} | '
            f'Bound: {bound_count}/{self.num_histones} | '
            f'Free: {self.num_histones - bound_count} | '
            f'Speed: {self.histone_speed:.1f} | '
            f'Radius: {self.binding_radius:.1f}'
        )
        
        return self.dna_line, self.backbone_line, self.free_histones_scatter, self.bound_histones_scatter

    def start_animation(self, event=None):
        """Start the animation"""
        if not hasattr(self, 'anim'):
            # Create animation
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
        if self.animation_paused:
            print("Simulation paused")
        else:
            print("Simulation resumed")

    def reset_animation(self, event=None):
        """Reset animation to initial state"""
        self.animation_running = False
        self.animation_paused = False
        self.current_step = 0
        
        # Reinitialize histones
        self.histones = []
        self.bound_histones = []
        self.setup_initial_histones()
        
        # Reset plot
        self.update_plot(0)
        self.fig.canvas.draw_idle()
        print("Simulation reset!")

    def show_initial_state(self):
        """Show the initial state before animation"""
        self.update_plot(0)
        plt.show()

    def print_initial_stats(self):
        """Print initial simulation parameters"""
        print(f"\nSimulation Parameters:")
        print(f"DNA Length: {self.dna_length} base pairs")
        print(f"Number of Histones: {self.num_histones}")
        print(f"Rotations per Histone: {self.rotations_per_histone}")
        print(f"Simulation Steps: {self.simulation_steps}")
        print(f"Histone Speed: {self.histone_speed}")
        print(f"Binding Radius: {self.binding_radius}")
        print(f"\nClick 'Start' to begin simulation!")

# Simple version for quick testing
class QuickHistoneSimulation:
    def __init__(self, dna_length=80, num_histones=6, steps=150):
        self.dna_length = dna_length
        self.num_histones = num_histones
        self.steps = steps
        
        self.dna_points = self._generate_dna_backbone()
        self.histones = []
        self.bound_histones = []
        
        # Initialize free histones
        for i in range(self.num_histones):
            self.histones.append({
                'pos': np.random.uniform(-15, 15, 3),
                'bound': False,
                'bound_pos': None,
                'id': i
            })
    
    def _generate_dna_backbone(self):
        points = []
        for i in range(self.dna_length):
            t = i * 0.3
            x = np.cos(t) * 2
            y = np.sin(t) * 2
            z = i * 0.1
            points.append([x, y, z])
        return np.array(points)
    
    def run(self):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        def animate(frame):
            ax.clear()
            
            # Move unbound histones
            for histone in self.histones:
                if not histone['bound']:
                    # Visible random movement
                    histone['pos'] += np.random.uniform(-1, 1, 3) * 0.8
                    
                    # Check for binding
                    distances = np.linalg.norm(self.dna_points - histone['pos'], axis=1)
                    if np.min(distances) < 2.0:
                        histone['bound'] = True
                        histone['bound_pos'] = np.argmin(distances)
                        self.bound_histones.append(histone)
                        print(f"Step {frame}: Histone {histone['id']} bound!")
            
            # Plot DNA
            ax.plot(self.dna_points[:, 0], self.dna_points[:, 1], self.dna_points[:, 2], 
                   'b-', alpha=0.7, linewidth=2, label='DNA')
            
            # Plot histones
            free_x, free_y, free_z = [], [], []
            bound_x, bound_y, bound_z = [], [], []
            
            for histone in self.histones:
                if histone['bound']:
                    pos = self.dna_points[histone['bound_pos']]
                    bound_x.append(pos[0])
                    bound_y.append(pos[1])
                    bound_z.append(pos[2])
                else:
                    free_x.append(histone['pos'][0])
                    free_y.append(histone['pos'][1])
                    free_z.append(histone['pos'][2])
            
            if free_x:
                ax.scatter(free_x, free_y, free_z, c='orange', s=100, label='Free Histones')
            if bound_x:
                ax.scatter(bound_x, bound_y, bound_z, c='red', s=150, label='Bound Histones')
            
            ax.set_xlim(-20, 20)
            ax.set_ylim(-20, 20)
            ax.set_zlim(-5, self.dna_length * 0.1 + 5)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            ax.set_title(f'Step {frame+1}/{self.steps} - Bound: {len(self.bound_histones)}/{self.num_histones}')
        
        anim = animation.FuncAnimation(fig, animate, frames=self.steps, interval=100, repeat=False)
        plt.show()

# Main execution
if __name__ == "__main__":
    print("Choose simulation mode:")
    print("1. Full Interactive (with controls - RECOMMENDED)")
    print("2. Quick Animation (simple version)")
    
    choice = input("Enter choice [1]: ") or "1"
    
    if choice == "1":
        model = InteractiveHistoneDNAModel()
        model.print_initial_stats()
        model.show_initial_state()
    else:
        # Quick simulation with default parameters
        sim = QuickHistoneSimulation(
            dna_length=80,
            num_histones=6,
            steps=150
        )
        sim.run()
