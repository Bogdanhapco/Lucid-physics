import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
import matplotlib.animation as animation
from io import BytesIO
import time

# Page configuration
st.set_page_config(
    page_title="Interactive Physics Simulator",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better appearance
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding: 1rem 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🌌 Interactive Physics Simulator</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Explore the beauty of physics through accurate simulations</p>', unsafe_allow_html=True)

# Sidebar for simulation selection
st.sidebar.title("⚙️ Simulation Controls")
simulation_type = st.sidebar.selectbox(
    "Choose a Physics Simulation",
    ["Double Pendulum", "N-Body Gravity", "Wave Interference", "Collision Physics", "Fluid Dynamics"]
)

st.sidebar.markdown("---")

# ==================== DOUBLE PENDULUM ====================
def simulate_double_pendulum(L1, L2, m1, m2, theta1_0, theta2_0, steps, dt):
    """Simulate a chaotic double pendulum system"""
    g = 9.81
    
    # State: [theta1, omega1, theta2, omega2]
    state = np.array([theta1_0, 0, theta2_0, 0])
    
    positions1 = []
    positions2 = []
    
    for _ in range(steps):
        theta1, omega1, theta2, omega2 = state
        
        # Equations of motion for double pendulum
        delta = theta2 - theta1
        den1 = (m1 + m2) * L1 - m2 * L1 * np.cos(delta) * np.cos(delta)
        den2 = (L2 / L1) * den1
        
        # Angular accelerations
        alpha1 = (m2 * L1 * omega1**2 * np.sin(delta) * np.cos(delta) +
                  m2 * g * np.sin(theta2) * np.cos(delta) +
                  m2 * L2 * omega2**2 * np.sin(delta) -
                  (m1 + m2) * g * np.sin(theta1)) / den1
        
        alpha2 = (-m2 * L2 * omega2**2 * np.sin(delta) * np.cos(delta) +
                  (m1 + m2) * (g * np.sin(theta1) * np.cos(delta) -
                               L1 * omega1**2 * np.sin(delta) -
                               g * np.sin(theta2))) / den2
        
        # Update state
        state[1] += alpha1 * dt
        state[0] += state[1] * dt
        state[3] += alpha2 * dt
        state[2] += state[3] * dt
        
        # Calculate positions
        x1 = L1 * np.sin(state[0])
        y1 = -L1 * np.cos(state[0])
        x2 = x1 + L2 * np.sin(state[2])
        y2 = y1 - L2 * np.cos(state[2])
        
        positions1.append([x1, y1])
        positions2.append([x2, y2])
    
    return np.array(positions1), np.array(positions2)

# ==================== N-BODY GRAVITY ====================
def simulate_nbody(bodies, steps, dt):
    """Simulate N-body gravitational interactions"""
    G = 1.0  # Gravitational constant (scaled)
    positions = [[] for _ in range(len(bodies))]
    
    for _ in range(steps):
        # Calculate forces
        for i, body1 in enumerate(bodies):
            force = np.array([0.0, 0.0])
            for j, body2 in enumerate(bodies):
                if i != j:
                    r = body2['pos'] - body1['pos']
                    dist = np.linalg.norm(r)
                    if dist > 0.1:  # Avoid singularities
                        force += G * body1['mass'] * body2['mass'] * r / (dist**3)
            
            # Update velocity and position
            body1['vel'] += force / body1['mass'] * dt
            body1['pos'] += body1['vel'] * dt
            positions[i].append(body1['pos'].copy())
    
    return [np.array(pos) for pos in positions]

# ==================== WAVE INTERFERENCE ====================
def simulate_wave_interference(sources, grid_size, wavelength, steps):
    """Simulate wave interference patterns"""
    x = np.linspace(-5, 5, grid_size)
    y = np.linspace(-5, 5, grid_size)
    X, Y = np.meshgrid(x, y)
    
    frames = []
    for t in range(steps):
        wave = np.zeros_like(X)
        for source in sources:
            sx, sy, amp, freq = source
            r = np.sqrt((X - sx)**2 + (Y - sy)**2)
            wave += amp * np.sin(2 * np.pi * (r / wavelength - freq * t / steps))
        frames.append(wave)
    
    return frames, X, Y

# ==================== COLLISION PHYSICS ====================
def simulate_collisions(particles, steps, dt, box_size):
    """Simulate elastic collisions between particles"""
    trajectories = [[] for _ in range(len(particles))]
    
    for _ in range(steps):
        # Update positions
        for p in particles:
            p['pos'] += p['vel'] * dt
            trajectories[particles.index(p)].append(p['pos'].copy())
        
        # Check wall collisions
        for p in particles:
            for i in range(2):
                if p['pos'][i] - p['radius'] < -box_size/2:
                    p['pos'][i] = -box_size/2 + p['radius']
                    p['vel'][i] *= -0.95  # Slight energy loss
                elif p['pos'][i] + p['radius'] > box_size/2:
                    p['pos'][i] = box_size/2 - p['radius']
                    p['vel'][i] *= -0.95
        
        # Check particle collisions
        for i, p1 in enumerate(particles):
            for p2 in particles[i+1:]:
                diff = p2['pos'] - p1['pos']
                dist = np.linalg.norm(diff)
                if dist < p1['radius'] + p2['radius']:
                    # Elastic collision
                    normal = diff / dist
                    rel_vel = p1['vel'] - p2['vel']
                    vel_along_normal = np.dot(rel_vel, normal)
                    
                    if vel_along_normal > 0:
                        continue
                    
                    # Conservation of momentum and energy
                    impulse = 2 * vel_along_normal / (p1['mass'] + p2['mass'])
                    p1['vel'] -= impulse * p2['mass'] * normal
                    p2['vel'] += impulse * p1['mass'] * normal
                    
                    # Separate particles
                    overlap = p1['radius'] + p2['radius'] - dist
                    p1['pos'] -= normal * overlap * 0.5
                    p2['pos'] += normal * overlap * 0.5
    
    return [np.array(traj) for traj in trajectories]

# ==================== FLUID DYNAMICS ====================
def simulate_fluid(grid_size, steps, viscosity=0.01):
    """Simulate fluid flow using simplified Navier-Stokes"""
    # Initialize velocity field
    u = np.zeros((grid_size, grid_size))
    v = np.zeros((grid_size, grid_size))
    
    # Add initial vortex
    x = np.linspace(-1, 1, grid_size)
    y = np.linspace(-1, 1, grid_size)
    X, Y = np.meshgrid(x, y)
    
    r = np.sqrt(X**2 + Y**2)
    theta = np.arctan2(Y, X)
    
    u = -np.sin(theta) * np.exp(-r**2) * 5
    v = np.cos(theta) * np.exp(-r**2) * 5
    
    frames_u = []
    frames_v = []
    
    for _ in range(steps):
        # Simple advection
        u_new = u.copy()
        v_new = v.copy()
        
        # Diffusion (simplified)
        u_new[1:-1, 1:-1] += viscosity * (
            u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4*u[1:-1, 1:-1]
        )
        v_new[1:-1, 1:-1] += viscosity * (
            v[2:, 1:-1] + v[:-2, 1:-1] + v[1:-1, 2:] + v[1:-1, :-2] - 4*v[1:-1, 1:-1]
        )
        
        u = u_new * 0.99  # Slight damping
        v = v_new * 0.99
        
        frames_u.append(u.copy())
        frames_v.append(v.copy())
    
    return frames_u, frames_v, X, Y

# ==================== SIMULATION EXECUTION ====================

if simulation_type == "Double Pendulum":
    st.header("🎭 Double Pendulum - Chaos in Motion")
    st.markdown("*Watch as tiny changes in initial conditions lead to wildly different trajectories - a hallmark of chaotic systems.*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        L1 = st.slider("Length 1 (m)", 0.5, 3.0, 1.0, 0.1)
        L2 = st.slider("Length 2 (m)", 0.5, 3.0, 1.0, 0.1)
        m1 = st.slider("Mass 1 (kg)", 0.5, 5.0, 1.0, 0.5)
        m2 = st.slider("Mass 2 (kg)", 0.5, 5.0, 1.0, 0.5)
        theta1 = st.slider("Initial Angle 1 (°)", 0, 180, 90, 5)
        theta2 = st.slider("Initial Angle 2 (°)", 0, 180, 90, 5)
        
        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Simulating chaos..."):
                theta1_rad = np.radians(theta1)
                theta2_rad = np.radians(theta2)
                
                pos1, pos2 = simulate_double_pendulum(L1, L2, m1, m2, theta1_rad, theta2_rad, 1000, 0.01)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.set_xlim(-L1-L2-0.5, L1+L2+0.5)
                ax.set_ylim(-L1-L2-0.5, L1+L2+0.5)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                
                # Plot trajectories
                ax.plot(pos2[:, 0], pos2[:, 1], 'cyan', alpha=0.3, linewidth=0.5, label='Trajectory')
                
                # Plot current state (last few frames)
                n_frames = min(50, len(pos1))
                for i in range(-n_frames, 0):
                    alpha = (i + n_frames) / n_frames * 0.8
                    ax.plot([0, pos1[i, 0]], [0, pos1[i, 1]], 'white', alpha=alpha*0.3, linewidth=1)
                    ax.plot([pos1[i, 0], pos2[i, 0]], [pos1[i, 1], pos2[i, 1]], 'white', alpha=alpha*0.3, linewidth=1)
                
                # Final position
                ax.plot([0, pos1[-1, 0]], [0, pos1[-1, 1]], 'yellow', linewidth=3)
                ax.plot([pos1[-1, 0], pos2[-1, 0]], [pos1[-1, 1], pos2[-1, 1]], 'orange', linewidth=3)
                ax.plot(0, 0, 'wo', markersize=12)
                ax.plot(pos1[-1, 0], pos1[-1, 1], 'yo', markersize=m1*5)
                ax.plot(pos2[-1, 0], pos2[-1, 1], 'ro', markersize=m2*5)
                
                ax.legend(facecolor='#262730', edgecolor='white')
                ax.set_title('Double Pendulum Trajectory', color='white', fontsize=16, pad=20)
                ax.tick_params(colors='white')
                
                with col2:
                    st.pyplot(fig)
                    plt.close()

elif simulation_type == "N-Body Gravity":
    st.header("🌍 N-Body Gravitational System")
    st.markdown("*Observe the intricate dance of celestial bodies governed by Newton's law of universal gravitation.*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_bodies = st.slider("Number of Bodies", 3, 8, 4)
        simulation_time = st.slider("Simulation Steps", 100, 1000, 500, 50)
        
        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Computing gravitational interactions..."):
                # Initialize random bodies
                np.random.seed(42)
                bodies = []
                colors = plt.cm.tab10(np.linspace(0, 1, n_bodies))
                
                for i in range(n_bodies):
                    angle = 2 * np.pi * i / n_bodies
                    radius = 3 + np.random.rand() * 2
                    bodies.append({
                        'mass': 0.5 + np.random.rand() * 1.5,
                        'pos': np.array([radius * np.cos(angle), radius * np.sin(angle)]),
                        'vel': np.array([-np.sin(angle), np.cos(angle)]) * 0.5,
                        'color': colors[i]
                    })
                
                trajectories = simulate_nbody(bodies, simulation_time, 0.01)
                
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_xlim(-8, 8)
                ax.set_ylim(-8, 8)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.2)
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                
                # Plot trajectories
                for i, traj in enumerate(trajectories):
                    ax.plot(traj[:, 0], traj[:, 1], color=bodies[i]['color'], 
                           alpha=0.4, linewidth=1, label=f'Body {i+1}')
                    # Current position
                    ax.plot(traj[-1, 0], traj[-1, 1], 'o', 
                           color=bodies[i]['color'], markersize=bodies[i]['mass']*8)
                
                ax.legend(facecolor='#262730', edgecolor='white')
                ax.set_title('Gravitational N-Body System', color='white', fontsize=16, pad=20)
                ax.tick_params(colors='white')
                
                with col2:
                    st.pyplot(fig)
                    plt.close()

elif simulation_type == "Wave Interference":
    st.header("🌊 Wave Interference Patterns")
    st.markdown("*Explore the superposition principle as waves from multiple sources create beautiful interference patterns.*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_sources = st.slider("Number of Wave Sources", 2, 4, 2)
        wavelength = st.slider("Wavelength", 0.5, 2.0, 1.0, 0.1)
        grid_size = st.slider("Grid Resolution", 50, 150, 100, 25)
        
        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Computing wave interference..."):
                # Create sources in a circle
                sources = []
                for i in range(n_sources):
                    angle = 2 * np.pi * i / n_sources
                    radius = 2.0
                    sources.append([
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                        1.0,  # amplitude
                        1.0   # frequency
                    ])
                
                frames, X, Y = simulate_wave_interference(sources, grid_size, wavelength, 20)
                
                fig, ax = plt.subplots(figsize=(10, 10))
                
                # Show one frame with high quality
                im = ax.contourf(X, Y, frames[10], levels=50, cmap='RdBu_r', vmin=-n_sources, vmax=n_sources)
                
                # Mark sources
                for source in sources:
                    ax.plot(source[0], source[1], 'y*', markersize=20, markeredgecolor='black', markeredgewidth=1)
                
                plt.colorbar(im, ax=ax, label='Amplitude')
                ax.set_aspect('equal')
                ax.set_title('Wave Interference Pattern', fontsize=16, pad=20)
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                
                with col2:
                    st.pyplot(fig)
                    plt.close()

elif simulation_type == "Collision Physics":
    st.header("💥 Elastic Collision Dynamics")
    st.markdown("*Watch particles bounce off walls and each other, conserving momentum and energy in perfectly elastic collisions.*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        n_particles = st.slider("Number of Particles", 3, 12, 6)
        box_size = st.slider("Box Size", 5, 15, 10)
        
        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Simulating collisions..."):
                np.random.seed(42)
                particles = []
                colors = plt.cm.tab10(np.linspace(0, 1, n_particles))
                
                for i in range(n_particles):
                    particles.append({
                        'pos': np.random.uniform(-box_size/3, box_size/3, 2),
                        'vel': np.random.uniform(-2, 2, 2),
                        'mass': 0.5 + np.random.rand() * 1.5,
                        'radius': 0.3 + np.random.rand() * 0.3,
                        'color': colors[i]
                    })
                
                trajectories = simulate_collisions(particles, 500, 0.05, box_size)
                
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.set_xlim(-box_size/2-1, box_size/2+1)
                ax.set_ylim(-box_size/2-1, box_size/2+1)
                ax.set_aspect('equal')
                ax.set_facecolor('#0e1117')
                fig.patch.set_facecolor('#0e1117')
                
                # Draw box
                box = Rectangle((-box_size/2, -box_size/2), box_size, box_size,
                               fill=False, edgecolor='white', linewidth=2)
                ax.add_patch(box)
                
                # Plot trajectories and particles
                for i, (traj, particle) in enumerate(zip(trajectories, particles)):
                    ax.plot(traj[:, 0], traj[:, 1], color=particle['color'], 
                           alpha=0.3, linewidth=0.8)
                    circle = Circle(traj[-1], particle['radius'], 
                                  color=particle['color'], alpha=0.8)
                    ax.add_patch(circle)
                
                ax.grid(True, alpha=0.2, color='white')
                ax.set_title('Elastic Collision System', color='white', fontsize=16, pad=20)
                ax.tick_params(colors='white')
                
                with col2:
                    st.pyplot(fig)
                    plt.close()

elif simulation_type == "Fluid Dynamics":
    st.header("🌀 Fluid Flow Simulation")
    st.markdown("*Visualize fluid motion with vorticity and velocity fields based on simplified Navier-Stokes equations.*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        grid_size = st.slider("Grid Resolution", 30, 80, 50, 10)
        viscosity = st.slider("Viscosity", 0.001, 0.05, 0.01, 0.005)
        steps = st.slider("Simulation Steps", 20, 100, 50, 10)
        
        if st.button("🚀 Run Simulation", type="primary"):
            with st.spinner("Computing fluid dynamics..."):
                frames_u, frames_v, X, Y = simulate_fluid(grid_size, steps, viscosity)
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
                
                # Velocity magnitude
                velocity_mag = np.sqrt(frames_u[-1]**2 + frames_v[-1]**2)
                im1 = ax1.contourf(X, Y, velocity_mag, levels=20, cmap='hot')
                ax1.streamplot(X, Y, frames_u[-1], frames_v[-1], color='cyan', 
                              density=1.5, linewidth=1, arrowsize=1.2)
                plt.colorbar(im1, ax=ax1, label='Velocity Magnitude')
                ax1.set_title('Velocity Field with Streamlines', fontsize=14)
                ax1.set_aspect('equal')
                
                # Vorticity
                vorticity = np.gradient(frames_v[-1])[0] - np.gradient(frames_u[-1])[1]
                im2 = ax2.contourf(X, Y, vorticity, levels=20, cmap='RdBu_r')
                plt.colorbar(im2, ax=ax2, label='Vorticity')
                ax2.set_title('Vorticity Field', fontsize=14)
                ax2.set_aspect('equal')
                
                plt.tight_layout()
                
                with col2:
                    st.pyplot(fig)
                    plt.close()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>Interactive Physics Simulator</strong></p>
    <p>Built with Streamlit | Powered by NumPy & Matplotlib</p>
    <p style='font-size: 0.9rem;'>All simulations use accurate physics equations including conservation laws, 
    Newtonian mechanics, and fundamental physical principles.</p>
</div>
""", unsafe_allow_html=True)
