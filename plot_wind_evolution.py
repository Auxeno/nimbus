"""Plot wind evolution over time using Ornstein-Uhlenbeck process."""

import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp

from nimbus.core.config import WindConfig
from nimbus.core.state import Wind
from nimbus.core.interface import update_wind
from nimbus.core.primitives import FLOAT_DTYPE

# Configuration
wind_config = WindConfig()  # Using default values
dt = 0.01  # 100 Hz update rate
simulation_time = 60.0  # 60 seconds total
num_steps = int(simulation_time / dt)

# Initialize wind state
wind = Wind(
    mean=jnp.array([10.0, 5.0, 0.0], dtype=FLOAT_DTYPE),  # Base wind: 10 m/s north, 5 m/s east
    gust=jnp.zeros(3, dtype=FLOAT_DTYPE),
)

# Storage for time series
time_points = np.arange(0, simulation_time, dt)
wind_history = {
    'north': np.zeros(num_steps),
    'east': np.zeros(num_steps),
    'down': np.zeros(num_steps),
    'gust_north': np.zeros(num_steps),
    'gust_east': np.zeros(num_steps),
    'gust_down': np.zeros(num_steps),
}

# Run simulation
key = jax.random.PRNGKey(42)
for i in range(num_steps):
    # Store current values
    total_wind = wind.mean + wind.gust
    wind_history['north'][i] = float(total_wind[0])
    wind_history['east'][i] = float(total_wind[1])
    wind_history['down'][i] = float(total_wind[2])
    wind_history['gust_north'][i] = float(wind.gust[0])
    wind_history['gust_east'][i] = float(wind.gust[1])
    wind_history['gust_down'][i] = float(wind.gust[2])
    
    # Update wind
    key, subkey = jax.random.split(key)
    wind = update_wind(
        subkey,
        wind,
        wind_config,
        jnp.array(dt, dtype=FLOAT_DTYPE),
    )

# Create figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
fig.suptitle(f'Wind Evolution Over Time (OU Process)\n'
             f'Config: intensity={wind_config.gust_intensity} m/s, '
             f'duration={wind_config.gust_duration} s, '
             f'vertical_damping={wind_config.vertical_damping}', 
             fontsize=14)

# Plot total wind (mean + gust)
ax = axes[0, 0]
ax.plot(time_points, wind_history['north'], 'b-', linewidth=1.5, alpha=0.8, label='North')
ax.plot(time_points, wind_history['east'], 'g-', linewidth=1.5, alpha=0.8, label='East')
ax.plot(time_points, wind_history['down'], 'r-', linewidth=1.5, alpha=0.8, label='Down')
ax.axhline(y=float(wind.mean[0]), color='b', linestyle='--', alpha=0.3, label='Mean North')
ax.axhline(y=float(wind.mean[1]), color='g', linestyle='--', alpha=0.3, label='Mean East')
ax.axhline(y=float(wind.mean[2]), color='r', linestyle='--', alpha=0.3, label='Mean Down')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Wind Velocity (m/s)')
ax.set_title('Total Wind (Mean + Gust)')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot gust components only
ax = axes[0, 1]
ax.plot(time_points, wind_history['gust_north'], 'b-', linewidth=1.5, alpha=0.8, label='North')
ax.plot(time_points, wind_history['gust_east'], 'g-', linewidth=1.5, alpha=0.8, label='East')
ax.plot(time_points, wind_history['gust_down'], 'r-', linewidth=1.5, alpha=0.8, label='Down')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Gust Velocity (m/s)')
ax.set_title('Gust Components Only')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Plot horizontal wind magnitude
horizontal_mag = np.sqrt(wind_history['north']**2 + wind_history['east']**2)
horizontal_gust_mag = np.sqrt(wind_history['gust_north']**2 + wind_history['gust_east']**2)
ax = axes[1, 0]
ax.plot(time_points, horizontal_mag, 'purple', linewidth=1.5, alpha=0.8)
ax.axhline(y=np.sqrt(float(wind.mean[0])**2 + float(wind.mean[1])**2), 
           color='purple', linestyle='--', alpha=0.3, label='Mean magnitude')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Speed (m/s)')
ax.set_title('Horizontal Wind Speed (magnitude of N-E components)')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot wind direction (meteorological convention - where wind comes FROM)
wind_dir = np.degrees(np.arctan2(-wind_history['east'], -wind_history['north'])) % 360
ax = axes[1, 1]
ax.plot(time_points, wind_dir, 'orange', linewidth=1.5, alpha=0.8)
mean_dir = np.degrees(np.arctan2(-float(wind.mean[1]), -float(wind.mean[0]))) % 360
ax.axhline(y=mean_dir, color='orange', linestyle='--', alpha=0.3, label=f'Mean direction: {mean_dir:.1f}°')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Direction (degrees)')
ax.set_title('Wind Direction (meteorological - FROM)')
ax.set_ylim([0, 360])
ax.legend()
ax.grid(True, alpha=0.3)

# Plot power spectral density of gusts
from scipy import signal
ax = axes[2, 0]
for component, color, label in [('gust_north', 'b', 'North'), 
                                 ('gust_east', 'g', 'East'), 
                                 ('gust_down', 'r', 'Down')]:
    freqs, psd = signal.periodogram(wind_history[component], fs=1/dt)
    ax.loglog(freqs[1:], psd[1:], color=color, alpha=0.7, label=label)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('PSD (m²/s²/Hz)')
ax.set_title('Power Spectral Density of Gusts')
ax.legend()
ax.grid(True, which='both', alpha=0.3)

# Plot autocorrelation of horizontal gust
ax = axes[2, 1]
from scipy.stats import pearsonr
max_lag = 100  # 1 second at 100Hz
lags = np.arange(0, max_lag) * dt
autocorr_north = []
autocorr_east = []
for lag in range(max_lag):
    if lag == 0:
        autocorr_north.append(1.0)
        autocorr_east.append(1.0)
    else:
        corr_n = np.corrcoef(wind_history['gust_north'][:-lag], 
                             wind_history['gust_north'][lag:])[0, 1]
        corr_e = np.corrcoef(wind_history['gust_east'][:-lag], 
                            wind_history['gust_east'][lag:])[0, 1]
        autocorr_north.append(corr_n if not np.isnan(corr_n) else 0)
        autocorr_east.append(corr_e if not np.isnan(corr_e) else 0)

ax.plot(lags, autocorr_north, 'b-', alpha=0.8, label='North')
ax.plot(lags, autocorr_east, 'g-', alpha=0.8, label='East')
# Theoretical exponential decay for OU process
theoretical_decay = np.exp(-lags / wind_config.gust_duration)
ax.plot(lags, theoretical_decay, 'k--', alpha=0.5, 
        label=f'Theory (τ={wind_config.gust_duration}s)')
ax.axhline(y=0, color='black', linestyle='-', alpha=0.2)
ax.set_xlabel('Lag (s)')
ax.set_ylabel('Autocorrelation')
ax.set_title('Autocorrelation of Horizontal Gusts')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistics
print("\nWind Statistics:")
print(f"Mean wind: North={float(wind.mean[0]):.1f}, East={float(wind.mean[1]):.1f}, Down={float(wind.mean[2]):.1f} m/s")
print(f"\nGust statistics (m/s):")
print(f"  North: mean={np.mean(wind_history['gust_north']):.3f}, std={np.std(wind_history['gust_north']):.3f}")
print(f"  East:  mean={np.mean(wind_history['gust_east']):.3f}, std={np.std(wind_history['gust_east']):.3f}")
print(f"  Down:  mean={np.mean(wind_history['gust_down']):.3f}, std={np.std(wind_history['gust_down']):.3f}")
print(f"\nHorizontal gust magnitude: mean={np.mean(horizontal_gust_mag):.3f}, std={np.std(horizontal_gust_mag):.3f}")
print(f"Theoretical gust std (intensity): {wind_config.gust_intensity:.3f}")