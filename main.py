from math import pi, sqrt, atan2, cos, sin, radians, degrees
import matplotlib.pyplot as plt
import os
from concurrent.futures import ProcessPoolExecutor
from random import randint, seed

RAD = 1.571
AIR_DENSITY = 1.225
BB_DIAMETER = 6e-3
BB_RADIUS = BB_DIAMETER/2
INITIAL_POSITION = [0, 1.8]
SPHERE_DRAG_COEF = 0.47
SPHERE_FRONTAL_AREA = pi*BB_RADIUS**2
TIMESTEP = 1/100
GRAVITY = 9.81
M_TO_FEET = 3.28084

def calc_vortex_strength(angular_velocity: float) -> float:
    return 2*pi*BB_RADIUS**2*angular_velocity


def get_intiial_velocity(mass: float, energy: float) -> float:
    angle = radians(0)
    velocity = sqrt(energy/(0.5*mass))
    velocity_x = cos(angle) * velocity
    velocity_y = sin(angle) * velocity
    # calculates velocity according to mass and energy
    return [velocity_x, velocity_y]


def get_initial_hop_angular_velocity(component_velocity: float, mass: float) -> float:
    # Gets initial hop spin for straight flight
    # Gets hop spin according to magnus effect https://www.fxsolver.com/browse/formulas/Magnus+effect
    # G     = F/(diameter * air density * velocity)
    # omega = G/(2*pi*r^2)
    F = mass * GRAVITY
    G = F/(BB_DIAMETER * AIR_DENSITY * component_velocity)
    angular_velocity = G/(2*pi*BB_RADIUS**2)
    return angular_velocity


def calc_magnus_effect(velocity: list, mass: float, angular_velocity: float) -> float:
    G = calc_vortex_strength(angular_velocity)
    speed = sqrt(velocity[0]**2 + velocity[1]**2)
    direction = atan2(velocity[0], velocity[1])
    F = AIR_DENSITY*speed*G*BB_DIAMETER
    delta_speed = F/mass
    delta_X = cos(direction) * delta_speed
    delta_Y = sin(direction) * delta_speed
    #print("Magnus delta ", delta_X, delta_Y, velocity)
    return [delta_X, delta_Y]


def calc_drag(velocity: float, mass: float) -> float:
    # https://www.engineeringtoolbox.com/drag-coefficient-d_627.html
    speed = sqrt(velocity[0]**2 + velocity[1]**2)
    direction = atan2(velocity[0], velocity[1])
    F = SPHERE_DRAG_COEF*0.5*AIR_DENSITY*speed**2*SPHERE_FRONTAL_AREA
    delta_speed = F/mass
    delta_X = sin(direction) * delta_speed
    delta_Y = cos(direction) * delta_speed
    #print("Drag delta ", delta_X, delta_Y, velocity)
    return [-delta_X, -delta_Y]


def update_position(position: list, velocity: list) -> list:
    return [position[0] + velocity[0]*TIMESTEP, position[1] + velocity[1]*TIMESTEP]


def update_velocity(velocity: list, deltas: list) -> list:
    for delta in deltas:
        velocity[0] += delta[0]*TIMESTEP
        velocity[1] += delta[1]*TIMESTEP
    velocity[1] += -GRAVITY * TIMESTEP  # Gravity
    return velocity.copy()


def calc_bb(mass: float, energy: float) -> dict:
    velocity = get_intiial_velocity(mass, energy)
    position = INITIAL_POSITION.copy()
    angular_velocity = get_initial_hop_angular_velocity(velocity[0], mass)
    flight_time = 0
    points = []
    while position[1] > 0:
        delta_magnus = calc_magnus_effect(velocity, mass, angular_velocity)
        delta_drag = calc_drag(velocity, mass)
        velocity = update_velocity(velocity, [delta_magnus, delta_drag])
        position = update_position(position, velocity)
        flight_time += TIMESTEP
        points.append((flight_time, position, velocity))
    # [print(result) for result in points]
    #return [mass, energy, round(flight_time, 2), round(position[0], 2)]
    return {
        "mass" : mass,
        "energy" : energy,
        "time" : flight_time,
        "summary" : (mass, energy, round(flight_time, 2), round(position[0], 2)),
        "points" : points
    }

def get_bb_fps(result: dict) -> str:
    return f' ({round(sqrt(result["energy"]/(0.5*result["mass"])) * M_TO_FEET)}fps)'

def get_plot_label(subject, result) -> str:
    fps = get_bb_fps(result)
    if subject == "J":
        label = f'{round(result["mass"]*1000, 2)}g'
    else:
        label = f'{round(result["energy"], 2)}J'
    return label + fps

def plot_graphs(series, datapoint, subject):
    if subject == "J":
        directory = f'energy/{datapoint}J.png'
        trajectory_title = f'Trajectories at {datapoint}J for various bb sizes'
        time_distance_title = f'Time-distance graphs at {datapoint}J for various bb sizes'
        time_velocity_title = f'Time-velocity graphs at {datapoint}J for various bb sizes'
        print(f'Plotting graphs for {datapoint}J bbs')
    else:
        mass_g = round(datapoint*1000, 2)
        directory = f'mass/{mass_g}g.png'
        trajectory_title = f'Trajectories for {mass_g}g bbs at various energy levels'
        time_distance_title = f'Time-distance graphs for {mass_g}g bbs at various energy levels'
        time_velocity_title = f'Time-velocity graphs for {mass_g}g bbs at various energy levels'
        print(f'Plotting graphs for {mass_g}g bbs')
    plot_trajectory(series, trajectory_title, subject, directory)
    plot_time_distance(series, time_distance_title, subject, directory)
    plot_velocity_time(series, time_velocity_title, subject, directory)
    
def plot_trajectory(series, title, subject, directory):
    fig = plt.figure(dpi = 200)
    ax = fig.add_subplot()
    ax.set_title(title)
    ax.set_ylabel('bb Height (ft)')
    ax.set_xlabel('bb Distance (ft)')
    for result in series:
        points = result["points"]
        positions_x = [point[1][0]*M_TO_FEET for point in points]
        positions_y = [point[1][1]*M_TO_FEET for point in points]
        trajectory = ax.plot(positions_x, positions_y)[0]
        trajectory.set_lw(0.75)
        trajectory.set_ls('--')
        trajectory.set_label(get_plot_label(subject, result))
    ax.legend(loc='lower left')
    directory = "trajectory/"+directory
    fig.savefig(directory)
    plt.close()

def plot_time_distance(series, title, subject, directory):
    fig = plt.figure(dpi = 200, figsize = (10, 5))
    ax = fig.add_subplot()
    ax.set_title(title)
    ax.set_ylabel('Distance Travelled (ft)')
    ax.set_xlabel('Time (s)')
    for i, result in enumerate(series):
        points = result["points"]
        time = [point[0] for point in points]
        distance = [point[1][0]*M_TO_FEET for point in points]
        trajectory = ax.plot(time, distance)[0]
        trajectory.set_lw(0.75)
        trajectory.set_ls('--')
        trajectory.set_zorder(len(series)-i)
        trajectory.set_label(get_plot_label(subject, result))
    ax.legend(loc='lower right')
    directory = "time_distance/"+directory
    fig.savefig(directory)
    plt.close()

def plot_velocity_time(series, title, subject, directory):
    fig = plt.figure(dpi = 200)
    ax = fig.add_subplot()
    ax.set_title(title)
    ax.set_ylabel('bb velocity (fps)')
    ax.set_xlabel('Time elapsed (s)')
    for result in series:
        points = result["points"]
        time = [point[0] for point in points]
        velocity = [point[2][0]*M_TO_FEET for point in points]
        trajectory = ax.plot(time, velocity)[0]
        trajectory.set_lw(0.75)
        trajectory.set_ls('--')
        trajectory.set_label(get_plot_label(subject, result))
    ax.legend(loc='upper right')
    directory = "velocity/"+directory
    fig.savefig(directory)
    plt.close()

def main():
    for directory in (
        "trajectory/energy/",
        "trajectory/mass/",
        "time_distance/energy/",
        "time_distance/mass/",
        "velocity/energy/",
        "velocity/mass/",
    ):
        if not os.path.isdir(directory):
            os.makedirs(directory)
    results = []
    energies = (1, 1.138, 1.486, 1.881, 2.322)
    masses = (0.0002, 0.00025, 0.00028, 0.0003, 0.00032, 0.00035, 0.0004, 0.00045, 0.0005)
    for energy in energies:
        for mass in masses:
            result = calc_bb(mass, energy)
            results.append(result)
            print(result["summary"])
    with ProcessPoolExecutor() as pe:
        for energy in energies:
            series = [result for result in results if result["energy"] == energy]
            #plot_graphs(series, energy, results, "J")
            pe.submit(plot_graphs, series, energy, "J")
        for mass in masses:
            series = [result for result in results if result["mass"] == mass]
            #plot_graphs(series, mass, results, "M")
            pe.submit(plot_graphs, series, mass, "M")
    # plot_graphs([results[0]], 1, "J")

if __name__ == "__main__":
    main()
