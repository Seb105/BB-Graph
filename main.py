from math import pi, sqrt, atan2, cos, sin, radians, degrees
import matplotlib.pyplot as plt
import os

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
    return velocity


def calc_bb(mass: float, energy: float):
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
    #[print(result) for result in results]
    #return [mass, energy, round(flight_time, 2), round(position[0], 2)]
    return {
        "mass" : mass,
        "energy" : energy,
        "time" : flight_time,
        "summary" : (mass, energy, round(flight_time, 2), round(position[0], 2)),
        "points" : points
    }

def get_bb_fps(result):
    return f' ({round(sqrt(result["energy"]/(0.5*result["mass"])) * M_TO_FEET)}fps)'

def get_plot_label(subject, result):
    fps = get_bb_fps(result)
    if subject == "J":
        label = f'{round(result["mass"]*1000, 2)}g'
    else:
        label = f'{round(result["energy"], 2)}J'
    return label + fps

    
def plot_trajectory(results, title, subject, directory):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel('BB Height')
    ax.set_xlabel('BB Distance')
    for result in results:
        points = result["points"]
        positions_x = [point[1][0]*M_TO_FEET for point in points]
        positions_y = [point[1][1]*M_TO_FEET for point in points]
        trajectory = ax.plot(positions_x, positions_y)
        label = get_plot_label(subject, result)
        trajectory[0].set_label(label)
    ax.legend(loc='lower left')
    directory = "trajectory/"+directory
    fig.savefig(directory)

def plot_time_to_target(results, title, subject, directory):
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_ylabel('BB Height')
    ax.set_xlabel('BB Distance')
    for result in results:
        points = result["points"]
        positions_x = [point[1][0]*M_TO_FEET for point in points]
        positions_y = [point[1][1]*M_TO_FEET for point in points]
        trajectory = ax.plot(positions_x, positions_y)
        label = get_plot_label(subject, result)
        trajectory[0].set_label(label)
    ax.legend(loc='lower left')
    directory = "trajectory/"+directory
    # if not os.path.exists(directory):
    #     os.makedirs(directory)
    fig.savefig(directory)

def plot_energy_trajectories(energy, results):
    series = [result for result in results if result["energy"] == energy]
    trajectory_title = f'Trajectories at {energy}J for various bb sizes'
    subject = "J"
    directory = f'energy/{energy}J.png'
    plot_trajectory(series, trajectory_title, subject, directory)

def plot_mass_trajectories(mass, results):
    series = [result for result in results if result["mass"] == mass]
    mass_g = round(mass*1000, 2)
    trajectory_title = f'Trajectories for {mass_g}g bbs at various energy levels'
    subject = "M"
    directory = f'mass/{mass_g}g.png'
    plot_trajectory(series, trajectory_title, subject, directory)

def main():
    results = []
    energies = (1, 1.138, 1.486, 1.881, 2.322)
    masses = (0.0002, 0.00025, 0.00028, 0.0003, 0.00032, 0.00035, 0.0004, 0.00045, 0.0005)
    for energy in energies:
        for mass in masses:
            result = calc_bb(mass, energy)
            results.append(result)
            print(result["summary"])
    #test_results = [result for result in results if result["energy"] == 1]
    for energy in energies:
        plot_energy_trajectories(energy, results)
    for mass in masses:
        plot_mass_trajectories(mass, results)


main()
