from math import pi, sqrt, atan2, cos, sin, radians
import matplotlib.pyplot as plt
from matplotlib import ticker
import os
import concurrent.futures

RAD = 1.571
AIR_DENSITY = 1.225
AIR_DYNAMIC_VISCOSITY = 1.81e-5
BB_DIAMETER = 6e-3
BB_RADIUS = BB_DIAMETER/2
SPHERE_DRAG_COEF = 0.47
SPHERE_FRONTAL_AREA = pi*BB_RADIUS**2
TIMESTEP = 1/1000
GRAVITY = 9.81
FEET_PER_METRE = 3.28084
INITIAL_POSITION = [0, 5/FEET_PER_METRE]
ENERGIES = (0.9, 1.0, 1.138, 1.486, 1.881, 2.322)
MASSES = (0.0002, 0.00025, 0.00028, 0.0003, 0.00032, 0.00035, 0.0004, 0.00045, 0.0005)

def remap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def m2ft(x):
    return x*FEET_PER_METRE

def ft2m(x):
    return x/FEET_PER_METRE

class BB_Class:
    def __init__(self, mass, energy):
        self.mass = mass
        self.initial_energy = energy
        self.velocity = BB_Class.get_intial_velocity(mass, energy)
        self.angular_velocity = BB_Class.get_initial_hop_angular_velocity(self.velocity[0], mass)
        self.position = INITIAL_POSITION.copy()
        self.moment_of_inerta = (2/5)*mass*BB_RADIUS**2
        # self.ballistic_coefficient = mass/(SPHERE_DRAG_COEF + SPHERE_FRONTAL_AREA)
        self.flight_time = 0

    def reynolds_number(self):
        flow_speed = sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        return (AIR_DENSITY*flow_speed*BB_DIAMETER)/AIR_DYNAMIC_VISCOSITY

    def drag_coef(self):
        return 24/self.reynolds_number()

    def vortex_strength(self) -> float:
        return 2*pi*BB_RADIUS**2*self.angular_velocity

    @classmethod
    def get_intial_velocity(cls, mass: float, energy: float) -> list:
        angle = radians(0)
        velocity = sqrt(energy/(0.5*mass))
        velocity_x = cos(angle) * velocity
        velocity_y = sin(angle) * velocity
        # calculates velocity according to mass and energy
        return [velocity_x, velocity_y]

    @classmethod
    def get_initial_hop_angular_velocity(cls, component_velocity: float, mass: float) -> float:
        # Gets initial hop spin for straight flight
        # Gets hop spin according to magnus effect https://www.fxsolver.com/browse/formulas/Magnus+effect
        F = mass * GRAVITY
        G = F/(BB_DIAMETER * AIR_DENSITY * component_velocity)
        angular_velocity = G/(2*pi*BB_RADIUS**2)
        return angular_velocity

    def update_angular_velocity(self):
        # https://physics.stackexchange.com/questions/304742/angular-drag-on-body
        shear_force = 2*pi*AIR_DYNAMIC_VISCOSITY * \
            self.angular_velocity*BB_DIAMETER*BB_RADIUS
        shear_torque = BB_RADIUS * shear_force
        shear_delta = (shear_torque / self.moment_of_inerta) * TIMESTEP

        friction_torque = pi*self.angular_velocity**2*BB_RADIUS**4 * \
            BB_DIAMETER*AIR_DENSITY*SPHERE_DRAG_COEF
        friction_delta = (friction_torque / self.moment_of_inerta) * TIMESTEP
        self.angular_velocity = self.angular_velocity - shear_delta - friction_delta

    def calc_magnus_effect(self):
        G = self.vortex_strength()
        speed = sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        direction = atan2(self.velocity[0], self.velocity[1]) - RAD
        F = AIR_DENSITY*speed*G*BB_DIAMETER
        delta_speed = F/self.mass
        delta_X = sin(direction) * delta_speed
        delta_Y = cos(direction) * delta_speed
        # print("Magnus delta ", delta_X, delta_Y, velocity)
        return [delta_X, delta_Y]

    def calc_drag(self) -> list:
        speed = sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        direction = atan2(self.velocity[0], self.velocity[1])
        # https://en.wikipedia.org/wiki/Drag_equation
        F = SPHERE_DRAG_COEF*0.5*AIR_DENSITY*speed**2*SPHERE_FRONTAL_AREA
        # F = 6*pi*BB_RADIUS*AIR_DYNAMIC_VISCOSITY*speed # https://en.wikipedia.org/wiki/Stokes%27_law
        delta_speed = F/self.mass
        delta_X = sin(direction) * delta_speed
        delta_Y = cos(direction) * delta_speed
        return [-delta_X, -delta_Y]

    def update_position(self):
        self.position = [
            self.position[0] + self.velocity[0] * TIMESTEP, 
            self.position[1] + self.velocity[1] * TIMESTEP
        ]

    def update_velocity(self):
        deltas = [
            self.calc_magnus_effect(),
            self.calc_drag()
        ]
        velocity = self.velocity.copy()
        for delta in deltas:
            velocity[0] += delta[0]*TIMESTEP
            velocity[1] += delta[1]*TIMESTEP
        velocity[1] += -GRAVITY * TIMESTEP  # Gravity
        self.velocity = velocity

    def run_sim(self) -> dict:
        points = []
        points.append((self.flight_time, self.position, self.velocity))
        while self.position[1] > 0:
            self.update_angular_velocity()
            self.update_velocity()
            self.update_position()
            self.flight_time += TIMESTEP
            points.append((self.flight_time, self.position, self.velocity))
        return {
            "mass": self.mass,
            "energy": self.initial_energy,
            "time": self.flight_time,
            "summary": (self.mass, self.initial_energy, round(self.flight_time, 2), round(self.position[0], 2)),
            "points": points
        }


def get_bb_fps(result: dict) -> str:
    return f' ({round(sqrt(result["energy"]/(0.5*result["mass"])) * FEET_PER_METRE)}fps)'


def get_plot_label(subject, result) -> str:
    fps = get_bb_fps(result)
    if subject == "J":
        label = f'{round(result["mass"]*1000, 2)}g'
    else:
        label = f'{round(result["energy"], 2)}J'
    return label + fps


def plot_graphs(series, datapoint, subject):
    fig = plt.figure(dpi=200, figsize=(20, 15), tight_layout=True)
    if subject == "J":
        directory = f'results/energy/{datapoint}J.png'
        trajectory_title = f'Trajectories at {datapoint} joules for various bb weights'
        time_distance_title = f'Time-distance graphs at {datapoint} joules for various bb weights'
        time_velocity_title = f'Time-velocity graphs at {datapoint} joules for various bb weights'
        # print(f'Plotting graphs for {datapoint}J bbs')
    else:
        mass_g = round(datapoint*1000, 2)
        directory = f'results/mass/{mass_g}g.png'
        trajectory_title = f'Trajectories for {mass_g}g bbs at various energy levels'
        time_distance_title = f'Time-distance graphs for {mass_g}g bbs at various energy levels'
        time_velocity_title = f'Time-velocity graphs for {mass_g}g bbs at various energy levels'
        # print(f'Plotting graphs for {mass_g}g bbs')
    plot_trajectory(fig, series, trajectory_title, subject)
    plot_time_distance(fig, series, time_distance_title, subject)
    plot_time_velocity(fig, series, time_velocity_title, subject)
    fig.savefig(directory)


def configure_line(z, line, subject, result):
    line.set_lw(2)
    line.set_alpha(0.5)
    line.set_ls('--')
    line.set_label(get_plot_label(subject, result))
    line.set_zorder(z)


def plot_trajectory(fig, series, title, subject):
    ax = fig.add_subplot(2, 1, 1)
    ax.grid(b=True)
    ax.set_title(title)
    ax.set_ylabel('Height (ft)')
    ax.set_xlabel('Distance (ft)')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(25))
    for i, result in enumerate(series):
        points = result["points"]
        positions_x = [point[1][0]*FEET_PER_METRE for point in points]
        positions_y = [point[1][1]*FEET_PER_METRE for point in points]
        trajectory = ax.plot(positions_x, positions_y)[0]
        configure_line(i, trajectory, subject, result)
    ax.set_xlim(xmin=0, xmax=275)
    ax.set_ylim(ymin=0, ymax=6.5)
    secxax = ax.secondary_xaxis('top', functions=(ft2m, m2ft))
    secyax = ax.secondary_yaxis('right', functions=(ft2m, m2ft))
    secxax.set_xlabel("Distance (m)")
    secyax.set_ylabel("Height (m)")
    ax.legend(loc='lower left')


def plot_time_distance(fig, series, title, subject):
    ax = fig.add_subplot(2, 2, 3)
    ax.grid(b=True)
    ax.set_title(title)
    ax.set_ylabel('Distance Travelled (ft)')
    ax.set_xlabel('Time (s)')
    for i, result in enumerate(series):
        points = result["points"]
        time = [point[0] for point in points]
        distance = [point[1][0]*FEET_PER_METRE for point in points]
        trajectory = ax.plot(time, distance)[0]
        configure_line(len(series)-i, trajectory, subject, result)
    ax.set_xlim(xmin=0, xmax=1.6)
    ax.set_ylim(ymin=0, ymax=275)
    secyax = ax.secondary_yaxis('right', functions=(ft2m, m2ft))
    secyax.set_ylabel("Distance Travelled (m)")
    ax.legend(loc='lower right')


def plot_time_velocity(fig, series, title, subject):
    ax = fig.add_subplot(2, 2, 4)
    ax.grid(b=True)
    ax.set_title(title)
    ax.set_ylabel('Velocity (fps)')
    ax.set_xlabel('Time (s)')
    for i, result in enumerate(series):
        points = result["points"]
        time = [point[0] for point in points]
        velocity = [point[2][0]*FEET_PER_METRE for point in points]
        trajectory = ax.plot(time, velocity)[0]
        configure_line(len(series)-i, trajectory, subject, result)
    ax.set_xlim(xmin=0, xmax=1.6)
    ax.set_ylim(ymin=0, ymax=500)
    secyax = ax.secondary_yaxis('right', functions=(ft2m, m2ft))
    secyax.set_ylabel("Velocity (m/s)")
    ax.legend(loc='upper right')

# finds correct hop for 1ft of rise over flight time.
def run_bb_optimally(pair):
    mass, energy = pair
    best_result = BB_Class(mass, energy).run_sim()
    step = 0.01
    i = 1 + step
    while True:
        bb = BB_Class(mass, energy)
        bb.angular_velocity *= i
        result = bb.run_sim()
        points = result['points']
        max_y = max([point[1][1]*FEET_PER_METRE for point in points])
        if max_y < 6:
            best_result = result
            i += step
        else:
            break
    print(f'Done {mass}, {energy}')
    return best_result


def main():
    for directory in (
        "results/energy",
        "results/mass",
    ):
        if not os.path.isdir(directory):
            os.makedirs(directory)
    results = []
    pairs = []
    for energy in ENERGIES:
        for mass in MASSES:
            pairs.append((mass, energy))
    with concurrent.futures.ProcessPoolExecutor() as pe:
        # result_futures = pe.map(run_bb_optimally, pairs)
        # list(concurrent.futures.as_completed(result_futures))
        results = list(map(run_bb_optimally, pairs))
        for energy in ENERGIES:
            series = [result for result in results if result["energy"] == energy]
            pe.submit(plot_graphs, series, energy, "J")
        for mass in MASSES:
            series = [result for result in results if result["mass"] == mass]
            pe.submit(plot_graphs, series, mass, "M")


if __name__ == "__main__":
    main()
