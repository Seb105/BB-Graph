from math import pi, sqrt, atan2, cos, sin, radians
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np
import os
import concurrent.futures
import time
import ujson as json

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
MASSES =    (0.0002, 0.00025, 0.00028, 0.0003, 
            0.00032, 0.00035, 0.0004, 0.00045, 0.0005)
CLAMP_AXES = True

def remap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def m2ft(x):
    return x*FEET_PER_METRE

def ft2m(x):
    return x/FEET_PER_METRE

class BB_Class:
    def __init__(self, mass, energy, angle = 0, hop_multiplier = 1):
        self.mass = mass
        self.initial_energy = energy
        self.initial_angle = angle
        self.hop_multiplier = hop_multiplier
        self.velocity = BB_Class.get_intial_velocity(mass, energy, angle)
        self.angular_velocity = BB_Class.get_initial_hop_angular_velocity(self.velocity[0], mass)*hop_multiplier
        self.initial_rpm = self.angular_velocity * 9.5493
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
    def get_intial_velocity(cls, mass: float, energy: float, angle = 0) -> list:
        angle = radians(angle)
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
        max_x = 0
        max_y = 0
        while self.position[1] > 0:
            self.update_angular_velocity()
            self.update_velocity()
            self.update_position()
            max_x = max(max_x, self.position[0])
            max_y = max(max_y, self.position[1])
            self.flight_time += TIMESTEP
            points.append((self.flight_time, self.position, self.velocity))
        result = {
            "mass": self.mass,
            "energy": self.initial_energy,
            "angle": round(self.initial_angle, 2),
            "rpm": round(self.initial_rpm),
            "hop_mod": round(self.hop_multiplier, 2),
            "time": self.flight_time,
            "points": points,
            "max_x": max_x,
            "max_y": max_y,
        }
        return result

def get_bb_fps(result: dict) -> str:
    return f' ({round(sqrt(result["energy"]/(0.5*result["mass"])) * FEET_PER_METRE)}fps)'


def get_plot_label(subject, result) -> str:
    fps = get_bb_fps(result)
    stuff = f' {result["angle"]}° {result["rpm"]}rpm {result["hop_mod"]}x'
    if subject == "J":
        label = f'{round(result["mass"]*1000, 2)}g'
    else:
        label = f'{round(result["energy"], 2)}J'
    return label + fps + stuff


def plot_graphs(series, datapoint, subject):
    fig = plt.figure(dpi=200, figsize=(20, 15), tight_layout=True)
    if subject == "J":
        directory = f'graphs/energy/{datapoint}J.png'
        trajectory_title = f'Trajectories at {datapoint} joules for various bb weights'
        time_distance_title = f'Time-distance graphs at {datapoint} joules for various bb weights'
        time_velocity_title = f'Time-velocity graphs at {datapoint} joules for various bb weights'
        # print(f'Plotting graphs for {datapoint}J bbs')
    else:
        mass_g = round(datapoint*1000, 2)
        directory = f'graphs/mass/{mass_g}g.png'
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
    if CLAMP_AXES:
        ax.set_xlim(xmin=0, xmax=275)
        ax.set_ylim(ymin=0, ymax=6.5)
    secxax = ax.secondary_xaxis('top', functions=(ft2m, m2ft))
    secyax = ax.secondary_yaxis('right', functions=(ft2m, m2ft))
    secxax.set_xlabel("Distance (m)")
    secyax.set_ylabel("Height (m)")
    ax.legend(loc='lower left' if CLAMP_AXES else 'upper left')


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
    if CLAMP_AXES:
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
    if CLAMP_AXES:
        ax.set_xlim(xmin=0, xmax=1.6)
        ax.set_ylim(ymin=0, ymax=500)
    secyax = ax.secondary_yaxis('right', functions=(ft2m, m2ft))
    secyax.set_ylabel("Velocity (m/s)")
    ax.legend(loc='upper right')

def plot_spin_mods(results):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15,10))
    fig.tight_layout()
    fig.set_dpi(200)
    ax.view_init(30, -30)

    masses = np.asarray([result["mass"]*1000 for result in results])
    ax.set_xlabel('Mass (g)')
    ax.set_xticks (masses)
    ax.invert_xaxis()

    energies = np.asarray([result["energy"] for result in results])
    ax.set_ylabel('Energy (j)')
    ax.set_yticks(energies)

    angular_velocities = np.asarray([result["rpm"] for result in results])
    ax.set_zlabel('RPM for 1ft hop rise')

    surf = ax.plot_trisurf(masses, energies, angular_velocities, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    for x, y, z in zip(masses, energies, angular_velocities):
        ax.text(x, y, z, f'{int(z)}', fontsize="small")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    fig.savefig("hop_rpm.png")


# finds correct hop for 1ft of rise over flight time.
def run_bb_1ft_hop(pair):
    mass, energy = pair
    bb = BB_Class(mass, energy)
    best_result = bb.run_sim()
    best_x = best_result["max_x"]
    angle_step = 0.001
    hop_step = 0.001
    angle = 0
    while True:
        hop_multiplier = 1
        while hop_multiplier < 10:
            Progress_Bar.print(f'{angle}deg, {round(hop_multiplier, 2)}x, {round(mass*1000, 2)}g, {energy}j')
            result = BB_Class(mass, energy, angle, hop_multiplier).run_sim()
            max_x = result["max_x"]
            max_y = result["max_y"]
            if max_y > ft2m(6):
                break
            if max_x > best_x:
                best_x = max_x
                best_result = result
            hop_multiplier += hop_step
        angle += angle_step
        if angle>15 or (max_y>ft2m(6) and hop_multiplier == 1):
            break
    Progress_Bar.print(f'Done {mass*1000}g, {energy}j with {best_result["angle"]}deg and {best_result["hop_mod"]}', 2)
    return best_result

# finds correct hop and shooting angle for maximum distance.
def run_bb_max_dist(pair):
    mass, energy = pair
    bb = BB_Class(mass, energy)
    best_result = bb.run_sim()
    best_x = best_result["max_x"]
    angle_step = 1
    hop_step = 0.1
    angle = 0
    while True:
        angle_is_improvement = False
        hop_multiplier = 1
        while hop_multiplier < 10:
            Progress_Bar.print(f'{angle}deg, {round(hop_multiplier, 2)}x, {round(mass*1000, 2)}g, {energy}j')
            result = BB_Class(mass, energy, angle, hop_multiplier).run_sim()
            max_x = result["max_x"]
            if max_x > best_x:
                best_x = max_x
                best_result = result
                angle_is_improvement = True
            hop_multiplier += hop_step
        angle += angle_step
        if not angle_is_improvement:
            break
    Progress_Bar.print(f'Done {mass*1000}g, {energy}j with {best_result["angle"]}deg and {best_result["hop_mod"]}', 2)
    return best_result

def main():
    for directory in (
        "graphs/energy",
        "graphs/mass",
        "cache",
    ):
        if not os.path.isdir(directory):
            os.makedirs(directory)
    pairs = []
    for energy in ENERGIES:
        for mass in MASSES:
            pairs.append((mass, energy))
    with concurrent.futures.ProcessPoolExecutor() as pe:
        futures = []
        results = []
        if not os.path.isfile("cache/results.json"):
            for pair in pairs:
                futures.append(pe.submit(run_bb_1ft_hop, pair))
            progress_bar = Progress_Bar("Running sims", len(pairs))
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress_bar.update_progress()
            del(progress_bar)
            with open("cache/results.json", "w") as f:
                json.dump(results, f)
        else:
            with open("cache/results.json", "r") as f:
                results = json.load(f)
        # list(pe.map(run_bb_1ft_hop, pairs))
        #results = map(run_bb_1ft_hop, pairs)
        results = sorted(results, key=lambda result: result["energy"])
        results = sorted(results, key=lambda result: result["mass"])
        for energy in ENERGIES:
            series = [result for result in results if result["energy"] == energy]
            pe.submit(plot_graphs, series, energy, "J")
        for mass in MASSES:
            series = [result for result in results if result["mass"] == mass]
            pe.submit(plot_graphs, series, mass, "M")
        #plot_spin_mods(results)
    print("Done")

class Progress_Bar():
    def __init__(self, activity, count):
        self.activity = activity
        self.count = count
        self.i = -1
        self.barlength = 20
        self.start_time = time.time()
        self.last_update = self.start_time-2
        print("")
        self.update_progress()

    def update_exact(self, i):
        self.i = i-1
        self.update_progress()
        
    @classmethod
    def print(cls, text, row=1):
        #print(f"\033[{100}G {text}                                     ", end="")
        newlines = "\n"*row
        endlines = "\033[F"*row
        print(f"{newlines}{text}                                     {endlines}", end="")
        
    def update_progress(self):
        self.i += 1
        progress = self.i/self.count
        if self.last_update+1<time.time() or progress >= 1:
            self.last_update=time.time()
            block = int(round(self.barlength*progress))
            if progress == 0:
                seconds_remaining = -1
            else:
                seconds_remaining = int(((time.time() - self.start_time)/progress) * (1-progress))
            blocks = "█" * block + "░" * (self.barlength - block)
            text = f"\r{self.activity}: [{blocks}] {round(progress*100, 2)}%. {seconds_remaining} seconds remaining"
            lineEnd = '\r' if progress<1.0 else '\n\n'
            # print("                                                                                               ", end="\r")
            print(text, end=lineEnd)

    def __del__(self):
        print("")
        print(f"Done in {int(time.time()-self.start_time)}s")


if __name__ == "__main__":
    main()
