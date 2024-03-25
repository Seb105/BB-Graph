from math import pi, sqrt, atan2, cos, sin, radians
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import numpy as np
import os
import concurrent.futures
import time
import ujson as json
import sys
if os.name == 'nt': # Only if we are running on Windows
    from ctypes import windll
    k = windll.kernel32
    k.SetConsoleMode(k.GetStdHandle(-11), 7)
sys.stdout.write(f"\x1b[8;{64};{100}t") # Set console size to 100x64

RAD = 1.571
## All at 15c
AIR_DENSITY = 1.225
AIR_DYNAMIC_VISCOSITY = 1.802e-5
AIR_KINEMATIC_VISCOSITY = AIR_DYNAMIC_VISCOSITY/AIR_DENSITY
BB_DIAMETER = 5.95e-3
BB_RADIUS = BB_DIAMETER/2
SPHERE_DRAG_COEF = 0.40
SPHERE_FRONTAL_AREA = pi*BB_RADIUS**2
TIMESTEP = 1/1000
GRAVITY = 9.81
FEET_PER_METRE = 3.28084
INITIAL_POSITION = [0, 5/FEET_PER_METRE]
ENERGIES = (0.7, 0.8, 0.9, 1.0, 1.138, 1.486, 1.881, 2.322)
MASSES =    (0.0002, 0.00025, 0.00028, 0.0003, 
            0.00032, 0.00035, 0.0004, 0.00045, 0.0005)
CLAMP_AXES = True
USE_CACHE = True
RPM_PER_RAD_PER_SEC = 9.5493

GRAPH_DPI = 200
FIG_SIZE_MOD = 0.75

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
        # https://en.wikipedia.org/wiki/Reynolds_number
        surface_speed = BB_RADIUS * self.angular_velocity
        flow_speed = sqrt(self.velocity[0]**2 + self.velocity[1]**2) + surface_speed
        return (AIR_DENSITY*flow_speed*BB_DIAMETER)/AIR_DYNAMIC_VISCOSITY

    def drag_coef(self):
        # return 24/self.reynolds_number()#
        #http://www.ecourses.ou.edu/cgi-bin/ebook.cgi?topic=fl&chap_sec=09.1&page=theory
        sphere_drags = (
            (0, 0.42),
            (3e5, 0.39),
            (3.5e5, 0.1),
            (3.55e5, 0.09),
            (3.6e5, 0.08),

        )
        reynolds_number = self.reynolds_number()
        for i in range(len(sphere_drags)-1):
            prev_reynolds = sphere_drags[i][0]
            next_reynolds = sphere_drags[i+1][0]
            if prev_reynolds <= reynolds_number <= next_reynolds:
                prev_drag = sphere_drags[i][1]
                next_drag = sphere_drags[i+1][1]
                drag_coef = remap(reynolds_number, prev_reynolds, next_reynolds, prev_drag, next_drag)
                return drag_coef
        raise Exception(f"Reynolds number {reynolds_number} is out of range")
        
        

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
        # G = F/(BB_DIAMETER * AIR_DENSITY * component_velocity)
        # angular_velocity = G/(2*pi*BB_RADIUS**2)


        # F = (4/3) * (4*(pi**2)*(BB_RADIUS**3)*angular_velocity*AIR_DENSITY*speed)
        # solve for angular_velocity
        angular_velocity = (F*3)/(16*(pi**2)*(BB_RADIUS**3)*AIR_DENSITY*component_velocity)
        return angular_velocity

    def update_angular_velocity(self):
        # https://physics.stackexchange.com/questions/304742/angular-drag-on-body
        shear_force = 2*pi*AIR_DYNAMIC_VISCOSITY * \
            self.angular_velocity*BB_DIAMETER*BB_RADIUS
        shear_torque = BB_RADIUS * shear_force

        friction_torque = pi*self.angular_velocity**2*BB_RADIUS**4 * \
            BB_DIAMETER*AIR_DENSITY*SPHERE_DRAG_COEF


        total_torque = shear_torque + friction_torque
        total_acceleration = total_torque/self.moment_of_inerta
        total_delta = total_acceleration * TIMESTEP

        self.angular_velocity = self.angular_velocity - total_delta

    def update_angular_velocity_viscous(self):
        # https://hobbieroth.blogspot.com/2018/01/the-viscous-torque-on-rotating-sphere.html
        angular_velocity = self.angular_velocity
        torque = 8*pi*AIR_DYNAMIC_VISCOSITY*BB_RADIUS**3*angular_velocity
        angular_acceleration = torque/self.moment_of_inerta
        angular_delta = angular_acceleration * TIMESTEP

        self.angular_velocity = angular_velocity - angular_delta

    def calc_magnus_effect(self):
        speed = sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        # G = self.vortex_strength()
        direction = atan2(self.velocity[0], self.velocity[1]) - RAD
        # F_old = AIR_DENSITY*speed*G*BB_DIAMETER
        angular_velocity = self.angular_velocity
        # https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/ideal-lift-of-a-spinning-ball/
        F_ideal = (4/3) * (4*(pi**2)*(BB_RADIUS**3)*angular_velocity*AIR_DENSITY*speed)
        # https://www.grc.nasa.gov/www/k-12/VirtualAero/BottleRocket/airplane/beach.html
        # V_r = 2 * pi * BB_RADIUS * angular_velocity
        # G_c = 2 * pi * BB_RADIUS * V_r
        # F_corrected = (AIR_DENSITY * G_c * speed) * BB_DIAMETER * (pi/4)
        delta_speed = F_ideal/self.mass
        delta_X = sin(direction) * delta_speed
        delta_Y = cos(direction) * delta_speed
        # print("Magnus delta ", delta_X, delta_Y, velocity)
        return [delta_X, delta_Y]

    def calc_drag(self) -> list:
        speed = sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        direction = atan2(self.velocity[0], self.velocity[1])
        drag_coef = SPHERE_DRAG_COEF #self.drag_coef()
        # https://en.wikipedia.org/wiki/Drag_equation
        force = drag_coef*0.5*AIR_DENSITY*(speed**2)*SPHERE_FRONTAL_AREA
        # force = 6*pi*BB_RADIUS*AIR_DYNAMIC_VISCOSITY*speed # https://en.wikipedia.org/wiki/Stokes%27_law
        delta_speed = force/self.mass
        delta_X = sin(direction) * delta_speed
        delta_Y = cos(direction) * delta_speed
        return [-delta_X, -delta_Y]

    def update_position(self):
        self.position = [
            self.position[0] + self.velocity[0] * TIMESTEP, 
            self.position[1] + self.velocity[1] * TIMESTEP
        ]

    def update_velocity(self):
        acccelerations = [
            self.calc_magnus_effect(),
            self.calc_drag()
        ]
        acceleration = [0, 0]
        for delta in acccelerations:
            acceleration[0] += delta[0]
            acceleration[1] += delta[1]
        acceleration[1] += -GRAVITY  # Gravity
        velocity = self.velocity.copy()
        velocity[0] += acceleration[0] * TIMESTEP
        velocity[1] += acceleration[1] * TIMESTEP
        self.velocity = velocity
        return acceleration

    def run_sim(self) -> dict:
        points = []
        points.append((self.flight_time, self.position, self.velocity, [0, 0]))
        max_x = 0
        max_y = 0
        while self.position[1] > 0:
            self.update_angular_velocity_viscous()
            acceleration = self.update_velocity()
            self.update_position()
            max_x = max(max_x, self.position[0])
            max_y = max(max_y, self.position[1])
            self.flight_time += TIMESTEP
            points.append((self.flight_time, self.position, self.velocity, acceleration))
        result = {
            "mass": self.mass,
            "energy": self.initial_energy,
            "angle": round(self.initial_angle, 3),
            "hop_mod": round(self.hop_multiplier, 3),
            "rpm": round(self.initial_rpm),
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
    if subject == "J":
        label = f'{round(result["mass"]*1000, 2)}g'
    else:
        label = f'{round(result["energy"], 2)}J'
    return label + fps


def plot_graphs(series, datapoint, subject):
    fig = plt.figure(dpi=GRAPH_DPI, figsize=(20 * FIG_SIZE_MOD, 15 * FIG_SIZE_MOD), tight_layout=True)
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
    line.set_lw(2*FIG_SIZE_MOD)
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
        ax.set_xlim(xmin=0, xmax=310)
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
    if CLAMP_AXES:
        ax.set_xlim(xmin=0, xmax=1.8)
        ax.set_ylim(ymin=0, ymax=310)
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
        ax.set_xlim(xmin=0, xmax=1.8)
        ax.set_ylim(ymin=0, ymax=500)
    secyax = ax.secondary_yaxis('right', functions=(ft2m, m2ft))
    secyax.set_ylabel("Velocity (m/s)")
    ax.legend(loc='upper right')

def plot_spin_mods(results):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15*FIG_SIZE_MOD,10*FIG_SIZE_MOD))
    fig.tight_layout()
    fig.set_dpi(GRAPH_DPI)
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


# finds correct hop for max of 6ft rise
def run_bb_custom_hop(pair, row, max_ft = 6):
    mass, energy = pair
    angle = 0
    bb = BB_Class(mass, energy, angle)
    best_result = bb.run_sim()
    best_x = best_result["max_x"]
    hop_step = 0.001
    hop_multiplier = 1
    while True:
        Progress_Bar.print(f'{round(mass*1000, 2)}g, {energy}J, {angle}deg, {round(hop_multiplier, 2)}x {round(best_result["max_x"], 3)}m', row)
        result = BB_Class(mass, energy, angle, hop_multiplier).run_sim()
        max_x = result["max_x"]
        max_y = result["max_y"]
        if max_y > ft2m(max_ft):
            break
        if max_x > best_x:
            best_x = max_x
            best_result = result
        hop_multiplier = round(hop_multiplier + hop_step, 3)
    Progress_Bar.print(f'Done {mass*1000}g, {energy}j with {best_result["angle"]}deg and {best_result["hop_mod"]}x hop mod', row)
    return best_result

# Finds correct hop for max of 6ft rise but it can shoot downwards
def run_bb_1ft_hop_new(pair, row):
    mass, energy = pair
    best_result = BB_Class(mass, energy).run_sim()
    best_x = 0
    hop_step = 0.01
    hop_multiplier = 1
    maximum_height = ft2m(6)
    minimum_height = ft2m(4.5)
    angle = 0
    angle_step = -0.1
    angle_did_something = True
    while angle_did_something:
        angle_did_something = False
        while True:
            Progress_Bar.print(f'{round(mass*1000, 2)}g, {energy}J, {angle}deg, {round(hop_multiplier, 2)}x {round(best_result["max_x"], 3)}m', row)
            result = BB_Class(mass, energy, angle, hop_multiplier).run_sim()
            hop_multiplier = round(hop_multiplier + hop_step, 3)
            max_y = result["max_y"]
            if max_y > maximum_height:
                break
            points = result["points"]
            with_upwards_velocity = [x for x in points if x[3][1] >= 0]
            # If there are no points with upwards velocity, bb always fell so ignore this.
            # This causes a bug of 1x hop mod never being used, but this will never be the best hop mod anyway
            min_y = min(x[1][1] for x in with_upwards_velocity)
            # If it is dipping below minimum then further reducing the angle is useless
            if min_y < minimum_height:
                continue
            max_x = result["max_x"]
            if max_x > best_x:
                angle_did_something = True
                best_x = max_x
                best_result = result
            
        angle = round(angle + angle_step, 1)
    Progress_Bar.print(f'Done {mass*1000}g, {energy}j with {best_result["angle"]}deg and {best_result["hop_mod"]}x hop mod', row)
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
    for mass in MASSES:
        for energy in ENERGIES:
            pairs.append((mass, energy))
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=32) as pe:
        futures = []
        results = []
        if os.path.isfile("cache/results.json") and USE_CACHE:
            with open("cache/results.json", "r") as f:
                results = json.load(f)
        else:
            progress_bar = Progress_Bar("Running sims", len(pairs))
            for i, pair in enumerate(pairs):
                Progress_Bar.print(f'{pair} not started', i)
                futures.append(pe.submit(run_bb_custom_hop, pair, i))
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                progress_bar.update_progress()
            del(progress_bar)

            with open("cache/results.json", "w") as f:
                json.dump(results, f)
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
        pe.shutdown(wait=True)
    print("Done")

class Progress_Bar():
    def __init__(self, activity, count):
        self.activity = activity
        self.count = count
        self.i = 0
        self.barlength = 20
        self.start_time = time.time()
        self.last_update = self.start_time-2
        print("")
        self.print_bar()


    def update_exact(self, i):
        self.i = i-1
        self.update_progress()
        
    @classmethod
    def print(cls, text, row=0):
        row+=1
        newlines = "\033[s" + f"\033[{row}E"
        endlines = "\033[u"
        sys.stdout.flush()
        sys.stdout.write(f"{newlines}{text}      {endlines}")

    def print_bar(self):
        progress = self.i/self.count
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

    def update_progress(self):
        self.i += 1
        if time.time() - self.last_update > 1:
            self.print_bar()

    def __del__(self):
        print("")
        print(f"Done in {int(time.time()-self.start_time)}s")



if __name__ == "__main__":
    main()
