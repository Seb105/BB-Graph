from math import pi, sqrt, atan2, cos, sin

RAD = 1.571
AIR_DENSITY = 1.225
BB_DIAMETER = 6e-3
BB_RADIUS = BB_DIAMETER/2
BB_JOULE = 1
INITIAL_POSITION = [0, 1.8]
SPHERE_DRAG_COEF = 0.47
SPHERE_FRONTAL_AREA = pi*BB_RADIUS**2
TIMESTEP = 1/100
GRAVITY = 9.81

def calc_vortex_strength(angular_velocity: float) -> float:
    return 2*pi*BB_RADIUS**2*angular_velocity*1.2


def get_intiial_velocity(mass: float) -> float:
    # calculates velocity according to mass and energy
    return [sqrt(BB_JOULE/(0.5*mass)), 0]


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

def update_velocity(velocity, deltas):
    for delta in deltas:
        velocity[0] += delta[0]*TIMESTEP
        velocity[1] += delta[1]*TIMESTEP
    velocity[1] += -GRAVITY * TIMESTEP # Gravity
    return velocity


def calc_bb(mass: float):
    velocity = get_intiial_velocity(mass)
    position = INITIAL_POSITION.copy()
    angular_velocity = get_initial_hop_angular_velocity(velocity[0], mass)
    flight_time = 0
    results = []
    while position[1] > 0:
        delta_magnus = calc_magnus_effect(velocity, mass, angular_velocity)
        delta_drag = calc_drag(velocity, mass)
        velocity = update_velocity(velocity, [delta_magnus, delta_drag])
        position = update_position(position, velocity)
        flight_time += TIMESTEP
        results.append([flight_time, position, velocity])
    #[print(result) for result in results]
    return [mass, flight_time, position[0]]


def main():
    for mass in (0.0002, 0.00025, 0.00028, 0.0003, 0.0004):
        print(calc_bb(mass))


main()
