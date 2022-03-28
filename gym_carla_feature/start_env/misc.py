def write_yaml(dict):
    import os
    from ruamel import yaml
    file_path = os.path.dirname(os.path.realpath(__file__))
    print('write in ', file_path)
    yamlpath = os.path.join(file_path, "carla_config.yaml")
    with open(yamlpath, 'w', encoding="utf-8") as f:
        yaml.dump(dict, f, Dumper=yaml.RoundTripDumper)


def read_yaml():
    import os
    import yaml
    yamlpath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "carla_config.yaml")
    context = open(yamlpath, 'r').read()
    dict = yaml.load(context, Loader=yaml.Loader)
    return dict


""" Module with auxiliary functions. """

import math
import numpy as np
import carla
import skimage
import pygame


def draw_waypoints(world, waypoints, z=0.5):
    """
    Draw a list of waypoints at a certain height given in z.

        :param world: carla.world object
        :param waypoints: list or iterable container with the waypoints to draw
        :param z: height in meters
    """
    for wpt in waypoints:
        wpt_t = wpt.transform
        begin = wpt_t.location + carla.Location(z=z)
        angle = math.radians(wpt_t.rotation.yaw)
        end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
        world.debug.draw_arrow(begin, end, arrow_size=0.3, life_time=1.0)


def get_speed(vehicle):
    """
    Compute speed of a vehicle in Km/h.

        :param vehicle: the vehicle for which speed is calculated
        :return: speed as a float in Km/h
    """
    vel = vehicle.get_velocity()

    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_transform: location of the target object
    :param current_transform: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x,
                              target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0


def is_within_distance(target_location, current_location, orientation, max_distance, d_angle_th_up, d_angle_th_low=0):
    """
    Check if a target object is within a certain distance from a reference object.
    A vehicle in front would be something around 0 deg, while one behind around 180 deg.

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :param max_distance: maximum allowed distance
        :param d_angle_th_up: upper thereshold for angle
        :param d_angle_th_low: low thereshold for angle (optional, default is 0)
        :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    forward_vector = np.array(
        [math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle_th_low < d_angle < d_angle_th_up


def compute_magnitude_angle(target_location, current_location, orientation):
    """
    Compute relative angle and distance between a target_location and a current_location

        :param target_location: location of the target object
        :param current_location: location of the reference object
        :param orientation: orientation of the reference object
        :return: a tuple composed by the distance to the object and the angle between both objects
    """
    target_vector = np.array([target_location.x - current_location.x, target_location.y - current_location.y])
    norm_target = np.linalg.norm(target_vector)

    forward_vector = np.array([math.cos(math.radians(orientation)), math.sin(math.radians(orientation))])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return (norm_target, d_angle)


def distance_vehicle(waypoint, vehicle_transform):
    """
    Returns the 2D distance from a waypoint to a vehicle

        :param waypoint: actual waypoint
        :param vehicle_transform: transform of the target vehicle
    """
    loc = vehicle_transform.location
    x = waypoint.transform.location.x - loc.x
    y = waypoint.transform.location.y - loc.y

    return math.sqrt(x * x + y * y)


def vector(location_1, location_2):
    """
    Returns the unit vector from location_1 to location_2

        :param location_1, location_2: carla.Location objects
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps

    return [x / norm, y / norm, z / norm]


def compute_distance(location_1, location_2):
    """
    Euclidean distance between 3D points

        :param location_1, location_2: 3D points
    """
    x = location_2.x - location_1.x
    y = location_2.y - location_1.y
    z = location_2.z - location_1.z
    norm = np.linalg.norm([x, y, z]) + np.finfo(float).eps
    return norm


def compute_angle(angle1, angle2):
    return min(abs(angle1 - angle2), abs(angle1 - (angle2 - 360)))


def positive(num):
    """
    Return the given number if positive, else 0

        :param num: value to check
    """
    return num if num > 0.0 else 0.0


def set_carla_transform(pose):
    """
    Get a carla transform object given pose.
    :param pose: list if size 3, indicating the wanted [x, y, yaw] of the transform
    :return: a carla transform object
    """
    transform = carla.Transform()
    transform.location.x = pose[0]
    transform.location.y = pose[1]
    transform.rotation.yaw = pose[2]
    return transform


def get_pos(vehicle):
    """
    Get the position of a vehicle
    :param vehicle: the vehicle whose position is to get
    :return: speed as a float in Kmh
    """
    trans = vehicle.get_transform()
    x = trans.location.x
    y = trans.location.y
    return x, y


def get_preview_lane_dis(waypts, x, y, idx=0):
    """
    Calculate distance from (x, y) to a certain waypoint
    :param waypt: certain waypoint like [[x0, y0,yawl0], [x1, y1,yawl1], ...]
    :param x: x position of vehicle
    :param y: y position of vehicle
    :param idx: index of the waypoint to which the distance is calculated
    :return: a tuple of the distance and the waypoint orientation unit vector
    """
    waypt = waypts[idx]
    vec = np.array([x - waypt[0], y - waypt[1]])
    lv = np.linalg.norm(np.array(vec))
    # yaw1 = waypt[2] % 360
    # yaw2 = waypts[idx - 1][2] % 360
    # if abs(yaw1 - yaw2) >= 180:
    #     if yaw1 > yaw2:
    #         yaw1 = yaw1 - 360
    #     else:
    #         yaw2 = yaw2 - 360
    # yaw = (yaw1 + yaw2) / 2 if idx > 0 else waypt[2]
    # yaw = (waypt[2] +  waypts[idx - 1][2]) / 2 if idx > 0 else waypt[2]
    yaw = waypt[2]
    w = np.array([np.cos(yaw / 180 * np.pi), np.sin(yaw / 180 * np.pi)])
    cross = np.cross(w, vec / lv)
    dis = - lv * cross
    return dis, w


def display_to_rgb(display, obs_size):
    """
    Transform image grabbed from pygame display to an rgb image uint8 matrix
    :param display: pygame display input
    :param obs_size: rgb image size
    :return: rgb image uint8 matrix
    """
    rgb = np.fliplr(np.rot90(display, 3))  # flip to regular view
    rgb = skimage.transform.resize(rgb, (obs_size, obs_size))  # resize
    rgb = rgb * 255
    return rgb


def rgb_to_display_surface(rgb, display_size):
    """
    Generate pygame surface given an rgb image uint8 matrix
    :param rgb: rgb image uint8 matrix
    :param display_size: display size
    :return: pygame surface
    """
    surface = pygame.Surface((display_size, display_size)).convert()
    display = skimage.transform.resize(rgb, (display_size, display_size))
    display = np.flip(display, axis=1)
    display = np.rot90(display, 1)
    pygame.surfarray.blit_array(surface, display)
    return surface


def is_within_distance_ahead(target_transform, current_transform, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_transform: location of the target object
    :param current_transform: location of the reference object
    :param orientation: orientation of the reference object
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_transform.location.x - current_transform.location.x,
                              target_transform.location.y - current_transform.location.y])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True

    if norm_target > max_distance:
        return False

    fwd = current_transform.get_forward_vector()
    forward_vector = np.array([fwd.x, fwd.y])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))

    return d_angle < 90.0


def vec_is_within_distance_ahead(target_vec, current_vec, max_distance):
    """
    Check if a target object is within a certain distance in front of a reference object.

    :param target_vec: [x,y,yaw]
    :param current_vec: [x,y,yaw]
    :param max_distance: maximum allowed distance
    :return: True if target object is within max_distance ahead of the reference object
    """
    target_vector = np.array([target_vec[0] - current_vec[0],
                              target_vec[1] - current_vec[1]])
    norm_target = np.linalg.norm(target_vector)

    # If the vector is too short, we can simply stop here
    if norm_target < 0.001:
        return True
    if norm_target > max_distance:
        return False
    forward_vector = np.array([np.cos(current_vec[2] / 180 * np.pi), np.sin(current_vec[2] / 180 * np.pi)])
    d_angle = math.degrees(math.acos(np.clip(np.dot(forward_vector, target_vector) / norm_target, -1., 1.)))
    return d_angle < 90.0
