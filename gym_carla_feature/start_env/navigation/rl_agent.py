import carla
from gym_carla_feature.start_env.navigation.agent import Agent
from gym_carla_feature.start_env.navigation.local_planner import LocalPlanner
from gym_carla_feature.start_env.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
from gym_carla_feature.start_env.navigation.global_route_planner import GlobalRoutePlanner


class RLAgent(Agent):
    def __init__(self, vehicle):
        super(RLAgent, self).__init__(vehicle)



    def set_destination(self, start_location, end_location, clean=False):
        """
        This method creates a list of waypoints from navigation's position to destination location
        based on the route returned by the global router.

            :param start_location: initial position
            :param end_location: final position
            :param clean: boolean to clean the waypoint queue
        """
        self.start_waypoint = self._map.get_waypoint(start_location)
        self.end_waypoint = self._map.get_waypoint(end_location)
        route_trace = self._trace_route(self.start_waypoint, self.end_waypoint)


    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the
        optimal route from start_waypoint to end_waypoint.

            :param start_waypoint: initial position
            :param end_waypoint: final position
        """
        # Setting up global router
        if self._grp is None:
            wld = self.vehicle.get_world()
            dao = GlobalRoutePlannerDAO(
                wld.get_map(), sampling_resolution=self._sampling_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp
        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)
        return route
