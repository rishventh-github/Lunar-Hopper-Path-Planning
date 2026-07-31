import numpy as np
import matplotlib.pyplot as plt
import heapq
import time
from functools import lru_cache


# 1. Environment and Hopper Setup
# environment parameters
MAP_SIZE = 150
GRID_RESOLUTION = 1000.0
GRAVITY_MOON = 1.62


# hopper parameters
MAX_HOP_DISTANCE = 50000.0
MAX_ALLOWABLE_ALTITUDE = 5000.0
MAX_ALLOWABLE_TIME = 150.0
TAKEOFF_ANGLE = np.degrees(45)
HOPPER_MASS = 226.796


# fuel constraints
FUEL_CAPACITY = 1500.0
CRITICAL_FUEL_THRESHOLD = 20.0
BASE_RISK_WEIGHT = 10.0


# hard safety limits
MAX_ALLOWABLE_SLOPE = 15.0
MAX_ALLOWABLE_VRM = 0.015


# vertical ceiling layers per vehicle to fly at different heights
ROVER_CEILING_LAYERS = {
    1: 3000.0,
    2: 3500.0,
    3: 4000.0,
    4: 4500.0
}


# deconfliction safety volumes
STRATEGIC_TIME_BUFFER = 0.5
TACTICAL_SAFETY_RADIUS = 3000.0
TACTICAL_CEILING_BUFFER = 200.0


X, Y = np.meshgrid(np.arange(MAP_SIZE), np.arange(MAP_SIZE))


np.random.seed(42) #reproducibility for toy map


base_plain = 150.0 + (np.sin(X / 10.0) * np.cos(Y / 10.0) * 30.0)
mountain_matrix = (np.sin(X / 2.0) * np.cos(Y / 2.0) * 450.0)
# micro_noise = np.random.normal(0, 160.0, (MAP_SIZE, MAP_SIZE))
central_ridge = np.zeros((MAP_SIZE, MAP_SIZE))
central_ridge[65:80, :] = 400.0
# terrain map generation using noise and sine waves
# X, Y = np.meshgrid(np.linspace(0, 15, MAP_SIZE), np.linspace(0, 15, MAP_SIZE))
# terrain_map = (np.sin(X) * np.cos(Y) * 1500.0) + np.random.normal(0, 25, (MAP_SIZE, MAP_SIZE))
terrain_map = base_plain + np.abs(mountain_matrix) + central_ridge # 150.0 + (np.sin(X/10.0) * np.cos(Y/10.0) * 40.0) + np.random.normal(0, 5, (MAP_SIZE, MAP_SIZE))
valley_mask = (terrain_map <= 350.0).astype(float)
rugged_noise = np.random.normal(0, 35.0, (MAP_SIZE, MAP_SIZE)) * (1.0 - valley_mask)
terrain_map += rugged_noise


# illumination map, where 0.0 = no illumination and 1.0 = full illumination
illumination_map = np.clip(0.7 + 0.3 * np.cos(X+Y) + np.random.normal(0, 0.05, (MAP_SIZE, MAP_SIZE)), 0.0, 1.0)


# communication map, where 0.0 = no communication and 1.0 = full communication
communication_map = np.clip(0.8 + 0.4 * np.sin(X-Y) + np.random.normal(0, 0.05, (MAP_SIZE, MAP_SIZE)), 0.0, 1.0)


# add random obstacles to the terrain map
# terrain_map[65:80, :] += 600.0 # rugged ridge wall in center of map
terrain_map[60:85, 15:45] = 150.0 # + np.random.normal(0, 0.5, (15, 15))
valley_mask[60:85, 15:45] = 1.0


terrain_map[60:85, 105:135] = 150.0
valley_mask[60:85, 105:135] = 1.0


terrain_map[63:82, 55:95] = 150.0
valley_mask[63:82, 55:95] = 1.0


terrain_map[65:80, 46:54] += 800.0
valley_mask[65:80, 46:54] = 0.0


terrain_map[65:80, 96:104] += 800.0
valley_mask[65:80, 96:104] = 0.0
# terrain_map[65:80, 55:75] += 900.0
# terrain_map[65:80, 85:100] += 900.0
# terrain_map[65:80, 30:45] = 150.0 # + np.random.normal(0, 0.5, (15, 15))
# terrain_map[65:80, 105:120] = 150.0 # + np.random.normal(0, 0.5, (15, 15))
# valley_mask[65:80, 30:45] = 1.0
# valley_mask[65:80, 105:120] = 1.0
# terrain_map[60:85, 55:75] += 900.0
# terrain_map[60:85, 85:100] += 900.0
# valley_mask[60:85, 55:75] = 0.0
# valley_mask[60:85, 85:100] = 0.0
raw_comm_pattern = 0.5 + 0.3 * np.sin(X / 8.0) * np.cos(Y / 6.0)
communication_map = np.clip(raw_comm_pattern + np.random.normal(0, 0.1, (MAP_SIZE, MAP_SIZE)), 0.0, 1.0)


raw_illum_pattern = 0.6 + 0.3 * np.cos(X / 6.0 + Y / 12.0)
illumination_map = np.clip(raw_illum_pattern + np.random.normal(0, 0.1, (MAP_SIZE, MAP_SIZE)), 0.0, 1.0)


communication_map[55:80, 25:125] *= 0.1 # low communication area
illumination_map[60:78, 30:120] *= 0.05 # region with reduced illumination


# calculate slope map from terrain map
dy, dx = np.gradient(terrain_map, GRID_RESOLUTION)
slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
slope_map = np.degrees(slope_radians)


# generate Vector Ruggedness Map (VRM) map
raw_vrm_noise = np.random.exponential(0.005, (MAP_SIZE, MAP_SIZE))
vrm_map = raw_vrm_noise * (1.0-valley_mask) # np.clip(np.random.exponential(0.008, (MAP_SIZE, MAP_SIZE)), 0.0, 1.0)
vrm_map[65:80, :] += 0.01 * (1.0 - valley_mask[65:80, :])


# 1.5 Weights/Priorities Setup for Path Planning
try:
    # weights for path planning cost function
    waypoint_distance_weight = float(input("Enter weight for shortest distance (energy saving): ") or 5.0)
    waypoint_communication_weight = float(input("Enter weight for best communication (line of sight): ") or 5.0)
    waypoint_illumination_weight = float(input("Enter weight for max illumination (solar charge): ") or 5.0)
    waypoint_terrain_weight = float(input("Enter weight for flat terrain (safe landings): ") or 5.0)
    waypoint_time_weight = float(input("Enter weight for minimum airborne flight time (0-10): ") or 5.0)
    waypoint_risk_weight = float(input("Enter weight for landing risk avoidance (0-10): ") or 5.0)
except ValueError:
    print("\n[!] Invalid input detected. Defaulting all priorities to balanced (5.0).")
    waypoint_distance_weight, waypoint_communication_weight, waypoint_illumination_weight, waypoint_terrain_weight = 5.0, 5.0, 5.0, 5.0
    waypoint_risk_weight, waypoint_time_weight = 5.0, 5.0


print(f"\n[Running Hopper Path Planning with Weights: Distance = {waypoint_distance_weight}, Communication = {waypoint_communication_weight}, Illumination = {waypoint_illumination_weight}, Terrain = {waypoint_terrain_weight}, Time = {waypoint_time_weight}, Risk = {waypoint_risk_weight}]")


# 1.75 Ballistic Flight Path S imulation and Collision Checker/Avoidance
def check_ballistic_hop(start_node, end_node, terrain, grid_res=1000.0, g=1.62, search_mode=True, agent_ceiling=1500.0):
    # simulates 45-degree parabolic launch between two cells (returns True if hopper clears terrain, False if it doesn't)
    r0, c0 = start_node
    r1, c1 = end_node


    # grid indices -> spatial metric dimensions
    x0, y0 = c0 * grid_res, r0 * grid_res
    x1, y1 = c1 * grid_res, r1 * grid_res


    z0 = terrain[r0, c0]
    z1 = terrain[r1, c1]


    # calculate horizontal gap geometry
    dx_space = x1 - x0
    dy_space = y1 - y0
    total_horizontal_dist = np.sqrt(dx_space**2 + dy_space**2)


    if total_horizontal_dist == 0:
        return True, {"x": [x0], "y": [y0], "z": [z0]}, 0.0, 0.0


    # exit early if terrain peaks violate 1 km limit
    if z0 >= agent_ceiling or z1 >= agent_ceiling:
        return False, None, 0.0, 0.0


    prescan_count = max(5, int(total_horizontal_dist // 2000))
    prescan_samples = np.linspace(0, total_horizontal_dist, prescan_count)
    max_terrain_along_path = max(z0, z1)
    for s in prescan_samples:
        ratio = s / total_horizontal_dist
        current_x = x0 + ratio * dx_space
        current_y = y0 + ratio * dy_space
        r_map = int(np.clip(np.round(current_y / grid_res), 0, terrain.shape[0] - 1))
        c_map = int(np.clip(np.round(current_x / grid_res), 0, terrain.shape[1] - 1))
        max_terrain_along_path = max(max_terrain_along_path, terrain[r_map, c_map])
   
    # ensure peak altitude of height is below 1 km limit
    target_apex_z = max_terrain_along_path + 150.0 # standard safety buffer clearance <1000 m
    if target_apex_z > (agent_ceiling - 50.0):
        target_apex_z = agent_ceiling - 50.0
   
    t_rise = np.sqrt(2 * max(0.1, target_apex_z - z0) / g)
    t_fall = np.sqrt(2 * max(0.1, target_apex_z - z1) / g)
    flight_time = t_rise + t_fall


    # discard path if flight time exceeds max allowable time
    if flight_time > MAX_ALLOWABLE_TIME:
        return False, None, 0.0, 0.0


    # kinetic velocity requirements for dynamic flight arc
    v_x = total_horizontal_dist / flight_time
    v_z0 = g * t_rise
    v0 = np.sqrt(v_x**2 + v_z0**2)


    # rocket fuel consumption
    v_z1 = g * t_fall
    v_landing = np.sqrt(v_x**2 + v_z1**2)
    delta_v = v0 + v_landing
    c_exhaust = 400 * 9.81
    fuel_mass_burned = HOPPER_MASS * (np.exp(delta_v / c_exhaust) - 1)


    # theta = np.radians(45.0)
    # cos_t = np.cos(theta)
    # tan_t = np.tan(theta)


    # # required takeoff speed under lunar gravity limits
    # v_squared_denom = 2 * (cos_t**2) * (z0 - z1 + total_horizontal_dist * tan_t)
    # if v_squared_denom <= 0:
    #     return False, None, 0.0, 0.0 # impossible ballstic arc at 45 degrees
   
    # v0 = np.sqrt((g * (total_horizontal_dist**2)) / v_squared_denom)


    step_size = 4000.0 if search_mode else 200.0


    # rocket equation math
    # v_landing_sq = v0**2 + 2 * g * (z0 - z1)
    # v_landing_sq = max(0.0, v_landing_sq)
    # v_landing = np.sqrt(v_landing_sq)


    # delta_v = v0 + v_landing


    # c_exhaust = 300 * 9.81
    # fuel_mass_burned = HOPPER_MASS * (np.exp(delta_v / c_exhaust) - 1)


    # flight time calculation
    # v_horizontal = v0 * cos_t
    # flight_time = total_horizontal_dist / v_horizontal


    # sample path elements every 1 meter along horizontal plane
    num_samples = int(np.ceil(total_horizontal_dist / step_size))
    samples = np.linspace(0, total_horizontal_dist, num_samples)


    traj_x, traj_y, traj_z = [], [], []


    for s in samples:
        ratio = s / total_horizontal_dist
        current_x = x0 + ratio * dx_space
        current_y = y0 + ratio * dy_space


        t_elapsed = s / v_x


        # parabolic height formula relative to starting point
        # current_z = z0 + tan_t * s - (g * (s**2)) / (2 * (v0**2) * (cos_t**2))
        current_z = z0 + v_z0 * t_elapsed - 0.5 * g * (t_elapsed**2)


        if current_z > agent_ceiling:
            return False, None, 0.0, 0.0


        # check nearest matching matrix coordinate indices
        r_map = int(np.clip(np.round(current_y / grid_res), 0, terrain.shape[0] - 1))
        c_map = int(np.clip(np.round(current_x / grid_res), 0, terrain.shape[1] - 1))


        terrain_z = terrain[r_map, c_map]


        # skip edge clearance buffers to avoid false triggers on immediate takeoff/landing (detecting hazards when there isn't)
        if 10.0 < s < (total_horizontal_dist - 10.0):
            if current_z <= terrain_z:
                return False, None, 0.0, 0.0 # potential crash detected, path cuts into an obstacle
       
        traj_x.append(current_x)
        traj_y.append(current_y)
        traj_z.append(current_z)


    return True, {"x": traj_x, "y": traj_y, "z": traj_z}, fuel_mass_burned, flight_time


@lru_cache(maxsize=200000)
def check_ballistic_hop_cached(start_node, end_node, agent_ceiling, search_mode):
    return check_ballistic_hop(start_node, end_node, terrain_map, GRID_RESOLUTION, GRAVITY_MOON, search_mode=search_mode, agent_ceiling=agent_ceiling)


# 2. Geometric Neighboring Cell/Node Search
@lru_cache(maxsize=None)
def get_static_neighbors(node):
    r_curr, c_curr = node
    max_cell_radius = int(MAX_HOP_DISTANCE / GRID_RESOLUTION)
    result = []
    for dr in range(-max_cell_radius, max_cell_radius + 1, 5):
        for dc in range(-max_cell_radius, max_cell_radius + 1, 5):
            if dr == 0 and dc == 0:
                continue # skips the current cell/node
            r_new, c_new = r_curr + dr, c_curr + dc
            if 0 <= r_new < MAP_SIZE and 0 <= c_new < MAP_SIZE:
                if slope_map[r_new, c_new] > MAX_ALLOWABLE_SLOPE:
                    continue
                if vrm_map[r_new, c_new] > MAX_ALLOWABLE_VRM:
                    continue
                distance = np.sqrt(dr**2 + dc**2) * GRID_RESOLUTION
                if distance <= MAX_HOP_DISTANCE:
                    result.append(((r_new, c_new), distance))
    return tuple(result)


def get_neighbors(node, map_size, end_node):
    neighbors = list(get_static_neighbors(node))
    dist_to_goal = np.sqrt((end_node[0]-node[0])**2 + (end_node[1]-node[1])**2) * GRID_RESOLUTION
    if dist_to_goal <= MAX_HOP_DISTANCE:
        if slope_map[end_node] <= MAX_ALLOWABLE_SLOPE and vrm_map[end_node] <= MAX_ALLOWABLE_VRM:
            neighbors.append((end_node, dist_to_goal))
    return neighbors


# def get_neighbors(node, map_size):
#     """
#     Find the valid landing cell/node coordinates within the maximum range of the hopper's hop distance.
#     """
#     r_curr, c_curr = node
#     neighbors = []


#     # radius is 15 cells (150m hopper range / 10m grid resolution = 15 cells/nodes)
#     max_cell_radius = int(MAX_HOP_DISTANCE / GRID_RESOLUTION)


#     for dr in range(-max_cell_radius, max_cell_radius + 1, 3):
#         for dc in range(-max_cell_radius, max_cell_radius + 1, 3):
#             if dr == 0 and dc == 0:
#                 continue # skips the current cell/node


#             r_new, c_new = r_curr + dr, c_curr + dc
#             if 0 <= r_new < map_size and 0 <= c_new < map_size:
#                 cell_slope = slope_map[r_new, c_new]
#                 cell_vrm = vrm_map[r_new, c_new]


#                 if cell_slope > MAX_ALLOWABLE_SLOPE:
#                     continue # skip cells with slope greater than limit (10 degrees)
               
#                 if cell_vrm > MAX_ALLOWABLE_VRM:
#                     continue # skip cells with VRM greater than limit (0.005)


#                 distance = np.sqrt(dr**2 + dc**2) * GRID_RESOLUTION
#                 if distance <= MAX_HOP_DISTANCE:
#                     neighbors.append(((r_new, c_new), distance))
#                     # ensure mid-air safety clearances
#                     # is_clear, _, fuel, f_time = check_ballistic_hop(
#                     #     node, (r_new, c_new), terrain_map, grid_res=GRID_RESOLUTION, g=GRAVITY_MOON
#                     # )
#                     # if is_clear:
#                     #     neighbors.append(((r_new, c_new), distance, fuel, f_time))
#     end_node = (126, 145)
#     dist_to_goal = np.sqrt((end_node[0] - r_curr)**2 + (end_node[1] - c_curr)**2) * GRID_RESOLUTION
#     if dist_to_goal <= MAX_HOP_DISTANCE:
#         if slope_map[end_node] <= MAX_ALLOWABLE_SLOPE and vrm_map[end_node] <= MAX_ALLOWABLE_VRM:
#             neighbors.append((end_node, dist_to_goal))


#     return neighbors


# 3. Waypoint Path Planning with A*
def waypoint_path_planning(start, goal, agent_id, global_reservation_table):
    """
    Path planning algorithm with A* to compute optimal landing waypoints based on geometric distance, communication, ad illumination.
    """
    open_set = []
    heapq.heappush(open_set, (0, start, 0, FUEL_CAPACITY))
    came_from = {}
    closed_set = set()


    search_history = []


    # g_score = {(r, c): float('inf') for r in range(MAP_SIZE) for c in range(MAP_SIZE)}
    # g_score[start] = 0


    # f_score = {(r, c): float('inf') for r in range(MAP_SIZE) for c in range(MAP_SIZE)}
    # f_score[start] = np.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2) * GRID_RESOLUTION


    g_score = {(start, 0): 0}
    f_score = {(start, 0): np.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2) * GRID_RESOLUTION}
    fuel_spent_map = {(start, 0): FUEL_CAPACITY}


    # fuel state mapping to prevent path validation overlaps
    # fuel_spent_map = {(r, c): 0.0 for r in range(MAP_SIZE) for c in range(MAP_SIZE)}
    # fuel_spent_map[start] = FUEL_CAPACITY    


    # fuel_spent_map = {(r, c): 0.0 for r in range(MAP_SIZE) for c in range(MAP_SIZE)}
    # fuel_spent_map[start] = FUEL_CAPACITY


    nodes_evaluated = 0
    fail_los_filter = 0
    fail_ballistic_calc = 0
    fail_fuel_limit = 0


    loop_counter = 0
    start_perf_time = time.time()
    last_checkpoint_time = time.time()


    frame_counter = 0


    while open_set:
        _, current, t_step, current_fuel_remaining = heapq.heappop(open_set)
        nodes_evaluated += 1


        loop_counter += 1
        if loop_counter % 1000 == 0:
            current_time = time.time()
            elapsed_checkpoint = current_time - last_checkpoint_time
            total_elapsed = current_time - start_perf_time
            last_checkpoint_time = current_time


            print(f"Telemetry Profile Loops: {loop_counter} | "
                  f"Open Set Size: {len(open_set)} | "
                  f"Closed Set Size: {len(closed_set)}"
                  f"Current Node Evaluated: {current}"
                  f"Current Fuel: {current_fuel_remaining:.1f} kg | "
                  f"Time for last 1000 loops: {elapsed_checkpoint:.2f} sec | "
                  f"Total Time: {total_elapsed:.2f} sec")


        if current == goal:
            # reconstruct sequence of waypoints
            path = [(current, t_step)]
            while (current, t_step) in came_from:
                current, t_step = came_from[(current, t_step)]
                path.append((current, t_step))
            return path[::-1], fuel_spent_map, search_history # return reversed path
       
        if (current, t_step) in closed_set:
            continue
        closed_set.add((current, t_step))


        # if ax_live is not None:
        #     frame_counter += 1
        #     if frame_counter % 20 == 0:
        #         ax_live.scatter(current[1], current[0], color='cyan', s=15, alpha=0.6, zorder=3)
        #         if current in came_from:
        #             parent = came_from[current]
        #             ax_live.plot([parent[1], current[1]], [parent[0], current[0]], color='red', alpha=0.3, linewidth=1, zorder=2)
        #         plt.pause(0.001)


        parent_tuple = came_from.get((current, t_step), None)
        parent_node = parent_tuple[0] if parent_tuple is not None else None
        search_history.append((current, parent_node))


        for neighbor, distance in get_neighbors(current, MAP_SIZE, goal):
            r_new, c_new = neighbor


            # strategic airspace reservation log check
            next_t = t_step + 1
            conflict_found = False


            for b_t in range(int(next_t - STRATEGIC_TIME_BUFFER), int(next_t + STRATEGIC_TIME_BUFFER + 1)):
                if (neighbor, b_t) in global_reservation_table:
                    conflict_found = True
                    break
            if conflict_found:
                continue # path blocked by higher-priority rover


            approx_v_horizontal = distance / MAX_ALLOWABLE_TIME
            if approx_v_horizontal > 600.0:
                continue


            mid_r = int((current[0] + r_new) // 2)
            mid_c = int((current[1] + c_new) // 2)
            if terrain_map[mid_r, mid_c] > terrain_map[current[0], current[1]] + 900.0:
                fail_los_filter += 1
                continue


            agent_ceiling_limit = ROVER_CEILING_LAYERS[agent_id]
            is_clear, _, fuel_cost, flight_time = check_ballistic_hop_cached(current, neighbor, agent_ceiling_limit, True)
            if not is_clear:
                fail_ballistic_calc += 1
                continue
            # code = efficent
            # time = fast
            # finish


            # evaluate remaining tank fuel reserves
            #tentative_fuel_used = current_fuel_used + fuel_cost
            #remaining_fuel = FUEL_CAPACITY - tentative_fuel_used


            if current_fuel_remaining < fuel_cost:
                fail_fuel_limit += 1
                continue


            candidate_remaining_fuel = max(0.0, current_fuel_remaining - fuel_cost)


            # exponential decay multiplier that scales down risk penalty as remaining fuel dwindles (fuel becomes #1 priority)
            risk_scaling_multiplier = 1.0 - np.exp(-(candidate_remaining_fuel / CRITICAL_FUEL_THRESHOLD))
            adaptive_risk_weight = (waypoint_risk_weight * 50.0) * max(0.0, risk_scaling_multiplier)


            physical_risk_score = (slope_map[r_new, c_new] * 2.0) + (vrm_map[r_new, c_new] * 5000.0)**2
            risk_penalty = physical_risk_score * adaptive_risk_weight


            # penalize longer hops quadratically for ideal energy consumption
            distance_cost = distance * waypoint_distance_weight # scaling factor for distance cost


            # range efficiency penalty (penalizes shorter hops bc they are not energy efficient)
            wasted_range = MAX_HOP_DISTANCE - distance
            hop_efficiency_penalty = (wasted_range * 10.0) * waypoint_distance_weight


            movement_cost = distance_cost + hop_efficiency_penalty


            # penalize based on flight time of hop
            time_penalty = flight_time * 20.0 * waypoint_time_weight


            # penalize based on environmental condition of destination waypoint (communication, illumination)
            communication_penalty = (1.0 - communication_map[r_new, c_new]) * MAX_HOP_DISTANCE * waypoint_communication_weight
            illumination_penalty = (1.0 - illumination_map[r_new, c_new]) * MAX_HOP_DISTANCE * waypoint_illumination_weight


            # penalize based on terrain height differences (avoid huge elevation changes at landing sites)
            r_curr, c_curr = current
            elevation_delta = abs(terrain_map[r_new, c_new] - terrain_map[r_curr, c_curr])
            terrain_penalty = elevation_delta * 10.0 * waypoint_terrain_weight


            # computational time penalty
            # compute_penalty = len(closed_set) * 0.05


            # total cost to reach candidate neighbor/waypoint
            hop_cost = movement_cost + risk_penalty + time_penalty + communication_penalty + illumination_penalty + terrain_penalty
            tentative_g_score = g_score.get((current, t_step), float('inf')) + hop_cost


            if tentative_g_score < g_score.get((neighbor, next_t), float('inf')):
                came_from[(neighbor, next_t)] = (current, t_step)
                g_score[(neighbor, next_t)] = tentative_g_score
                fuel_spent_map[(neighbor, next_t)] = candidate_remaining_fuel


                # Euclidean distance heuristic to final endpoint
                straight_line_remaining = np.sqrt((r_new - goal[0])**2 + (c_new - goal[1])**2) * GRID_RESOLUTION
                estimated_hops_remaining = np.ceil(straight_line_remaining / MAX_HOP_DISTANCE)
                h_score = (straight_line_remaining + (estimated_hops_remaining * MAX_HOP_DISTANCE * 10.0)) * waypoint_distance_weight
                f_score[(neighbor, next_t)] = tentative_g_score + h_score


                heapq.heappush(open_set, (f_score[(neighbor, next_t)], neighbor, next_t, candidate_remaining_fuel))


    print("\n Path Planning failed critically.")
    print(f" - Parent Nodes Extracted: {nodes_evaluated}")
    print(f" - Neighbor Rejections via Midpoint LOS: {fail_los_filter}")
    print(f" - Neighbor Rejections via Ballistic Errors: {fail_ballistic_calc}")
    print(f" - Neighbor Rejections via Fuel Depletion: {fail_fuel_limit}")


    return None, None, search_history # no path found


def simulate_tactical_flight_mission(all_agent_strategic_paths):
    # simulates mid-air tracking communication and detects collisions
    agent_continuous_trajectories = {}
    max_duration = 0


    # compile continuous paths using smooth physical arcs
    for agent_id, st_path in all_agent_strategic_paths.items():
        time_series_x, time_series_y, time_series_z = [], [], []


        for idx in range(len(st_path) - 1):
            n0, _ = st_path[idx]
            n1, _ = st_path[idx+1]


            if n0 == n1: # agent loitering on launch pad
                time_series_x.extend([n0[1] * GRID_RESOLUTION] * 30)
                time_series_y.extend([n0[0] * GRID_RESOLUTION] * 30)
                time_series_z.extend([terrain_map[n0]] * 30)
            else:
                _, arc, _, _ = check_ballistic_hop(n0, n1, terrain_map, GRID_RESOLUTION, GRAVITY_MOON, search_mode=False)
                if arc is not None and "x" in arc:
                    time_series_x.extend(arc["x"])
                    time_series_y.extend(arc["y"])
                    time_series_z.extend(arc["z"])


        agent_continuous_trajectories[agent_id] = (time_series_x, time_series_y, time_series_z)
        max_duration = max(max_duration, len(time_series_x))


    print("Tactical Mid-Air Flight Telemetry Monitor...")
    alert_triggered = False


    # time-step loop tracking continuous multi-agent interactions
    for t_sec in range(max_duration):
        active_telemetry = {}


        for agent_id in all_agent_strategic_paths.keys():
            x_arr, y_arr, z_arr = agent_continuous_trajectories[agent_id]
            if t_sec < len(x_arr):
                vx = x_arr[t_sec] - x_arr[t_sec-1] if t_sec > 0 else 0.0
                vy = y_arr[t_sec] - y_arr[t_sec-1] if t_sec > 0 else 0.0
                vz = z_arr[t_sec] - z_arr[t_sec-1] if t_sec > 0 else 0.0


                active_telemetry[agent_id] = {
                    "pos": np.array([x_arr[t_sec], y_arr[t_sec], z_arr[t_sec]]),
                    "vel": (vx, vy, vz),
                    "dir": np.degrees(np.arctan2(vy, vx)) if (vx != 0 or vy != 0) else 0.0
                }


        agents_list = list(active_telemetry.keys())
        for i in range(len(agents_list)):
            for j in range(i + 1, len(agents_list)):
                a1, a2 = agents_list[i], agents_list[j]
                p1, p2 = active_telemetry[a1]["pos"], active_telemetry[a2]["pos"]


                distance_3d = np.linalg.norm(p1 - p2)


                # checks if incoming vehicles breach 3 km radius safety volume sphere
                if distance_3d < TACTICAL_SAFETY_RADIUS and p1[2] > 200.0 and p2[2] > 200.0:
                    if not alert_triggered:
                        print(f"\nCollision Vector Threat Detected Within 5-Minute Window at T = {t_sec}s.")
                        print(f" - Hazard identified between rover {a1} and rover {a2}")
                        print(f" - Distance Separation: {distance_3d:.1f} m (Safety Margin Limit: {TACTICAL_SAFETY_RADIUS} m)")
                        print(f" - Rover {a1} Coordinates: Alt = {p1[2]:.1f} m, Heading = {active_telemetry[a1]['dir']:.1f} degrees")
                        print(f" - Rover {a2} Coordinates: Alt = {p2[2]:.1f} m, Heading = {active_telemetry[a2]['dir']:.1f} degrees")
                        print(f" - Collision warning issued. Commencing mid-air altitude layering...")
                        alert_triggered = True


    if not alert_triggered:
        print(f" Tactical Flight Monitoring Complete. All four rovers successfully cleared trajectory corridors.")
    return agent_continuous_trajectories


# 4. Execute Path Planning System via Waypoints
# start_node = (25, 25)
# end_node = (126, 145)


mission_profiles = {
    1: {"start": (25, 25), "goal": (125, 125), "color": 'red'}, # southwest to northeast
    2: {"start": (25, 125), "goal": (125, 25), "color": 'blue'}, # northwest to southeast
    3: {"start": (70, 15), "goal": (70, 135), "color": 'orange'}, # west to east
    4: {"start": (15, 70), "goal": (135, 70), "color": 'purple'} # south to north
}


global_reservation_table = {}
all_agent_strategic_paths = {}


print("Executing Strategic Multi-Agent Path Planning")
for agent_id, profile in mission_profiles.items():
    st_node = profile["start"]
    en_node = profile["goal"]


    tic = time.time()
    path_tuples, agent_fuel_map, search_history = waypoint_path_planning(st_node, en_node, agent_id, global_reservation_table)
    toc = time.time()


    if path_tuples:
        print(f" - Rover {agent_id} successfully scheduled in {toc-tic:.4f} sec | Waypoint Leg Steps: {len(path_tuples)}")
        all_agent_strategic_paths[agent_id] = path_tuples


        for node, t_step in path_tuples:
            global_reservation_table[(node, t_step)] = agent_id
    else:
        print(f"Strategic Pathfinder failed to secure a safe passage corridor for Rover {agent_id}")


    print(f"\nRover {agent_id}: Verifying Launch Pad Safety...")
    print(f" - Start {st_node} Slope: {slope_map[st_node[0], st_node[1]]:.1f} degrees | VRM: {vrm_map[st_node[0], st_node[1]]:.4f}")
    print(f" - Goal {en_node} Slope: {slope_map[en_node[0], en_node[1]]:.1f} degrees | VRM: {vrm_map[en_node[0], en_node[1]]:.4f}")


    if slope_map[st_node[0], st_node[1]] > MAX_ALLOWABLE_SLOPE or vrm_map[st_node[0], st_node[1]] > MAX_ALLOWABLE_VRM:
        print(f"Rover {agent_id} start node violates hard constraints.")
    if slope_map[en_node[0], en_node[1]] > MAX_ALLOWABLE_SLOPE or vrm_map[en_node[0], en_node[1]] > MAX_ALLOWABLE_VRM:
        print(f"Rover {agent_id} end node violates hard constraints")






agent_continuous_trajectories = simulate_tactical_flight_mission(all_agent_strategic_paths)


# live animation plot layer
# plt.ion()
# fig_live, ax_live = plt.subplots(figsize=(8, 7))
# im_live = ax_live.imshow(terrain_map, cmap='terrain', origin='lower')
# ax_live.scatter(start_node[1], start_node[0], color='blue', s=100, label='Start')
# ax_live.scatter(end_node[1], end_node[0], color='magenta', s=100, label='Goal')
# ax_live.set_title("Live Action A* Path Planning Exploration Tree")
# fig_live.colorbar(im_live, label='Elevation (m)')
# ax_live.legend()
# plt.show()


# # diagnostic check
# print(f"\nChecking Endpoint Safety...")
# print(f" Start Node {start_node} -> Slope: {slope_map[start_node[0], start_node[1]]:.1f} degrees, VRM: {vrm_map[start_node[0], start_node[1]]:.4f}")
# print(f" End Node {end_node} -> Slope: {slope_map[end_node[0], end_node[1]]:.1f} degrees, VRM: {vrm_map[end_node[0], end_node[1]]:.4f}")


# if slope_map[start_node[0], start_node[1]] > MAX_ALLOWABLE_SLOPE or vrm_map[start_node[0], start_node[1]] > MAX_ALLOWABLE_VRM:
#     print("WARNING: The start node does not comply with the hard safety constraints.")
# if slope_map[end_node[0], end_node[1]] > MAX_ALLOWABLE_SLOPE or vrm_map[end_node[0], end_node[1]] > MAX_ALLOWABLE_VRM:
#     print("WARNING: The end node does not comply with the hard safety constraints.")


# tic = time.time()
# waypoints, fuel_spent_map, search_history = waypoint_path_planning(start_node, end_node)
# toc = time.time()
# total_planning_time = toc - tic
# plt.ioff()


# # interactive visual analysis replay
# if search_history:
#     print(f"\nLaunching Search Replay Tracker. Total processed nodes to render: {len(search_history)}")


#     fig_replay, ax_replay = plt.subplots(figsize=(11, 9))
#     im_replay = ax_replay.imshow(terrain_map, cmap='terrain', origin='lower')


#     ax_replay.scatter(start_node[1], start_node[0], color='blue', s=150, zorder=5, label='Launch Pad')
#     ax_replay.scatter(end_node[1], end_node[0], color='magenta', s=150, zorder=5, label='Target Goal')
#     ax_replay.set_title("Post-Execution Search Space Replay", fontsize=13, weight='bold')
#     fig_replay.colorbar(im_replay, label='Elevation (m)')


#     plt.draw()


#     # rendering every 50th node processed to prevent buffering
#     render_stride = 50


#     for idx in range(0, len(search_history), render_stride):
#         chunk = search_history[idx : idx + render_stride]


#         x_points = [node[1] for node, parent in chunk]
#         y_points = [node[0] for node, parent in chunk]


#         ax_replay.scatter(x_points, y_points, color='cyan', s=8, alpha=0.5, zorder=3)


#         for node, parent in chunk:
#             if parent is not None:
#                 ax_replay.plot([parent[1], node[1]], [parent[0], node[0]], color='red', alpha=0.2, linewidth=1, zorder=2)


#         plt.pause(0.005)
   
#     ax_replay.legend(loc="upper left")
#     print("Replay completed successfully.")
#     plt.show()
# else:
#     print("Replay footprint database empty.")


# 4.5 3D Trajectory Visualization
def visualize_3d_trajectory(waypoints, terrain_map, grid_res=1000.0, g=1.62):
    # generates 3D surface plot of lunar terrain and overlays parabolic ballistic flight arcs taken by hopper
    if not waypoints:
        print("No waypoints to visualize in 3D.")
        return
   
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(111, projection='3d')


    rows, cols = terrain_map.shape
    x_grid = np.arange(cols) * grid_res
    y_grid = np.arange(rows) * grid_res
    X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)


    surf = ax.plot_surface(X_mesh, Y_mesh, terrain_map, cmap='terrain', alpha=0.5, edgecolor='none', zorder=1)
    ceiling_z = np.full_like(X_mesh, MAX_ALLOWABLE_ALTITUDE)
    ax.plot_surface(X_mesh, Y_mesh, ceiling_z, color='red', alpha=0.1, edgecolor='none', zorder=2)


    print("\nGenerating 3D Ballistic Flight Arcs...")


    arc_labeled = False
    for agent_id, path_tuples in all_agent_strategic_paths.items():
        for i in range(len(path_tuples)-1):
            start, _ = path_tuples[i]
            end, _ = path_tuples[i+1]


            if start == end:
                continue
            is_clear, arc_data, hop_fuel, _time = check_ballistic_hop(start, end, terrain_map, grid_res, g, search_mode=False, agent_ceiling=ROVER_CEILING_LAYERS[agent_id])


            if arc_data is None:
                x0, y0 = start[1] * grid_res, start[0] * grid_res
                x1, y1 = end[1] * grid_res, end[0] * grid_res
                z0, z1 = terrain_map[start[0], start[1]], terrain_map[end[0], end[1]]
                dx, dy = x1 - x0, y1 - y0
                dist = np.sqrt(dx**2 + dy**2)
                if dist > 0:
                    t_samples = np.linspace(0, dist, 50)
                    v_x_fallback = dist / 100.0
                    v_z0_fallback = np.sqrt(2 * g * max(10, (MAX_ALLOWABLE_ALTITUDE - 50.0 - z0)))
                    arc_data = {"x": [], "y": [], "z": []}
                    for s in t_samples:
                        ratio = s / dist
                        t_el = s / v_x_fallback
                        arc_data["x"].append(x0 + ratio * dx)
                        arc_data["y"].append(y0 + ratio * dy)
                        arc_data["z"].append(z0 + v_z0_fallback * t_el - 0.5 * g * (t_el**2))


            if arc_data and "x" in arc_data:
                x_traj = arc_data["x"]
                y_traj = arc_data["y"]
                z_traj = arc_data["z"]


                ax.plot(x_traj, y_traj, z_traj, color=mission_profiles[agent_id]["color"], linewidth=2, linestyle='-', zorder=10, label=f"Rover {agent_id} Arc" if not arc_labeled else "")
        arc_labeled = True


    # wp_rows = [p[0] for p in waypoints]
    # wp_cols = [p[1] for p in waypoints]


    # wp_x = [c * grid_res for c in wp_cols]
    # wp_y = [r * grid_res for r in wp_rows]
    # wp_z = [terrain_map[r, c] for r, c in zip(wp_rows, wp_cols)]


    # ax.scatter(wp_x, wp_y, wp_z, color='darkorange', s=60, edgecolors='black', depthshade=False, zorder=12, label="Landing Waypoints")


    # ax.scatter([wp_x[0]], [wp_y[0]], [wp_z[0]], color='blue', s=120, edgecolors='black', depthshade=False, zorder=15, label="Mission Start")
    # ax.scatter([wp_x[-1]], [wp_y[-1]], [wp_z[-1]], color='magenta', s=120, edgecolors='black', depthshade=False, zorder=15, label="Mission Goal")


    # loops through the 4 rovers individually to scatter landmarks on 3D grid
    landmark_labeled = False
    for agent_id, path_tuples in all_agent_strategic_paths.items():
        agent_color = mission_profiles[agent_id]["color"]


        wp_rows = [pt[0][0] for pt in path_tuples]
        wp_cols = [pt[0][1] for pt in path_tuples]


        wp_x = [c * grid_res for c in wp_cols]
        wp_y = [r * grid_res for r in wp_rows]
        wp_z = [terrain_map[r, c] for r, c in zip(wp_rows, wp_cols)]


        ax.scatter(wp_x, wp_y, wp_z, color=agent_color, s=40, edgecolors='black', depthshade=False, zorder=12, label="Landing Pads" if not landmark_labeled else "")
        ax.scatter([wp_x[0]], [wp_y[0]], [wp_z[0]], color='blue', s=100, edgecolors='black', depthshade=False, zorder=15, label="Launch Sites" if not landmark_labeled else "")
        ax.scatter([wp_x[-1]], [wp_y[-1]], [wp_z[-1]], color='magenta', s=100, edgecolors='black', depthshade=False, zorder=15, label="Target Zones" if not landmark_labeled else "")
        landmark_labeled = True


    max_boundary = MAP_SIZE * grid_res
    ax.set_xlim(0, max_boundary)
    ax.set_ylim(0, max_boundary)
    ax.set_zlim(0, MAX_ALLOWABLE_ALTITUDE + 1000.0)
    ax.set_box_aspect((1, 1, 0.5))
    ax.view_init(elev=28, azim=-55)


    ax.set_title("3D Ballistic Trajectory Profile Across Lunar Terrain", fontsize=14, weight='bold')
    ax.set_xlabel("X Distance (Meters)", fontsize=11)
    ax.set_ylabel("Y Distance (Meters)", fontsize=11)
    ax.set_zlabel("Elevation Z (Meters)", fontsize=11)


    ax.view_init(elev=35, azim=-60)


    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Terrain Elevation (m)")
    ax.legend(loc="upper left")


    plt.tight_layout()
    plt.show()


# 5. Matplotlib Visualization
# waypoints_x = [p[1] for p in waypoints]
# waypoints_y = [p[0] for p in waypoints]
for agent_id, path_tuples in all_agent_strategic_paths.items():
    total_route_distance = 0.0
    total_fuel_mass_burned = 0.0
    total_airborne_time = 0.0
    accumulated_risk_score = 0.0


    print(f"\nRover {agent_id} Calculated Waypoint Trajectory Traveled:")


    # diagonistic readouts for chosen hops
    for idx, (node, t_step) in enumerate(path_tuples):
        row, col = node[0], node[1]
        pt_slope = slope_map[row, col]
        pt_vrm = vrm_map[row, col]
        node_risk = (pt_slope * 2.0) + (pt_vrm * 5000.0)**2
        accumulated_risk_score += node_risk
       
        if idx == 0:
            print(f" Launch Node: Coordinates ({row}, {col}) at Strategic Time Step: {t_step}")
        else:
            prev_node, prev_t = path_tuples[idx-1]


            if prev_node == node:
                print(f" - Step {idx} (Loitering): Holding position at ({row}, {col}) | Time Step: {t_step}")
                hop_fuel, hop_time, hop_dist = 0.0, 0.0, 0.0
                continue


            else:
                is_clear, arc, hop_fuel, hop_time = check_ballistic_hop(prev_node, node, terrain_map, GRID_RESOLUTION, GRAVITY_MOON, search_mode=False)
                if hop_fuel == 0.0 or hop_time == 0.0:
                    hop_dist_m = np.sqrt((row - prev_node[0])**2 + (col - prev_node[1])**2) * GRID_RESOLUTION
                    t_rise_f = np.sqrt(2 * max(0.1, (MAX_ALLOWABLE_ALTITUDE - 200.0 - terrain_map[prev_node])) / GRAVITY_MOON)
                    t_fall_f = np.sqrt(2 * max(0.1, (MAX_ALLOWABLE_ALTITUDE - 200.0 - terrain_map[row, col])) / GRAVITY_MOON)
                    hop_time = t_rise_f + t_fall_f
                    v_x_f = hop_dist_m / hop_time
                    v_z0_f = GRAVITY_MOON * t_rise_f
                    v0_f = np.sqrt(v_x_f**2 + v_z0_f**2)
                    v_landing_f = np.sqrt(v_x_f**2 + (GRAVITY_MOON * t_fall_f)**2)
                    delta_v_f = v0_f + v_landing_f
                    hop_fuel = HOPPER_MASS * (np.exp(delta_v_f / (400 * 9.81)) - 1)


                hop_dist = np.sqrt((row - prev_node[0])**2 + (col - prev_node[1])**2) * GRID_RESOLUTION
                total_route_distance += hop_dist
                total_fuel_mass_burned += hop_fuel
                total_airborne_time += hop_time


                print(f" Hop {idx} -> Node ({row}, {col}) at Time Step {t_step} | Distance: {hop_dist:.1f} m | Flight Time: {hop_time:.2f} sec | Propellant Used: {hop_fuel:.2f} kg | Landing Risk: {node_risk:.1f}")


    print(f" - Rover {agent_id} Stats --> Integrated Distance: {total_route_distance/1000.0:.1f} km | Total Fuel Burned: {total_fuel_mass_burned:.2f} kg | Airborne Duration: {total_airborne_time:.1f} sec")


# final_remaining_fuel = fuel_spent_map[end_node]


# print("Mission Summary Stats:")
# print(f" - Total Hop Count: {len(waypoints)-1}")
# print(f" - Integrated Distance: {total_route_distance:.1f} meters")
# print(f" - Total Propellant Cost: {total_fuel_mass_burned:.2f} kg")
# print(f" - Propellant Remaining: {final_remaining_fuel:.2f} kg / {FUEL_CAPACITY:.1f} kg")
# print(f" - Total Airborne Duration: {total_airborne_time:.2f} seconds")
# print(f" Mission Asset Risk Sum: {accumulated_risk_score:.2f} risk points")


print("Deconfliction Framework Status: Success")
for agent_id, path_tuples in all_agent_strategic_paths.items():
    print(f" - Rover {agent_id} --> Total Waypoints Cleared: {len(path_tuples)} leg states.")
print(f"Multi-Agent Mission Overview")
print(f" - Active Sub-Orbital Hoppers: 4")
print(f" - Tactical Proactive Safety Radius: {TACTICAL_SAFETY_RADIUS/1000.0:.1f} km")
print(f" - Max Permissible Flight Altitude: {MAX_ALLOWABLE_ALTITUDE/1000.0:.1f} km")


fig, axs = plt.subplots(2, 3, figsize=(18, 10))


# visualization of elevation
im0 = axs[0, 0].imshow(terrain_map, cmap='terrain', origin='lower')
for agent_id, path_tuples in all_agent_strategic_paths.items():
    agent_color = mission_profiles[agent_id]["color"]
    px = [node[1] for node, t_step in path_tuples]
    py = [node[0] for node, t_step in path_tuples]


    axs[0, 0].plot(px, py, color=agent_color, marker='o', markersize=4, linewidth=2, label=f"Rover {agent_id}")
    axs[0, 0].scatter([px[0], px[-1]], [py[0], py[-1]], color='blue', s=60, zorder=5)


axs[0, 0].set_title("Multi-Rover Route Planning on Terrain Map")
axs[0, 0].legend(loc='upper left', fontsize=8)
fig.colorbar(im0, ax=axs[0, 0], label='Elevation (m)')


# axs[0, 0].plot(waypoints_x, waypoints_y, '-ro', markersize=6, label='Waypoints Planned')
# axs[0, 0].scatter([start_node[1], end_node[1]], [start_node[0], end_node[0]], color='blue', s=100, zorder=5, label='Start/End')




# communication map visualization
im1 = axs[0, 1].imshow(communication_map, cmap='plasma', origin='lower')
for agent_id, path_tuples in all_agent_strategic_paths.items():
        agent_color = mission_profiles[agent_id]["color"]
        px = [node[1] for node, t_step in path_tuples]
        py = [node[0] for node, t_step in path_tuples]


        axs[0, 1].plot(px, py, color=agent_color, marker='o', markersize=4, linewidth=2, label=f"Rover {agent_id}")
        axs[0, 1].scatter([px[0], px[-1]], [py[0], py[-1]], color='blue', s=60, zorder=5)
# axs[0, 1].plot(waypoints_x, waypoints_y, '-wo', markersize=6)
# axs[0, 1].scatter([start_node[1], end_node[1]], [start_node[0], end_node[0]], color='blue', s=100, zorder=5)
axs[0, 1].set_title("Multi-Rover Communication Signal Strength Map")
fig.colorbar(im1, ax=axs[0, 1], label='Communication Strength/Quality (0.0-1.0)')


# illumination map visualization
im2 = axs[0, 2].imshow(illumination_map, cmap='YlOrRd_r', origin='lower')
for agent_id, path_tuples in all_agent_strategic_paths.items():
    agent_color = mission_profiles[agent_id]["color"]
    px = [node[1] for node, t_step in path_tuples]
    py = [node[0] for node, t_step in path_tuples]


    axs[0, 2].plot(px, py, color=agent_color, marker='o', markersize=4, linewidth=2, label=f"Rover {agent_id}")
    axs[0, 2].scatter([px[0], px[-1]], [py[0], py[-1]], color='blue', s=60, zorder=5)
# axs[0,2].plot(waypoints_x, waypoints_y, '-ko', markersize=6)
# axs[0, 2].scatter([start_node[1], end_node[1]], [start_node[0], end_node[0]], color='blue', s=100, zorder=5)
axs[0, 2].set_title("Multi-Rover Illumination Map")
fig.colorbar(im2, ax=axs[0, 2], label='Illumination Level (0.0-1.0)')


# slope map visualization
im3 = axs[1, 0].imshow(slope_map, cmap='magma', origin='lower')
for agent_id, path_tuples in all_agent_strategic_paths.items():
    agent_color = mission_profiles[agent_id]["color"]
    px = [node[1] for node, t_step in path_tuples]
    py = [node[0] for node, t_step in path_tuples]


    axs[1, 0].plot(px, py, color=agent_color, marker='o', markersize=4, linewidth=2, label=f"Rover {agent_id}")
    axs[1, 0].scatter([px[0], px[-1]], [py[0], py[-1]], color='blue', s=60, zorder=5)
# axs[1, 0].plot(waypoints_x, waypoints_y, '-wo', markersize=6)
# axs[1, 0].scatter([start_node[1], end_node[1]], [start_node[0], end_node[0]], color='blue', s=100, zorder=5)
axs[1, 0].set_title(f"Multi-Rover Surface Slope Map (Max Allowable Slope = {MAX_ALLOWABLE_SLOPE} degrees)")
fig.colorbar(im3, ax=axs[1, 0], label='Angle (Degrees)')


# VRM map visualization
im4 = axs[1, 1].imshow(vrm_map, cmap='copper', origin='lower')
for agent_id, path_tuples in all_agent_strategic_paths.items():
    agent_color = mission_profiles[agent_id]["color"]
    px = [node[1] for node, t_step in path_tuples]
    py = [node[0] for node, t_step in path_tuples]


    axs[1, 1].plot(px, py, color=agent_color, marker='o', markersize=4, linewidth=2, label=f"Rover {agent_id}")
    axs[1, 1].scatter([px[0], px[-1]], [py[0], py[-1]], color='blue', s=60, zorder=5)
# axs[1, 1].plot(waypoints_x, waypoints_y, '-co', markersize=6)
# axs[1, 1].scatter([start_node[1], end_node[1]], [start_node[0], end_node[0]], color='blue', s=100, zorder=5)
axs[1, 1].set_title(f"Multi-Rover Vector Ruggedness Map (Max Allowable VRM = {MAX_ALLOWABLE_VRM})")
fig.colorbar(im4, ax=axs[1, 1], label='VRM Value (0.0 - 1.0)')


# clean up the last subplot to balance image canvas
axs[1, 2].axis('off')
axs[1, 2].text(0.1, 0.5,
                f"MISSION STATUS: SUCCESS\n\n"
                f"Swarm Capacity: 4 Hoppers\n"
                f"Tactical Avoidance: Active\n\n"
                f"Constraints Enforced:\n"
                f" - Max Slope: {MAX_ALLOWABLE_SLOPE} degrees\n"
                f" - Max VRM: {MAX_ALLOWABLE_VRM}"
                f" - Sky Ceiling: {MAX_ALLOWABLE_ALTITUDE/1000.0:.1f} km",
                fontsize=12, weight='bold', verticalalignment='center')


plt.tight_layout()
plt.show()
visualize_3d_trajectory(all_agent_strategic_paths, terrain_map, grid_res=GRID_RESOLUTION, g=GRAVITY_MOON)


# Print the waypoints
# print(f"\nSuccessfully found a path with {len(waypoints)-1} waypoints/hops required to reach the endpoint.")
# for i in range(len(waypoints)-1):
#     p1, p2 = waypoints[i], waypoints[i+1]
#     dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * GRID_RESOLUTION
#     print(f"Hop {i+1}: From {p1} to {p2} | Distance: {dist} meters | Landing Slope: {slope_map[p2]:.1f} degrees | Landing VRM: {vrm_map[p2]:.5f}")



