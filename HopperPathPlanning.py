import numpy as np
import matplotlib.pyplot as plt
import heapq


# 1. Environment and Hopper Setup
# environment parameters
MAP_SIZE = 50
GRID_RESOLUTION = 10.0
GRAVITY_MOON = 1.62


# hopper parameters
MAX_HOP_DISTANCE = 150.0
TAKEOFF_ANGLE = np.degrees(45)
HOPPER_MASS = 15.0


# hard safety limits
MAX_ALLOWABLE_SLOPE = 10.0
MAX_ALLOWABLE_VRM = 0.005


np.random.seed(42) #reproducibility for toy map


# terrain map generation using noise and sine waves
X, Y = np.meshgrid(np.linspace(0, 5, MAP_SIZE), np.linspace(0, 5, MAP_SIZE))
terrain_map = (np.sin(X) * np.cos(Y) * 30.0) + np.random.normal(0, 2, (MAP_SIZE, MAP_SIZE))


# illumination map, where 0.0 = no illumination and 1.0 = full illumination
illumination_map = np.clip(0.7 + 0.3 * np.cos(X+Y) + np.random.normal(0, 0.05, (MAP_SIZE, MAP_SIZE)), 0.0, 1.0)


# communication map, where 0.0 = no communication and 1.0 = full communication
communication_map = np.clip(0.8 + 0.4 * np.sin(X-Y) + np.random.normal(0, 0.05, (MAP_SIZE, MAP_SIZE)), 0.0, 1.0)


# add random obstacles to the terrain map
terrain_map[20:25, 10:40] += 50.0 # a hill
communication_map[18:27, 8:42] *= 0.1 # low communication area
illumination_map[20:26, 10:40] *= 0.05 # region with reduced illumination


# calculate slope map from terrain map
dy, dx = np.gradient(terrain_map, GRID_RESOLUTION)
slope_radians = np.arctan(np.sqrt(dx**2 + dy**2))
slope_map = np.degrees(slope_radians)


# generate Vector Ruggedness Map (VRM) map
vrm_map = np.clip(np.random.exponential(0.0008, (MAP_SIZE, MAP_SIZE)), 0.0, 1.0)
vrm_map[18:27, 8:42] += 0.0043


# 1.5 Weights/Priorities Setup for Path Planning
try:
    # weights for path planning cost function
    waypoint_distance_weight = float(input("Enter weight for shortest distance (energy saving): ") or 5.0)
    waypoint_communication_weight = float(input("Enter weight for best communication (line of sight): ") or 5.0)
    waypoint_illumination_weight = float(input("Enter weight for max illumination (solar charge): ") or 5.0)
    waypoint_terrain_weight = float(input("Enter weight for flat terrain (safe landings): ") or 5.0)
except ValueError:
    print("\n[!] Invalid input detected. Defaulting all priorities to balanced (5.0).")
    waypoint_distance_weight, waypoint_communication_weight, waypoint_illumination_weight, waypoint_terrain_weight = 5.0, 5.0, 5.0, 5.0


print(f"\n[Running Hopper Path Planning with Weights: Distance = {waypoint_distance_weight}, Communication = {waypoint_communication_weight}, Illumination = {waypoint_illumination_weight}, Terrain = {waypoint_terrain_weight}]")


# 2. Geometric Neighboring Cell/Node Search
def get_neighbors(node, map_size):
    """
    Find the valid landing cell/node coordinates within the maximum range of the hopper's hop distance.
    """
    r_curr, c_curr = node
    neighbors = []


    # radius is 15 cells (150m hopper range / 10m grid resolution = 15 cells/nodes)
    max_cell_radius = int(MAX_HOP_DISTANCE / GRID_RESOLUTION)


    for dr in range(-max_cell_radius, max_cell_radius + 1):
        for dc in range(-max_cell_radius, max_cell_radius + 1):
            if dr == 0 and dc == 0:
                continue # skips the current cell/node


            r_new, c_new = r_curr + dr, c_curr + dc
            if 0 <= r_new < map_size and 0 <= c_new < map_size:
                cell_slope = slope_map[r_new, c_new]
                cell_vrm = vrm_map[r_new, c_new]


                if cell_slope > MAX_ALLOWABLE_SLOPE:
                    continue # skip cells with slope greater than limit (10 degrees)
               
                if cell_vrm > MAX_ALLOWABLE_VRM:
                    continue # skip cells with VRM greater than limit (0.005)


                distance = np.sqrt(dr**2 + dc**2) * GRID_RESOLUTION
                if distance <= MAX_HOP_DISTANCE:
                    neighbors.append(((r_new, c_new), distance))
           
    return neighbors


# 3. Waypoint Path Planning with A*
def waypoint_path_planning(start, goal):
    """
    Path planning algorithm with A* to compute optimal landing waypoints based on geometric distance, communication, ad illumination.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    closed_set = set()


    g_score = {(x, y): float('inf') for x in range(MAP_SIZE) for y in range(MAP_SIZE)}
    g_score[start] = 0


    f_score = {(x, y): float('inf') for x in range(MAP_SIZE) for y in range(MAP_SIZE)}
    f_score[start] = np.sqrt((start[0] - goal[0])**2 + (start[1] - goal[1])**2) * GRID_RESOLUTION


    while open_set:
        _, current = heapq.heappop(open_set)


        if current == goal:
            # reconstruct sequence of waypoints
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1] # return reversed path
       
        if current in closed_set:
            continue
        closed_set.add(current)


        for neighbor, distance in get_neighbors(current, MAP_SIZE):
            r_new, c_new = neighbor


            # penalize longer hops quadratically for ideal energy consumption
            distance_cost = distance * waypoint_distance_weight # scaling factor for distance cost


            # range efficiency penalty (penalizes shorter hops bc they are not energy efficient)
            wasted_range = MAX_HOP_DISTANCE - distance
            hop_efficiency_penalty = (wasted_range * 10.0) * waypoint_distance_weight


            movement_cost = distance_cost + hop_efficiency_penalty


            # penalize based on environmental condition of destination waypoint (communication, illumination)
            communication_penalty = (1.0 - communication_map[r_new, c_new]) * 150.0 * waypoint_communication_weight
            illumination_penalty = (1.0 - illumination_map[r_new, c_new]) * 150.0 * waypoint_illumination_weight


            # penalize based on terrain height differences (avoid huge elevation changes at landing sites)
            elevation_delta = abs(terrain_map[r_new, c_new] - terrain_map[current[0], current[1]])
            terrain_penalty = elevation_delta * 4.0 * waypoint_terrain_weight


            # total cost to reach candidate neighbor/waypoint
            hop_cost = movement_cost + communication_penalty + illumination_penalty + terrain_penalty
            tentative_g_score = g_score[current] + hop_cost


            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score


                # Euclidean distance heuristic to final endpoint
                straight_line_remaining = np.sqrt((r_new - goal[0])**2 + (c_new - goal[1])**2) * GRID_RESOLUTION
                estimated_hops_remaining = np.ceil(straight_line_remaining / MAX_HOP_DISTANCE)
                h_score = (straight_line_remaining + (estimated_hops_remaining * MAX_HOP_DISTANCE * 10.0)) * waypoint_distance_weight
                f_score[neighbor] = tentative_g_score + h_score


                heapq.heappush(open_set, (f_score[neighbor], neighbor))


    return None # no path found


# 4. Execute Path Planning System via Waypoints
start_node = (9, 2)
end_node = (41, 37)


# diagnostic check
print(f"\nChecking Endpoint Safety...")
print(f" Start Node {start_node} -> Slope: {slope_map[start_node[0], start_node[1]]:.1f} degrees, VRM: {vrm_map[start_node[0], start_node[1]]:.4f}")
print(f" End Node {end_node} -> Slope: {slope_map[end_node[0], end_node[1]]:.1f} degrees, VRM: {vrm_map[end_node[0], end_node[1]]:.4f}")


if slope_map[start_node[0], start_node[1]] > MAX_ALLOWABLE_SLOPE or vrm_map[start_node[0], start_node[1]] > MAX_ALLOWABLE_VRM:
    print("WARNING: The start node does not comply with the hard safety constraints.")
if slope_map[end_node[0], end_node[1]] > MAX_ALLOWABLE_SLOPE or vrm_map[end_node[0], end_node[1]] > MAX_ALLOWABLE_VRM:
    print("WARNING: The end node does not comply with the hard safety constraints.")


waypoints = waypoint_path_planning(start_node, end_node)


# 5. Matplotlib Visualization
if waypoints:
    waypoints_x = [p[1] for p in waypoints]
    waypoints_y = [p[0] for p in waypoints]


    # diagonistic readouts for chosen hops
    print("\nCalculated Waypoint Trajectory Traveled:")
    for idx, pt in enumerate(waypoints):
        row, col = pt[0], pt[1]
        pt_slope = slope_map[row, col]
        pt_vrm = vrm_map[row, col]
        print(f" Hop {idx}: Coordinates ({row}, {col}) -> Slope: {pt_slope:.1f} degrees, VRM: {pt_vrm:.4f}")


    fig, axs = plt.subplots(2, 3, figsize=(18, 10))


    # visualization of elevation
    im0 = axs[0, 0].imshow(terrain_map, cmap='terrain', origin='lower')
    axs[0, 0].plot(waypoints_x, waypoints_y, '-ro', markersize=6, label='Waypoints Planned')
    axs[0, 0].scatter([start_node[1], end_node[1]], [start_node[0], end_node[0]], color='blue', s=100, zorder=5, label='Start/End')
    axs[0, 0].set_title("Waypoint Route Planning on Terrain Map")
    axs[0, 0].legend()
    fig.colorbar(im0, ax=axs[0, 0], label='Elevation (m)')


    # communication map visualization
    im1 = axs[0, 1].imshow(communication_map, cmap='plasma', origin='lower')
    axs[0, 1].plot(waypoints_x, waypoints_y, '-wo', markersize=6)
    axs[0, 1].scatter([start_node[1], end_node[1]], [start_node[0], end_node[0]], color='blue', s=100, zorder=5)
    axs[0, 1].set_title("Communication Signal Strength Map")
    fig.colorbar(im1, ax=axs[0, 1], label='Communication Strength/Quality (0.0-1.0)')


    # illumination map visualization
    im2 = axs[0, 2].imshow(illumination_map, cmap='YlOrRd_r', origin='lower')
    axs[0,2].plot(waypoints_x, waypoints_y, '-ko', markersize=6)
    axs[0, 2].scatter([start_node[1], end_node[1]], [start_node[0], end_node[0]], color='blue', s=100, zorder=5)
    axs[0, 2].set_title("Illumination Map")
    fig.colorbar(im2, ax=axs[0, 2], label='Illumination Level (0.0-1.0)')
   
    # slope map visualization
    im3 = axs[1, 0].imshow(slope_map, cmap='magma', origin='lower')
    axs[1, 0].plot(waypoints_x, waypoints_y, '-wo', markersize=6)
    axs[1, 0].scatter([start_node[1], end_node[1]], [start_node[0], end_node[0]], color='blue', s=100, zorder=5)
    axs[1, 0].set_title(f"Surface Slope Map (Max Allowable Slope = {MAX_ALLOWABLE_SLOPE} degrees)")
    fig.colorbar(im3, ax=axs[1, 0], label='Angle (Degrees)')


    # VRM map visualization
    im4 = axs[1, 1].imshow(vrm_map, cmap='copper', origin='lower')
    axs[1, 1].plot(waypoints_x, waypoints_y, '-co', markersize=6)
    axs[1, 1].scatter([start_node[1], end_node[1]], [start_node[0], end_node[0]], color='blue', s=100, zorder=5)
    axs[1, 1].set_title(f"Vector Ruggedness Map (Max Allowable VRM = {MAX_ALLOWABLE_VRM})")
    fig.colorbar(im4, ax=axs[1, 1], label='VRM Value (0.0 - 1.0)')


    # clean up the last subplot to balance image canvas
    axs[1, 2].axis('off')
    axs[1, 2].text(0.1, 0.5,
                   f"MISSION STATUS: SUCCESS\n\n"
                   f"Total Waypoints: {len(waypoints)}\n"
                   f"Total Hops: {len(waypoints)-1}\n\n"
                   f"Constraints Enforced:\n"
                   f" - Max Slope: {MAX_ALLOWABLE_SLOPE} degrees\n"
                   f" - Max VRM: {MAX_ALLOWABLE_VRM}",
                   fontsize=12, weight='bold', verticalalignment='center')


    plt.tight_layout()
    plt.show()


    # Print the waypoints
    print(f"\nSuccessfully found a path with {len(waypoints)-1} waypoints/hops required to reach the endpoint.")
    for i in range(len(waypoints)-1):
        p1, p2 = waypoints[i], waypoints[i+1]
        dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) * GRID_RESOLUTION
        print(f"Hop {i+1}: From {p1} to {p2} | Distance: {dist} meters | Landing Slope: {slope_map[p2]:.1f} degrees | Landing VRM: {vrm_map[p2]:.5f}")
   
else:
    print("No valid path found from start to end node.")

