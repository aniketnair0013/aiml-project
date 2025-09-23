import sys
import os
src_dir = os.path.dirname(os.path.abspath(__file__))  
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
import argparse
import time 
import matplotlib.pyplot as plt
import numpy as np
from grid import GridEnvironment
from planners import Planner

def main():
    parser = argparse.ArgumentParser(description='Autonomous Delivery Agent Path Planner')
    parser.add_argument('--planner', required=True, choices=['bfs', 'ucs', 'astar', 'sa'],
                        help='Planner type: bfs, ucs, astar, or sa')
    parser.add_argument('--map', required=True,
                        help='Map name (e.g., small, medium, large, dynamic)')
    parser.add_argument('--replan', action='store_true',
                        help='Enable replanning simulation for dynamic maps')
    parser.add_argument('--plot', action='store_true',
                        help='Generate and display path plot')
    args = parser.parse_args()

    # Load map and dynamic file if applicable (root-relative paths)
    map_file = f'maps/{args.map}.map'
    dyn_file = f'maps/{args.map}.dyn' if args.map == 'dynamic' else None
    if not os.path.exists(map_file):
        print(f"Error: Map file {map_file} not found. Available: small, medium, large, dynamic.")
        print("Ensure you're running from project root (D:\\autonomous-delivery-agent).")
        sys.exit(1)

    try:
        env = GridEnvironment(map_file, dyn_file)
    except ValueError as e:
        print(f"Error loading environment: {e}")
        sys.exit(1)

    planner = Planner(env)
    planner_name = args.planner.upper()

    print(f"Loading map: {map_file}{' with dynamics' if dyn_file else ''}")
    print(f"Start: {env.start}, Goal: {env.goal}")
    print(f"Planner: {planner_name}")

    # Run the selected planner with timing
    start_time = time.time()  # Now safe: time imported above
    if args.planner == 'bfs':
        path, metrics = planner.bfs(env.start)
    elif args.planner == 'ucs':
        path, metrics = planner.ucs(env.start)
    elif args.planner == 'astar':
        path, metrics = planner.astar(env.start)
    elif args.planner == 'sa':
        path, metrics = planner.simulated_annealing(env.start)
    end_time = time.time()
    # Ensure metrics dict has all keys (fallback if planner doesn't set them)
    if 'time' not in metrics:
        metrics['time'] = end_time - start_time
    if 'cost' not in metrics:
        metrics['cost'] = float('inf') if not path else sum(env.get_cost(path[i-1], path[i]) for i in range(1, len(path)))
    if 'nodes' not in metrics:
        metrics['nodes'] = 0

    if path:
        print(f"Initial Path found: {path[:5]}... (length {len(path)})")
        print(f"Metrics: Cost={metrics['cost']}, Nodes Expanded={metrics['nodes']}, Time={metrics['time']:.3f}s")

        # Replanning simulation for dynamic maps
        if args.replan and args.map == 'dynamic' and hasattr(env, 'moving_obstacles') and env.moving_obstacles:
            print("Starting replanning simulation...")
            total_cost = metrics['cost']
            replan_count = 0
            current_path = path[:]
            log_entries = []

            for t in range(1, len(current_path)):
                pos = current_path[t]
                if env.is_occupied(pos, t):
                    print(f"Blocked at step {t} (pos {pos}) by dynamic obstacle! Replanning...")
                    # Replan from previous position
                    prev_pos = current_path[t-1]
                    sub_start_time = time.time()
                    if args.planner == 'bfs':
                        subpath, sub_metrics = planner.bfs(prev_pos)
                    elif args.planner == 'ucs':
                        subpath, sub_metrics = planner.ucs(prev_pos)
                    elif args.planner == 'astar':
                        subpath, sub_metrics = planner.astar(prev_pos)
                    elif args.planner == 'sa':
                        subpath, sub_metrics = planner.simulated_annealing(prev_pos)
                    sub_end_time = time.time()
                    # Fallback for sub_metrics
                    if 'time' not in sub_metrics:
                        sub_metrics['time'] = sub_end_time - sub_start_time
                    if 'cost' not in sub_metrics:
                        sub_metrics['cost'] = float('inf') if not subpath else sum(env.get_cost(subpath[i-1], subpath[i]) for i in range(1, len(subpath)))

                    if subpath and len(subpath) > 1:
                        try:
                            sub_start_idx = subpath.index(prev_pos)
                            new_sub = subpath[sub_start_idx + 1:]
                            if new_sub and new_sub[-1] == env.goal:
                                current_path[t:] = new_sub
                                # Simple added cost: full subpath cost (approximate, ignores exact overlap edges)
                                total_cost += sub_metrics['cost']
                                replan_count += 1
                                log_entries.append(f"Step {t}: Blocked at {pos}. Replanned from {prev_pos}. New sub-path length: {len(new_sub)}")
                                print(f"Replanned: New path length {len(current_path)}, total cost now {total_cost:.1f}")
                            else:
                                print("Replan failed - sub-path doesn't reach goal.")
                                break
                        except ValueError:
                            print("Replan failed - prev_pos not in subpath.")
                            break
                    else:
                        print("Replan failed - no sub-path found.")
                        break

                if t >= len(current_path) - 1:
                    break

            path = current_path
            metrics['cost'] = total_cost
            print(f"Simulation complete: {replan_count} replans. Final Cost: {total_cost:.1f}, Final Path Length: {len(path)}")
            # Log to file (root-relative)
            log_path = 'replan_log.txt'
            with open(log_path, 'w') as f:
                f.write(f"Planner: {planner_name} on {args.map} map\n")
                f.write(f"Replans: {replan_count}\n")
                f.write(f"Final Path: {path}\n")
                for entry in log_entries:
                    f.write(entry + "\n")
            print(f"Replan details saved to {log_path}")
        elif args.replan:
            print("Replanning skipped: Not a dynamic map or no moving obstacles.")

        print("Delivery successful!")
    else:
        print("No path found. Cost: inf")
        path = []
        metrics = {'cost': float('inf'), 'nodes': 0, 'time': end_time - start_time}

    # Plotting if requested (root-relative save)
    if args.plot and path:
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Prepare grid for plotting
        plot_grid = np.copy(env.grid)
        plot_grid[np.isinf(plot_grid)] = 10  # Obstacles high
        plot_grid[plot_grid == -1] = 0  # Walls low
        plot_grid = np.maximum(plot_grid, 1)  # Min 1
        
        im = ax.imshow(plot_grid, cmap='Blues', origin='upper', vmin=0, vmax=10)
        plt.colorbar(im, ax=ax, label='Terrain Cost (1=free, 10=obstacle)')
        
        if len(path) > 1:
            path_x, path_y = zip(*path)
            ax.plot(path_y, path_x, 'r-', linewidth=3, label=f'Path (len={len(path)}, cost={metrics["cost"]:.1f})', marker='o', markersize=4)
        
        ax.plot(env.start[1], env.start[0], 'go', markersize=12, label='Start', markeredgecolor='k', markeredgewidth=1)
        ax.plot(env.goal[1], env.goal[0], 'ro', markersize=12, label='Goal', markeredgecolor='k', markeredgewidth=1)
        
        if hasattr(env, 'moving_obstacles') and env.moving_obstacles:
            for obs_id, obs_data in env.moving_obstacles.items():
                if obs_data.get('path'):
                    veh_pos = obs_data['path'][0]
                    ax.plot(veh_pos[1], veh_pos[0], 'k^', markersize=10, label=f'{obs_id} at t=0', markeredgecolor='y')
        
        ax.set_title(f'{planner_name} Path on {args.map.capitalize()} Map\n(Cost: {metrics["cost"]:.1f}, Nodes: {metrics["nodes"]}, Time: {metrics["time"]:.3f}s)')
        ax.set_xlabel('Column (Y)')
        ax.set_ylabel('Row (X)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save (root-relative)
        plot_filename = f'path_{args.planner}_{args.map}.png'
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"Plot saved as: {plot_filename}")
        plt.show()
    elif args.plot:
        print("No path found - skipping plot.")

if __name__ == '__main__':
    main()