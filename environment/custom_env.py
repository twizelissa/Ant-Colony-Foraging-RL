import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import math
import json

class AntFarmEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.window_size = (800, 600)
        
        # We have 3 food sources
        self.num_sources = 3
        # Actions: 0=Boost P1, 1=Boost P2, 2=Boost P3, 3=Do nothing
        self.action_space = spaces.Discrete(self.num_sources + 1)
        
        # State:
        # [0:3] Food remaining (normalized)
        # [3:6] Pheromone levels (normalized)
        # [6:9] Distance (normalized, fixed but good for context)
        # [9]   Time remaining (normalized)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(10,), dtype=np.float32
        )

        self.max_steps = 1000
        self.distances = np.array([150.0, 300.0, 450.0]) # Distance from nest
        self.food_capacities = np.array([50.0, 100.0, 200.0])
        
        self.ant_speed = 5.0
        self.num_ants = 40
        self.decay_rate = 0.99
        self.export_api = True
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.food_remaining = np.copy(self.food_capacities)
        self.pheromones = np.ones(self.num_sources) * 0.1
        self.total_collected = 0
        
        self.ants = []
        for _ in range(self.num_ants):
            self.ants.append({
                "path": -1, # -1 means at nest making decision
                "progress": 0.0,
                "has_food": False
            })
            
        if self.render_mode == "human":
            self._init_pygame()
            
        return self._get_obs(), {}

    def _get_obs(self):
        obs = np.zeros(10, dtype=np.float32)
        obs[0:3] = self.food_remaining / self.food_capacities
        obs[3:6] = np.clip(self.pheromones, 0.0, 1.0)
        obs[6:9] = self.distances / 500.0
        obs[9] = self.current_step / self.max_steps
        return obs

    def step(self, action):
        self.current_step += 1
        
        # Apply action (boost pheromones)
        if action < self.num_sources:
            self.pheromones[action] = min(1.0, self.pheromones[action] + 0.2)
            energy_penalty = 0.05
        else:
            energy_penalty = 0.0
            
        # Pheromones naturally decay
        self.pheromones *= self.decay_rate
        
        reward = 0.0 - energy_penalty
        
        # Simulate ants
        for ant in self.ants:
            if ant["path"] == -1: # At nest, pick a path
                # Probabilities based on pheromones + small random exploration
                probs = self.pheromones + 0.1
                # If food is empty, dramatically lower probability to avoid wasting time
                empty_mask = self.food_remaining <= 0
                probs[empty_mask] = 0.01 
                
                probs /= probs.sum()
                ant["path"] = np.random.choice(self.num_sources, p=probs)
                ant["progress"] = 0.0
                ant["has_food"] = False
                
            else:
                path_idx = ant["path"]
                dist = self.distances[path_idx]
                
                # Move
                ant["progress"] += self.ant_speed
                
                # Reached food source?
                if not ant["has_food"] and ant["progress"] >= dist:
                    if self.food_remaining[path_idx] > 0:
                        self.food_remaining[path_idx] -= 1
                        ant["has_food"] = True
                    ant["progress"] = dist # Cap at distance, start heading back
                    
                # Reach nest? (has food or found empty pile)
                if ant["progress"] >= dist * 2: # Completed round trip
                    if ant["has_food"]:
                        self.total_collected += 1
                        # Dense reward: +1 for bringing food back, adjusted by efficiency (shorter path = faster base)
                        reward += 1.0 
                    ant["path"] = -1 # Ready for next task
                    
        # Terminal condition
        done = self.current_step >= self.max_steps or np.sum(self.food_remaining) <= 0
        
        if getattr(self, 'export_api', False):
            state_data = {
                "step": self.current_step,
                "food": self.food_remaining.tolist(),
                "pheromones": self.pheromones.tolist(),
                "ants": [{"path": int(a["path"]), "progress": float(a["progress"]), "has_food": bool(a["has_food"])} for a in self.ants],
                "collected": int(self.total_collected)
            }
            try:
                with open("env_api_stream.json", "w") as f:
                    json.dump(state_data, f)
            except Exception:
                pass
        
        if self.render_mode == "human":
            self.render()
            
        return self._get_obs(), reward, done, False, {"collected": self.total_collected}

    def _init_pygame(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Ant Colony Optimizer")
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def render(self):
        if self.render_mode == "human":
            self._init_pygame()
            
            self.window.fill((30, 30, 30)) # Dark ground
            
            nest_pos = (100, 300)
            
            # Draw nest
            pygame.draw.circle(self.window, (139, 69, 19), nest_pos, 40)
            
            # Draw paths and sources
            source_y = [100, 300, 500]
            for i in range(self.num_sources):
                color_intensity = int(max(0, min(255, self.pheromones[i] * 255)))
                path_color = (0, color_intensity, 0)
                
                end_pos = (nest_pos[0] + int(self.distances[i]), source_y[i])
                
                # Draw trail
                pygame.draw.line(self.window, path_color, nest_pos, end_pos, 4)
                
                # Draw food pile
                food_ratio = self.food_remaining[i] / self.food_capacities[i]
                pile_radius = max(5, int(20 * food_ratio))
                if food_ratio > 0:
                    pygame.draw.circle(self.window, (255, 215, 0), end_pos, pile_radius)
                    
            # Draw ants
            for ant in self.ants:
                if ant["path"] != -1:
                    path_idx = ant["path"]
                    end_pos = (nest_pos[0] + int(self.distances[path_idx]), source_y[path_idx])
                    
                    dist = self.distances[path_idx]
                    if ant["progress"] <= dist: # Going to food
                        ratio = ant["progress"] / dist
                    else: # Returning to nest
                        ratio = 1.0 - ((ant["progress"] - dist) / dist)
                        
                    ant_x = nest_pos[0] + (end_pos[0] - nest_pos[0]) * ratio
                    ant_y = nest_pos[1] + (end_pos[1] - nest_pos[1]) * ratio
                    
                    color = (255, 50, 50) if ant["has_food"] else (200, 200, 200)
                    pygame.draw.circle(self.window, color, (int(ant_x), int(ant_y)), 3)
            
            # UI Text
            font = pygame.font.SysFont(None, 24)
            info = font.render(f"Food Collected: {self.total_collected}", True, (255, 255, 255))
            self.window.blit(info, (10, 10))
            
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
