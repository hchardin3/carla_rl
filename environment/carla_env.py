import carla
import gym
import numpy as np
import matplotlib.pyplot as plt

class CarlaEnv(gym.Env):
    def __init__(self, config: dict):
        """
        Initialize the Carla environment with the given configuration parameters.

        Parameters:
            config (dict): A dictionary containing the following keys:
                'host' (str): CARLA server host.
                'port' (int): CARLA server port.
                'max_acc' (float): Maximum forward acceleration.
                'back_acc' (float): Maximum backward acceleration.
                'max_decc' (float): Maximum braking.
                'max_wheel_angle' (float): Maximum steering angle.
                'town' (str): The CARLA town to load.
                'weather' (str): The weather preset to use.
                'vehicle_model' (str): The vehicle model to spawn.
                'spawn_point' (int): The spawn point index to use for the vehicle.
                'sensors' (dict): Configuration for the RGB camera and other sensors.
                'render' (bool): Whether to render the environment visually.
        """
        # Server connection
        self.client = carla.Client(config['host'], config['port'])
        self.client.set_timeout(config.get('timeout', 10.0))  # Default timeout is 10 seconds
        self.world = self.client.load_world(config['town'])  # Load the specified town

        # Weather
        weather_presets = {
            "ClearNoon": carla.WeatherParameters.ClearNoon,
            "WetNoon": carla.WeatherParameters.WetNoon,
            "WetSunset": carla.WeatherParameters.WetSunset,
            "CloudyNoon": carla.WeatherParameters.CloudyNoon,
            "HardRainNoon": carla.WeatherParameters.HardRainNoon,
        }
        self.world.set_weather(weather_presets.get(config['weather'], carla.WeatherParameters.ClearNoon))

        # Vehicle parameters
        self.vehicle_model = config['vehicle_model']
        self.spawn_point_index = config['spawn_point']

        # Action space and observation space
        self.max_acc = config['max_acc']
        self.back_acc = config['back_acc']
        self.max_decc = config['max_decc']
        self.max_wheel_angle = config['max_wheel_angle']

        # Define the action space: [throttle, brake, steer]
        self.action_space = gym.spaces.Box(
            low=np.array([-self.back_acc, 0, -self.max_wheel_angle]),
            high=np.array([self.max_acc, self.max_decc, self.max_wheel_angle]),
            dtype=np.float32
        )

        # Sensors configuration
        self.rgb_camera_config = config['sensors']['rgb_camera']
        self.collision_sensor_enabled = config['sensors']['collision_sensor']['enabled']
        self.render_enabled = config['render']

        # Observation space (84x84 RGB image)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.rgb_camera_config['image_size_x'], self.rgb_camera_config['image_size_y'], 3), dtype=np.uint8)

        # Initialize variables for vehicle and sensors
        self.vehicle = None
        self.rgb_camera = None
        self.rgb_image = None

    def reset(self):
        # Respawn the agent at a designated starting point
        blueprint_library = self.world.get_blueprint_library()
        vehicle_blueprint = blueprint_library.find(self.vehicle_model)
        
        # Spawn at the specified or random spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[self.spawn_point_index] if self.spawn_point_index < len(spawn_points) else np.random.choice(spawn_points)
        
        if self.vehicle:
            self.vehicle.destroy()
        self.vehicle = self.world.spawn_actor(vehicle_blueprint, spawn_point)
        
        # Attach the RGB camera to the vehicle
        self.rgb_camera = self.setup_rgb_camera(self.vehicle)

        # Return the first observation
        return self.get_observation()

    def step(self, action):
        throttle = action[0]
        brake = action[1]
        steer = action[2]

        # Create a CARLA VehicleControl object to apply the action
        control = carla.VehicleControl()

        if brake > 0:
            control.brake = brake / self.max_decc
            control.throttle = 0
        else:
            if throttle > 0:
                control.throttle = throttle / self.max_acc
            elif throttle < 0:
                control.reverse = True
                control.throttle = -throttle / self.back_acc
            control.brake = 0

        # Apply steering
        control.steer = steer / self.max_wheel_angle

        self.vehicle.apply_control(control)

        # Get new observation, reward, done, and info
        next_observation = self.get_observation()
        reward = self.compute_reward()
        done = self.is_done()
        info = {}

        return next_observation, reward, done, info

    def setup_rgb_camera(self, vehicle):
        blueprint_library = self.world.get_blueprint_library()
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        
        # Set camera attributes from configuration
        camera_bp.set_attribute('image_size_x', str(self.rgb_camera_config['image_size_x']))
        camera_bp.set_attribute('image_size_y', str(self.rgb_camera_config['image_size_y']))
        camera_bp.set_attribute('fov', str(self.rgb_camera_config['fov']))
        
        # Position the camera
        spawn_point = carla.Transform(
            carla.Location(x=self.rgb_camera_config['position']['x'], z=self.rgb_camera_config['position']['z'])
        )
        
        camera = self.world.spawn_actor(camera_bp, spawn_point, attach_to=vehicle)
        camera.listen(lambda image: self.process_image(image))
        
        return camera

    def process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))  # CARLA uses BGRA format
        self.rgb_image = array[:, :, :3]  # Take only the RGB channels

    def get_observation(self):
        # Return the latest RGB image captured by the camera
        if self.rgb_image is None:
            return np.zeros((self.rgb_camera_config['image_size_y'], self.rgb_camera_config['image_size_x'], 3), dtype=np.uint8)
        return self.rgb_image

    def compute_reward(self):
        # Placeholder reward function
        # TO BE COMPLETED
        return 1.0

    def is_done(self):
        # Placeholder done function
        # TO BE COMPLETED
        return False

    def render(self, mode='human'):
        if not self.render_enabled or self.rgb_image is None:
            return

        if mode == 'human':
            plt.imshow(self.rgb_image)
            plt.axis('off')
            plt.show()

    def close(self):
        if self.rgb_camera is not None:
            self.rgb_camera.stop()
            self.rgb_camera.destroy()

        if self.vehicle is not None:
            self.vehicle.destroy()

        # Clean up other actors
        self.client.apply_batch([carla.command.DestroyActor(x) for x in [self.vehicle, self.rgb_camera]])
        self.vehicle = None
        self.rgb_camera = None
