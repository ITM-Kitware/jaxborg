from jaxborg.scenarios.config import ScenarioConfig

CC4_CONFIG = ScenarioConfig(
    num_hosts=137,
    num_subnets=9,
    num_blue_agents=5,
    num_red_agents=6,
    num_services=5,
    num_decoy_types=4,
    mission_phases=3,
    max_steps=500,
    message_length=8,
    max_detection_randoms=1024,
    num_green_random_fields=8,
    num_red_policy_random_fields=3,
    min_hosts_per_subnet=4,
    max_hosts_per_subnet=17,
    max_user_hosts=10,
    max_server_hosts=6,
    blue_max_observed_subnets=3,
    # Peak observed: 33 with random blue over 50 seeds × 500 steps. 34 gives ~3% headroom.
    max_tracked_session_pids=34,
    max_tracked_suspicious_pids=34,
)

GLOBAL_MAX_HOSTS = CC4_CONFIG.num_hosts
NUM_SUBNETS = CC4_CONFIG.num_subnets
NUM_BLUE_AGENTS = CC4_CONFIG.num_blue_agents
NUM_RED_AGENTS = CC4_CONFIG.num_red_agents
NUM_SERVICES = CC4_CONFIG.num_services
NUM_DECOY_TYPES = CC4_CONFIG.num_decoy_types
MISSION_PHASES = CC4_CONFIG.mission_phases
MESSAGE_LENGTH = CC4_CONFIG.message_length
MAX_STEPS = CC4_CONFIG.max_steps
TOTAL_ACTION_ACTOR_SLOTS = CC4_CONFIG.total_action_actor_slots

SUBNET_NAMES = [
    "RESTRICTED_ZONE_A",
    "RESTRICTED_ZONE_B",
    "OPERATIONAL_ZONE_A",
    "OPERATIONAL_ZONE_B",
    "CONTRACTOR_NETWORK",
    "ADMIN_NETWORK",
    "OFFICE_NETWORK",
    "PUBLIC_ACCESS_ZONE",
    "INTERNET",
]

SUBNET_IDS = {name: i for i, name in enumerate(SUBNET_NAMES)}

CYBORG_SUBNET_SUFFIX = {
    "RESTRICTED_ZONE_A": "restricted_zone_a_subnet",
    "RESTRICTED_ZONE_B": "restricted_zone_b_subnet",
    "OPERATIONAL_ZONE_A": "operational_zone_a_subnet",
    "OPERATIONAL_ZONE_B": "operational_zone_b_subnet",
    "CONTRACTOR_NETWORK": "contractor_network_subnet",
    "ADMIN_NETWORK": "admin_network_subnet",
    "OFFICE_NETWORK": "office_network_subnet",
    "PUBLIC_ACCESS_ZONE": "public_access_zone_subnet",
    "INTERNET": "internet_subnet",
}

CYBORG_SUFFIX_TO_ID = {v: SUBNET_IDS[k] for k, v in CYBORG_SUBNET_SUFFIX.items()}

SERVICE_NAMES = ["SSHD", "APACHE2", "MYSQLD", "SMTP", "OTSERVICE"]
SERVICE_IDS = {name: i for i, name in enumerate(SERVICE_NAMES)}

DECOY_NAMES = ["HarakaSMPT", "Apache", "Tomcat", "Vsftpd"]
DECOY_IDS = {name: i for i, name in enumerate(DECOY_NAMES)}

COMPROMISE_NONE = 0
COMPROMISE_USER = 1
COMPROMISE_PRIVILEGED = 2

ACTIVITY_NONE = 0
ACTIVITY_SCAN = 1
ACTIVITY_EXPLOIT = 2

MAX_DETECTION_RANDOMS = CC4_CONFIG.max_detection_randoms
NUM_GREEN_RANDOM_FIELDS = CC4_CONFIG.num_green_random_fields
NUM_RED_POLICY_RANDOM_FIELDS = CC4_CONFIG.num_red_policy_random_fields

MIN_HOSTS_PER_SUBNET = CC4_CONFIG.min_hosts_per_subnet
MAX_HOSTS_PER_SUBNET = CC4_CONFIG.max_hosts_per_subnet

MAX_USER_HOSTS = CC4_CONFIG.max_user_hosts
MAX_SERVER_HOSTS = CC4_CONFIG.max_server_hosts
OBS_HOSTS_PER_SUBNET = CC4_CONFIG.obs_hosts_per_subnet
OBS_VECTOR_HOSTS_PER_SUBNET = CC4_CONFIG.obs_vector_hosts_per_subnet
ACTION_HOST_SLOTS = CC4_CONFIG.action_host_slots
BLUE_MAX_OBSERVED_SUBNETS = CC4_CONFIG.blue_max_observed_subnets
BLUE_ACTION_HOST_SLOTS = CC4_CONFIG.blue_action_host_slots
BLUE_TRAFFIC_SLOTS = CC4_CONFIG.blue_traffic_slots
NUM_MESSAGES = CC4_CONFIG.num_messages
MAX_TRACKED_SESSION_PIDS = CC4_CONFIG.max_tracked_session_pids
MAX_TRACKED_SUSPICIOUS_PIDS = CC4_CONFIG.max_tracked_suspicious_pids
ABSTRACT_RANK_NONE = 1_000_000
BLUE_OBS_SIZE = CC4_CONFIG.blue_obs_size
