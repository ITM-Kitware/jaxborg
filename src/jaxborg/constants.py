GLOBAL_MAX_HOSTS = 137
NUM_SUBNETS = 9
NUM_BLUE_AGENTS = 5
NUM_RED_AGENTS = 6
NUM_SERVICES = 5
NUM_DECOY_TYPES = 4
MISSION_PHASES = 3
MESSAGE_LENGTH = 8
MAX_STEPS = 500

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

MAX_DETECTION_RANDOMS = 1024
NUM_GREEN_RANDOM_FIELDS = 8
NUM_RED_POLICY_RANDOM_FIELDS = 3

MIN_HOSTS_PER_SUBNET = 4
MAX_HOSTS_PER_SUBNET = 17

MAX_USER_HOSTS = 10
MAX_SERVER_HOSTS = 6
# Hosts per subnet in obs_host_map — includes router for action encoding.
OBS_HOSTS_PER_SUBNET = MAX_USER_HOSTS + MAX_SERVER_HOSTS + 1
# Hosts per subnet in the observation vector — excludes router (matches CybORG BlueFlatWrapper).
OBS_VECTOR_HOSTS_PER_SUBNET = MAX_USER_HOSTS + MAX_SERVER_HOSTS
ACTION_HOST_SLOTS = NUM_SUBNETS * OBS_HOSTS_PER_SUBNET
NUM_MESSAGES = NUM_BLUE_AGENTS - 1
# PID identity slots used for red session PID tracking.
# Measured peak: 13 per (agent, host) across 10 random-action CC4 episodes.
# 32 provides 2.5x safety margin and aligns to GPU cache lines.
MAX_TRACKED_SESSION_PIDS = 16
# PID identity slots used for blue suspicious PID memory.
# Measured peak: 13 per (agent, host); proc_creation peak: 1.
MAX_TRACKED_SUSPICIOUS_PIDS = 16
ABSTRACT_RANK_NONE = 1_000_000
BLUE_OBS_SIZE = (
    1
    + 3 * (NUM_SUBNETS + NUM_SUBNETS + NUM_SUBNETS + OBS_VECTOR_HOSTS_PER_SUBNET + OBS_VECTOR_HOSTS_PER_SUBNET)
    + NUM_MESSAGES * MESSAGE_LENGTH
)
