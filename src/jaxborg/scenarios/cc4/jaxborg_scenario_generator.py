"""EnterpriseScenarioGenerator with op-zone server count fixed for CIA mode."""

from __future__ import annotations

from CybORG.Simulator.Scenarios.EnterpriseScenarioGenerator import EnterpriseScenarioGenerator

OP_ZONE_SUBNET_NAMES = frozenset({"operational_zone_a_subnet", "operational_zone_b_subnet"})


class JaxborgScenarioGenerator(EnterpriseScenarioGenerator):
    """Forces op-zone subnets to a fixed server count when op_zone_servers is set."""

    def __init__(self, *args, op_zone_servers: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.op_zone_servers = op_zone_servers

    def _generate_hosts(self, subnets):
        if self.op_zone_servers is None:
            return super()._generate_hosts(subnets)

        host_list = []
        for subnet in subnets.values():
            ip_addresses = list(subnet.cidr.hosts())

            if subnet.name == "internet_subnet":
                hostname = "root_internet_host_0"
                subnet.hosts.append(hostname)
                idx = self.np_random.choice(len(ip_addresses))
                ip = ip_addresses.pop(idx)
                subnet.ip_addresses.append(ip)
                host_list.append(self._generate_linux_host(hostname, ip, subnet))
                subnet.size = 1
                continue

            hostname = f"{subnet.name}_router"
            subnet.hosts.append(hostname)
            idx = self.np_random.choice(len(ip_addresses))
            ip = ip_addresses.pop(idx)
            subnet.ip_addresses.append(ip)
            host_list.append(self._generate_linux_host(hostname, ip, subnet))

            num_user_hosts = self.np_random.integers(self.MIN_USER_HOSTS, self.MAX_USER_HOSTS, endpoint=True)
            for i in range(num_user_hosts):
                hostname = f"{subnet.name}_user_host_{i}"
                subnet.hosts.append(hostname)
                idx = self.np_random.choice(len(ip_addresses))
                ip = ip_addresses.pop(idx)
                subnet.ip_addresses.append(ip)
                host_list.append(self._generate_linux_host(hostname, ip, subnet))

            if subnet.name in OP_ZONE_SUBNET_NAMES:
                num_server_hosts = self.op_zone_servers
            else:
                num_server_hosts = self.np_random.integers(self.MIN_SERVER_HOSTS, self.MAX_SERVER_HOSTS, endpoint=True)
            for i in range(num_server_hosts):
                hostname = f"{subnet.name}_server_host_{i}"
                ip = ip_addresses.pop()
                subnet.ip_addresses.append(ip)
                host_list.append(self._generate_linux_host(hostname, ip, subnet))
                subnet.hosts.append(hostname)

            subnet.size = num_user_hosts + num_server_hosts + 1

        return {host.hostname: host for host in host_list}
