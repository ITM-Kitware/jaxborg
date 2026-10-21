"""Red profiles from Singh et al., arXiv:2410.17351, section 5.2.

https://arxiv.org/abs/2410.17351
The released Hierarchical-MARL repository imports stock FiniteStateRedAgent;
these profiles implement the probability changes described in the paper.
Host selection, transitions, discovery, deception and withdrawal stay stock.
"""

HMARL_REDS = ("fsm", "aggressive", "stealthy", "impact")

# State names, source action column, destination action column. Columns follow
# CC4's FSM: discover, aggressive, stealth, deception, exploit, privesc,
# impact, degrade, withdraw. Move the entire source probability to destination.
HMARL_RED_TRANSFERS = {
    "aggressive": (("K", "KD"), 2, 1),
    "stealthy": (("K", "KD"), 1, 2),
    "impact": (("R", "RD"), 7, 6),
}
