# Description
This is a project to generate positions: (x, y) and orientations : degree for a SmolVLA pick&place task's training and evaluation dataset.

# Sampling theory
Poisson disk

# Main script
position_sampling_v5.py


# Config to fit the specific scene
master_seed,
CUBE_SPEC,
BOX_SPEC,
CROSS_OBJECT_MIN_DIST,
CROSS_CUBE_MIN_DIST,
box_anchors in D0a,
cube_anchors in D0b,
OUTER_CUBE_SPEC，


# Scene types
## D0
1 object (cube), 1 destination (box), both sample based on the poisson disk distribution
## D0a-cubeOnly
1 object (cube) samples based on the poisson disk distribution, fixed 1 destination (box)
## D0b-boxOnly
fixed 1 object (cube), 1 destination (box) based on the poisson disk distribution
## OOD
orientation or position OOD. check script position_sampling_ood.py