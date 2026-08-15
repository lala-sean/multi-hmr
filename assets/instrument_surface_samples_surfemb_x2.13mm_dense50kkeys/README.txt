SurfEmb instrument surface samples.
Gripper static rule: points_part_m[:,0] < 0.002130000 m (2.130 mm) => effective_part_ids=2 and no gripper rotation.
Wrist and each gripper are sampled at one third of the previous 50k per-part dense default.
Shaft is sampled with area weights biased toward the distal x top region.
Visible projection visualizations use full mesh-triangle depth buffer plus front-facing normal filtering.
shaft: n=80000, original_part_id=1, effective_wrist_static=0, moving=80000, sampling=x_biased_top_start=0.187795667m top_len=0.030m rear_density=0.120, bounds=[[-0.26112099999999994, -0.007934, -0.007938], [0.21779600000000002, 0.007934, 0.007938]]
wrist: n=50000, original_part_id=2, effective_wrist_static=0, moving=50000, sampling=uniform_area, bounds=[[-0.00302, -0.0032, -0.0032], [0.011331, 0.0032, 0.0032]]
l_gripper: n=50000, original_part_id=3, effective_wrist_static=23052, moving=26948, sampling=uniform_area, bounds=[[-0.0024129999999999993, -0.002413, -0.001031], [0.009651999999999999, 0.002412, 0.001969]]
r_gripper: n=50000, original_part_id=4, effective_wrist_static=23221, moving=26779, sampling=uniform_area, bounds=[[-0.0024129999999999993, -0.002412, -0.001994], [0.009651999999999999, 0.002413, 0.001005]]
